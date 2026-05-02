import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Dict, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment-dependent
    np = None

from config import settings

try:
    from sqlalchemy import delete, func, select
    from database import AsyncSessionLocal, CacheEntry
except Exception:  # pragma: no cover - environment-dependent
    delete = func = select = None
    AsyncSessionLocal = None
    CacheEntry = None

try:
    import faiss
except ImportError:  # pragma: no cover - environment-dependent
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - environment-dependent
    SentenceTransformer = None


class SemanticCacheService:
    def __init__(self) -> None:
        self.dimension = 384
        self.threshold = float(settings.CACHE_SIMILARITY_THRESHOLD)
        self.index_path = Path(f"{settings.FAISS_INDEX_PATH}.index")
        self.model = None
        self.index = None
        self.entry_ids = []
        self.query_count = 0
        self.hit_count = 0
        self.cost_saved_usd = 0.0
        self.initialized = False
        self._lock = asyncio.Lock()
        self._memory_entries: Dict[str, Dict[str, object]] = {}
        self._memory_vectors: Dict[str, list] = {}

    async def initialize(self) -> None:
        async with self._lock:
            if self.initialized:
                return
            self.model = self._load_embedding_model()
            self.index = self._load_or_create_index()
            await self._load_metadata_from_postgres()
            self.initialized = True

    def _load_embedding_model(self):
        if SentenceTransformer is None:
            return None
        return SentenceTransformer("all-MiniLM-L6-v2")

    def _load_or_create_index(self):
        if faiss is None:
            return None
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            try:
                return faiss.read_index(str(self.index_path))
            except Exception:
                pass
        return faiss.IndexFlatL2(self.dimension)

    def _embed_text(self, text: str):
        if self.model is not None:
            emb = self.model.encode([text], normalize_embeddings=True)[0]
            if np is not None:
                return np.asarray(emb, dtype="float32")
            return [float(x) for x in emb]
        # Deterministic fallback when sentence-transformers is unavailable.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        if np is not None:
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=self.dimension).astype("float32")
            norm = np.linalg.norm(vector)
            return vector / norm if norm else vector
        # Pure Python deterministic embedding fallback.
        values = []
        state = seed
        for _ in range(self.dimension):
            state = (1664525 * state + 1013904223) % (2**32)
            values.append((state / 2**32) * 2.0 - 1.0)
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values] if norm else values

    @staticmethod
    def _intent_ttl(intent: str) -> timedelta:
        mapping = {
            "code": timedelta(hours=24),
            "math": timedelta(days=7),
            "factual": timedelta(hours=48),
            "creative": timedelta(hours=12),
        }
        return mapping.get(intent, timedelta(hours=24))

    @staticmethod
    def _to_similarity(distance: float) -> float:
        similarity = 1.0 - (distance / 2.0)
        return max(0.0, min(1.0, float(similarity)))

    @staticmethod
    def _cosine_similarity(v1: list, v2: list) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    async def _load_metadata_from_postgres(self) -> None:
        self.entry_ids = []
        if self.index is None:
            return
        if AsyncSessionLocal is None or CacheEntry is None or select is None:
            for entry_id, row in self._memory_entries.items():
                self.entry_ids.append(entry_id)
                vector = self._embed_text(row["prompt_text"])
                self._memory_vectors[entry_id] = vector if isinstance(vector, list) else vector.tolist()
                if np is not None and faiss is not None:
                    self.index.add(vector.reshape(1, -1).astype("float32"))
            return

        now = datetime.now(timezone.utc)
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(CacheEntry).where(CacheEntry.expires_at > now).order_by(CacheEntry.created_at.asc())
                )
                rows = result.scalars().all()
        except Exception:
            return

        vectors = []
        for row in rows:
            vector = self._embed_text(row.prompt_text)
            vectors.append(vector)
            self.entry_ids.append(str(row.id))
            self._memory_vectors[str(row.id)] = vector if isinstance(vector, list) else vector.tolist()
        if vectors and np is not None and faiss is not None:
            self.index.reset()
            self.index.add(np.vstack(vectors).astype("float32"))

    async def get_cached_response(self, text: str) -> Optional[Dict[str, object]]:
        await self.initialize()
        self.query_count += 1
        if self.query_count % 100 == 0:
            await self.evict_low_quality_entries()
        query_vector = self._embed_text(text)
        if self.index is not None and faiss is not None and np is not None and self.index.ntotal > 0:
            distances, indices = self.index.search(query_vector.reshape(1, -1), 1)
            idx = int(indices[0][0])
            if idx < 0 or idx >= len(self.entry_ids):
                return None
            entry_id = self.entry_ids[idx]
            similarity = self._to_similarity(float(distances[0][0]))
        else:
            if not self._memory_vectors:
                return None
            best_id = None
            best_sim = -1.0
            query_list = query_vector if isinstance(query_vector, list) else query_vector.tolist()
            for entry_id, vector in self._memory_vectors.items():
                sim = self._cosine_similarity(query_list, vector)
                if sim > best_sim:
                    best_sim = sim
                    best_id = entry_id
            if best_id is None:
                return None
            entry_id = best_id
            similarity = best_sim
        if similarity < self.threshold:
            return None
        self.hit_count += 1
        self.cost_saved_usd += float(settings.GROQ_MIXTRAL_COST)

        if AsyncSessionLocal is None or CacheEntry is None or select is None:
            entry = self._memory_entries.get(entry_id)
            if not entry:
                return None
            entry["hit_count"] += 1
            return {
                "response_text": entry["response_text"],
                "similarity_score": round(similarity, 4),
                "cache_entry_id": entry_id,
            }

        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(CacheEntry).where(CacheEntry.id == entry_id))
                entry = result.scalar_one_or_none()
                if entry is None:
                    return None
                entry.hit_count += 1
                await session.commit()
                return {
                    "response_text": entry.response_text,
                    "similarity_score": round(similarity, 4),
                    "cache_entry_id": str(entry.id),
                }
        except Exception:
            return None

    async def store_in_cache(self, text: str, response: str, quality: float, intent: str) -> bool:
        await self.initialize()
        if quality <= 0.70:
            return False

        now = datetime.now(timezone.utc)
        expires_at = now + self._intent_ttl(intent)
        prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        vector = self._embed_text(text)

        if AsyncSessionLocal is None or CacheEntry is None:
            entry_id = str(uuid.uuid4())
            self._memory_entries[entry_id] = {
                "prompt_embedding_hash": prompt_hash,
                "prompt_text": text,
                "response_text": response,
                "quality_score": float(quality),
                "hit_count": 0,
                "intent": intent,
                "created_at": now,
                "expires_at": expires_at,
            }
        else:
            try:
                async with AsyncSessionLocal() as session:
                    entry = CacheEntry(
                        prompt_embedding_hash=prompt_hash,
                        prompt_text=text,
                        response_text=response,
                        quality_score=float(quality),
                        hit_count=0,
                        intent=intent,
                        created_at=now,
                        expires_at=expires_at,
                    )
                    session.add(entry)
                    await session.flush()
                    entry_id = str(entry.id)
                    await session.commit()
            except Exception:
                return False

        if self.index is not None and faiss is not None and np is not None:
            self.index.add(vector.reshape(1, -1).astype("float32"))
            self.entry_ids.append(entry_id)
            if faiss is not None:
                try:
                    faiss.write_index(self.index, str(self.index_path))
                except Exception:
                    pass
        self._memory_vectors[entry_id] = vector if isinstance(vector, list) else vector.tolist()
        if entry_id not in self.entry_ids:
            self.entry_ids.append(entry_id)
        return True

    async def evict_low_quality_entries(self) -> int:
        await self.initialize()
        now = datetime.now(timezone.utc)
        ids_to_remove = []

        if AsyncSessionLocal is None or CacheEntry is None or select is None or delete is None:
            for entry_id, row in list(self._memory_entries.items()):
                if row["quality_score"] < 0.50 or row["expires_at"] <= now:
                    ids_to_remove.append(entry_id)
                    self._memory_entries.pop(entry_id, None)
        else:
            try:
                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(CacheEntry.id).where((CacheEntry.quality_score < 0.50) | (CacheEntry.expires_at <= now))
                    )
                    ids_to_remove = [str(row[0]) for row in result.all()]
                    if ids_to_remove:
                        await session.execute(delete(CacheEntry).where(CacheEntry.id.in_(ids_to_remove)))
                        await session.commit()
            except Exception:
                ids_to_remove = []

        if self.index is not None and ids_to_remove:
            await self._load_metadata_from_postgres()
            if faiss is not None:
                try:
                    faiss.write_index(self.index, str(self.index_path))
                except Exception:
                    pass
        for entry_id in ids_to_remove:
            self._memory_vectors.pop(entry_id, None)
        return len(ids_to_remove)

    async def get_cache_stats(self) -> Dict[str, object]:
        await self.initialize()
        if AsyncSessionLocal is None or CacheEntry is None or select is None or func is None:
            total_entries = len(self._memory_entries)
            avg_quality = (
                sum(float(e["quality_score"]) for e in self._memory_entries.values()) / total_entries
                if total_entries
                else 0.0
            )
            entries_by_intent: Dict[str, int] = {}
            for row in self._memory_entries.values():
                entries_by_intent[row["intent"]] = entries_by_intent.get(row["intent"], 0) + 1
        else:
            total_entries = 0
            avg_quality = 0.0
            entries_by_intent = {}
            try:
                async with AsyncSessionLocal() as session:
                    total_entries = int((await session.execute(select(func.count(CacheEntry.id)))).scalar_one() or 0)
                    avg_quality = float(
                        (await session.execute(select(func.avg(CacheEntry.quality_score)))).scalar_one() or 0.0
                    )
                    grouped = await session.execute(
                        select(CacheEntry.intent, func.count(CacheEntry.id)).group_by(CacheEntry.intent)
                    )
                    entries_by_intent = {row[0]: int(row[1]) for row in grouped.all()}
            except Exception:
                pass

        hit_rate = (self.hit_count / self.query_count * 100.0) if self.query_count else 0.0
        return {
            "total_entries": total_entries,
            "hit_rate": round(hit_rate, 2),
            "avg_quality": round(avg_quality, 4),
            "entries_by_intent": entries_by_intent,
            "estimated_cost_saved": round(self.cost_saved_usd, 6),
        }


cache_service = SemanticCacheService()


async def get_cached_response(text: str) -> Optional[Dict[str, object]]:
    return await cache_service.get_cached_response(text)


async def store_in_cache(text: str, response: str, quality: float, intent: str) -> bool:
    return await cache_service.store_in_cache(text=text, response=response, quality=quality, intent=intent)


async def evict_low_quality_entries() -> int:
    return await cache_service.evict_low_quality_entries()


async def get_cache_stats() -> Dict[str, object]:
    return await cache_service.get_cache_stats()
