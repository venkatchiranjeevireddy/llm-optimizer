import json
import pickle
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import func, select

from config import settings
from database import AsyncSessionLocal, QueryLog, RouterTrainingLog
from services.cost_service import calculate_baseline_cost
from services.router_service import describe_router_engine

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    accuracy_score = None
    train_test_split = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


MODEL_RANK = {"groq_simple": 0, "groq_mixtral": 1, "gemini_pro": 2}
MODEL_UPGRADE = {"groq_simple": "groq_mixtral", "groq_mixtral": "gemini_pro", "gemini_pro": "gemini_pro"}
FEATURE_ORDER = [
    "prompt_length",
    "token_original",
    "has_code",
    "has_math",
    "has_reasoning",
    "question_count",
    "sentence_count",
    "avg_word_length",
    "has_creative",
]


def _models_root() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


def _model_file_path() -> Path:
    return _models_root() / "router_model.pkl"


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


async def log_query_result(query_data: Dict[str, object]) -> bool:
    try:
        async with AsyncSessionLocal() as session:
            row = QueryLog(
                id=query_data.get("id", uuid.uuid4()),
                prompt_original=str(query_data.get("prompt_original", "")),
                prompt_compressed=str(query_data.get("prompt_compressed", "")),
                prompt_length=_safe_int(query_data.get("prompt_length"), 0),
                token_original=_safe_int(query_data.get("token_original"), 0),
                token_compressed=_safe_int(query_data.get("token_compressed"), 0),
                tokens_saved=_safe_int(query_data.get("tokens_saved"), 0),
                has_code=bool(query_data.get("has_code", False)),
                has_math=bool(query_data.get("has_math", False)),
                has_reasoning=bool(query_data.get("has_reasoning", False)),
                question_count=_safe_int(query_data.get("question_count"), 0),
                sentence_count=_safe_int(query_data.get("sentence_count"), 0),
                avg_word_length=_safe_float(query_data.get("avg_word_length"), 0.0),
                has_creative=bool(query_data.get("has_creative", False)),
                intent=str(query_data.get("intent", "factual")),
                routed_to=str(query_data.get("routed_to", "gemini_pro")),
                router_confidence=_safe_float(query_data.get("router_confidence"), 0.0),
                routing_reason=str(query_data.get("routing_reason", "")),
                response_text=str(query_data.get("response_text", "")),
                llm_judge_score=_safe_float(query_data.get("llm_judge_score"), 0.0),
                bleu_score=_safe_float(query_data.get("bleu_score"), 0.0),
                semantic_similarity=_safe_float(query_data.get("semantic_similarity"), 0.0),
                hallucination_flag=bool(query_data.get("hallucination_flag", False)),
                overall_quality_score=_safe_float(query_data.get("overall_quality_score"), 0.0),
                feedback_signal=_safe_int(query_data.get("feedback_signal"), 0),
                latency_ms=_safe_int(query_data.get("latency_ms"), 0),
                cost_usd=_safe_float(query_data.get("cost_usd"), 0.0),
                cache_hit=bool(query_data.get("cache_hit", False)),
                ab_test_group=str(query_data.get("ab_test_group", "A")),
                created_at=query_data.get("created_at", datetime.utcnow()),
            )
            session.add(row)
            await session.commit()
        return True
    except Exception:
        return False


async def check_retrain_trigger() -> Optional[str]:
    async with AsyncSessionLocal() as session:
        total_queries = int((await session.execute(select(func.count(QueryLog.id)))).scalar_one() or 0)
        if total_queries > 0 and total_queries % settings.RETRAIN_EVERY_N_QUERIES == 0:
            return "scheduled_retrain"

        feedback_rows = (
            await session.execute(
                select(QueryLog.feedback_signal).order_by(QueryLog.created_at.desc()).limit(100)
            )
        ).scalars().all()

    if not feedback_rows:
        return None

    negatives = sum(1 for s in feedback_rows if int(s) == -1)
    bad_rate = negatives / len(feedback_rows)
    if bad_rate > settings.BAD_RESPONSE_RATE_THRESHOLD:
        return "emergency_bad_response_rate"
    return None


def _extract_features_from_row(row: QueryLog) -> List[float]:
    return [
        float(_safe_int(row.prompt_length, 0)),
        float(_safe_int(row.token_original, 0)),
        float(1 if row.has_code else 0),
        float(1 if row.has_math else 0),
        float(1 if row.has_reasoning else 0),
        float(_safe_int(row.question_count, 0)),
        float(_safe_int(row.sentence_count, 0)),
        float(_safe_float(row.avg_word_length, 0.0)),
        float(1 if row.has_creative else 0),
    ]


def _current_model_accuracy(rows: List[QueryLog]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        if row.feedback_signal >= 0:
            correct += 1
    return correct / len(rows)


def _build_training_arrays(rows: List[QueryLog]):
    X, y = [], []
    for row in rows:
        if row.feedback_signal == 0:
            continue
        label = str(row.routed_to)
        if row.feedback_signal == -1:
            label = MODEL_UPGRADE.get(label, "gemini_pro")
        X.append(_extract_features_from_row(row))
        y.append(MODEL_RANK.get(label, MODEL_RANK["gemini_pro"]))
    return X, y


async def retrain_router(trigger_reason: str = "manual") -> Dict[str, object]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(QueryLog)
                .where(QueryLog.feedback_signal != 0)
                .order_by(QueryLog.created_at.desc())
                .limit(2000)
            )
        ).scalars().all()

    if not rows:
        return {
            "training_samples": 0,
            "accuracy_before": 0.0,
            "accuracy_after": 0.0,
            "improved": False,
            "trigger_reason": trigger_reason,
            "message": "No eligible samples for retraining.",
        }

    accuracy_before = _current_model_accuracy(rows)
    X, y = _build_training_arrays(rows)
    if len(X) < 20 or len(set(y)) < 2 or np is None or xgb is None or train_test_split is None or accuracy_score is None:
        accuracy_after = accuracy_before
        improved = False
    else:
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y, dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=0.20, random_state=42, stratify=y_np
        )
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy_after = float(accuracy_score(y_test, preds))
        improved = accuracy_after > accuracy_before

        if improved:
            path = _model_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                pickle.dump(model, f)
            # Reload in router_service module state.
            import services.router_service as router_service

            router_service.ROUTER_MODEL = model

    async with AsyncSessionLocal() as session:
        session.add(
            RouterTrainingLog(
                training_samples=len(X),
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                trigger_reason=trigger_reason,
                improved=improved,
                triggered_at=datetime.utcnow(),
            )
        )
        await session.commit()

    return {
        "training_samples": len(X),
        "accuracy_before": round(accuracy_before, 6),
        "accuracy_after": round(accuracy_after, 6),
        "improved": improved,
        "trigger_reason": trigger_reason,
        "model_saved": bool(improved),
    }


async def get_learning_progress() -> List[Dict[str, object]]:
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(select(RouterTrainingLog).order_by(RouterTrainingLog.triggered_at.asc()))
        ).scalars().all()

    return [
        {
            "id": str(row.id),
            "training_samples": row.training_samples,
            "accuracy_before": _safe_float(row.accuracy_before, 0.0),
            "accuracy_after": _safe_float(row.accuracy_after, 0.0),
            "trigger_reason": row.trigger_reason,
            "improved": bool(row.improved),
            "triggered_at": row.triggered_at.isoformat() if row.triggered_at else None,
        }
        for row in rows
    ]


async def get_system_metrics() -> Dict[str, object]:
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(QueryLog))).scalars().all()
        total_queries = len(rows)
        if total_queries == 0:
            return {
                "total_queries": 0,
                "cache_hit_rate": 0.0,
                "avg_cost_per_query": 0.0,
                "avg_latency_ms": 0.0,
                "avg_latency_cache_hit_ms": 0.0,
                "avg_latency_cache_miss_ms": 0.0,
                "model_distribution": {"groq_simple": 0.0, "groq_mixtral": 0.0, "gemini_pro": 0.0},
                "model_cost_totals_usd": {"groq_simple": 0.0, "groq_mixtral": 0.0, "gemini_pro": 0.0},
                "total_cost_usd": 0.0,
                "baseline_cost_usd": 0.0,
                "cost_saved_usd": 0.0,
                "cost_saved_percentage": 0.0,
                "avg_quality_score": 0.0,
                "router_accuracy_trend": [],
                "bad_response_rate": 0.0,
                "daily_budget_used": 0.0,
                "token_original_total": 0,
                "token_compressed_total": 0,
                "tokens_saved_total": 0,
                "router_engine": describe_router_engine(),
            }

        cache_hits = sum(1 for r in rows if r.cache_hit)
        total_cost = sum(_safe_float(r.cost_usd, 0.0) for r in rows)
        # Baseline = hypothetical cost if every logged query used Gemini Flash at same compressed-input + rough output size.
        baseline_cost = 0.0
        token_original_total = token_compressed_total = tokens_saved_total = 0
        model_cost_totals = {"groq_simple": 0.0, "groq_mixtral": 0.0, "gemini_pro": 0.0}
        latency_hit_sum = latency_miss_sum = 0
        hit_count = miss_count = 0

        for r in rows:
            tin = _safe_int(r.token_compressed, 0)
            tout = len(str(r.response_text).split())
            baseline_cost += float(calculate_baseline_cost(tin, tout))
            token_original_total += _safe_int(r.token_original, 0)
            token_compressed_total += _safe_int(r.token_compressed, 0)
            tokens_saved_total += _safe_int(r.tokens_saved, 0)
            rt = str(r.routed_to)
            if rt in model_cost_totals:
                model_cost_totals[rt] += _safe_float(r.cost_usd, 0.0)
            lm = _safe_int(r.latency_ms, 0)
            if r.cache_hit:
                latency_hit_sum += lm
                hit_count += 1
            else:
                latency_miss_sum += lm
                miss_count += 1

        avg_latency = sum(_safe_int(r.latency_ms, 0) for r in rows) / total_queries
        avg_quality = sum(_safe_float(r.overall_quality_score, 0.0) for r in rows) / total_queries
        bad_responses = sum(1 for r in rows if _safe_float(r.overall_quality_score, 0.0) < 0.40)

        model_counts = {"groq_simple": 0, "groq_mixtral": 0, "gemini_pro": 0}
        for r in rows:
            if r.routed_to in model_counts:
                model_counts[r.routed_to] += 1

        trend_rows = (
            await session.execute(select(RouterTrainingLog).order_by(RouterTrainingLog.triggered_at.asc()))
        ).scalars().all()
        router_accuracy_trend = [_safe_float(r.accuracy_after, 0.0) for r in trend_rows]

        today = date.today()
        today_cost = sum(_safe_float(r.cost_usd, 0.0) for r in rows if r.created_at and r.created_at.date() == today)

    avg_latency_hit = (latency_hit_sum / hit_count) if hit_count else 0.0
    avg_latency_miss = (latency_miss_sum / miss_count) if miss_count else avg_latency

    saved_usd = baseline_cost - total_cost
    saved_pct = (saved_usd / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
    daily_budget_used = (today_cost / float(settings.DAILY_BUDGET_USD) * 100.0) if settings.DAILY_BUDGET_USD > 0 else 0.0

    return {
        "total_queries": total_queries,
        "cache_hit_rate": round((cache_hits / total_queries) * 100.0, 4),
        "avg_cost_per_query": round(total_cost / total_queries, 8),
        "avg_latency_ms": round(avg_latency, 4),
        "avg_latency_cache_hit_ms": round(avg_latency_hit, 4),
        "avg_latency_cache_miss_ms": round(avg_latency_miss, 4),
        "model_distribution": {
            "groq_simple": round((model_counts["groq_simple"] / total_queries) * 100.0, 4),
            "groq_mixtral": round((model_counts["groq_mixtral"] / total_queries) * 100.0, 4),
            "gemini_pro": round((model_counts["gemini_pro"] / total_queries) * 100.0, 4),
        },
        "model_cost_totals_usd": {
            k: round(v, 8) for k, v in model_cost_totals.items()
        },
        "total_cost_usd": round(total_cost, 8),
        "baseline_cost_usd": round(baseline_cost, 8),
        "cost_saved_usd": round(saved_usd, 8),
        "cost_saved_percentage": round(saved_pct, 4),
        "avg_quality_score": round(avg_quality, 6),
        "router_accuracy_trend": router_accuracy_trend,
        "bad_response_rate": round((bad_responses / total_queries) * 100.0, 4),
        "daily_budget_used": round(daily_budget_used, 4),
        "token_original_total": token_original_total,
        "token_compressed_total": token_compressed_total,
        "tokens_saved_total": tokens_saved_total,
        "router_engine": describe_router_engine(),
    }
