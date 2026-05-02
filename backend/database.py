import uuid
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import settings


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_original: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_compressed: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_length: Mapped[int] = mapped_column(Integer, nullable=False)
    token_original: Mapped[int] = mapped_column(Integer, nullable=False)
    token_compressed: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_saved: Mapped[int] = mapped_column(Integer, nullable=False)
    has_code: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_math: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_reasoning: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sentence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_word_length: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    has_creative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    intent: Mapped[str] = mapped_column(String, nullable=False)
    routed_to: Mapped[str] = mapped_column(String, nullable=False)
    router_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    routing_reason: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    llm_judge_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bleu_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    semantic_similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hallucination_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    overall_quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    feedback_signal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ab_test_group: Mapped[str] = mapped_column(String, nullable=False, default="A")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class RouterTrainingLog(Base):
    __tablename__ = "router_training_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy_before: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy_after: Mapped[float] = mapped_column(Float, nullable=False)
    trigger_reason: Mapped[str] = mapped_column(String, nullable=False)
    improved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    triggered_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class CacheEntry(Base):
    __tablename__ = "cache_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_embedding_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    intent: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class DailyCostLog(Base):
    __tablename__ = "daily_cost_log"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_queries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    groq_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    gemini_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    budget_limit: Mapped[float] = mapped_column(Float, nullable=False, default=settings.DAILY_BUDGET_USD)
    budget_exceeded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


engine = create_async_engine(
    settings.POSTGRES_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def test_db_connection() -> bool:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True
