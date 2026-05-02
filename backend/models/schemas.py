from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


IntentType = Literal["factual", "code", "creative", "summarization", "comparison", "math"]
RouteType = Literal["groq_simple", "groq_mixtral", "gemini_pro"]
ABTestGroup = Literal["A", "B"]


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=1000, ge=1, le=8192)
    use_cache: bool = True


class RouterDecision(BaseModel):
    intent: IntentType
    routed_to: RouteType
    router_confidence: float = Field(..., ge=0.0, le=1.0)
    routing_reason: str = Field(..., min_length=1)


class EvaluationResult(BaseModel):
    llm_judge_score: float = Field(..., ge=0.0, le=1.0)
    bleu_score: float = Field(..., ge=0.0, le=1.0)
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    hallucination_flag: bool
    overall_quality_score: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    id: UUID
    response_text: str = Field(..., min_length=1)
    prompt_original: str = Field(..., min_length=1)
    prompt_compressed: str = Field(..., min_length=1)
    token_original: int = Field(..., ge=0)
    token_compressed: int = Field(..., ge=0)
    tokens_saved: int = Field(..., ge=0)
    cache_hit: bool
    latency_ms: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)
    ab_test_group: ABTestGroup
    router_decision: RouterDecision
    evaluation: EvaluationResult
    created_at: datetime


class FeedbackRequest(BaseModel):
    query_id: UUID
    feedback_signal: int = Field(..., ge=-1, le=1)
    comment: Optional[str] = None


class MetricsResponse(BaseModel):
    total_queries: int = Field(..., ge=0)
    total_cost_usd: float = Field(..., ge=0.0)
    average_latency_ms: float = Field(..., ge=0.0)
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0)
    average_tokens_saved: float = Field(..., ge=0.0)
    bad_response_rate: float = Field(..., ge=0.0, le=1.0)
