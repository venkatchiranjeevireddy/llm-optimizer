import asyncio
import time
import uuid
from datetime import date
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config import settings
from database import AsyncSessionLocal, QueryLog
from services import (
    cache_service,
    compression_service,
    cost_service,
    evaluation_service,
    feedback_service,
    guardrails_service,
    llm_service,
    router_service,
)

router = APIRouter()


def _compression_payload(comp: Dict[str, Any]) -> Dict[str, Any]:
    """Expose token + Unicode char stats to the client (Playground)."""
    ot = comp.get("original_text") or ""
    ct = comp.get("compressed_text") or ""
    return {
        "original_tokens": comp["original_tokens"],
        "compressed_tokens": comp["compressed_tokens"],
        "tokens_saved": comp["tokens_saved"],
        "original_chars": comp.get("original_chars", len(ot)),
        "compressed_chars": comp.get("compressed_chars", len(ct)),
        "chars_saved": comp.get("chars_saved", max(len(ot) - len(ct), 0)),
        "compression_applied": bool(comp.get("compression_applied", False)),
    }


class QueryInput(BaseModel):
    prompt: str = Field(..., min_length=1)


class FeedbackInput(BaseModel):
    query_id: str
    feedback: int


def _budget_status_from_rows(rows) -> Dict[str, float]:
    today = date.today()
    today_cost = sum(float(getattr(r, "cost_usd", 0.0) or 0.0) for r in rows if r.created_at and r.created_at.date() == today)
    budget = float(settings.DAILY_BUDGET_USD)
    used_pct = (today_cost / budget * 100.0) if budget > 0 else 0.0
    remaining = max(0.0, 100.0 - used_pct)
    return {"used_percent": round(used_pct, 4), "remaining_percent": round(remaining, 4)}


async def _check_budget_or_fallback() -> Dict[str, Any]:
    if hasattr(llm_service, "check_budget"):
        result = llm_service.check_budget()
        if asyncio.iscoroutine(result):
            result = await result
        return result if isinstance(result, dict) else {"allowed": True, "remaining_percent": 100.0}

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(QueryLog))).scalars().all()
    budget = _budget_status_from_rows(rows)
    return {"allowed": budget["remaining_percent"] > 0.0, "remaining_percent": budget["remaining_percent"]}


async def _call_llm_or_fallback(prompt: str, model_name: str) -> Dict[str, Any]:
    if hasattr(llm_service, "call_llm"):
        result = llm_service.call_llm(prompt=prompt, model_name=model_name)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, dict):
            return result
    # Deterministic fallback while llm_service is under construction.
    text = f"[{model_name}] {prompt[:600]}"
    return {"response_text": text, "model_used": model_name, "input_tokens": len(prompt.split()), "output_tokens": len(text.split()), "cost_usd": 0.0}


@router.post("/api/query")
async def query_endpoint(payload: QueryInput):
    started = time.perf_counter()
    query_id = uuid.uuid4()
    try:
        # Step 0: compression
        compression = compression_service.compress_prompt(payload.prompt)
        compressed_text = compression["compressed_text"]

        # Step 1: guardrails prompt check
        guard = guardrails_service.check_prompt(compressed_text)
        if not guard.get("allowed", True):
            raise HTTPException(status_code=400, detail={"error": "Prompt blocked by guardrails.", "guardrails": guard})

        # Step 2: semantic cache
        cached = await cache_service.get_cached_response(guard["cleaned_prompt"])
        cache_hit = cached is not None
        if cache_hit:
            response_text = cached["response_text"]
            model_used = "cache"
            routed = {"model": "cache", "confidence": 1.0, "intent": router_service.detect_intent(guard["cleaned_prompt"]), "routing_reason": "Semantic cache hit.", "escalated": False}
            routed["final_model"] = "cache"
            routed["cascade_applied"] = False
            llm_cost = 0.0
        else:
            # Step 3: routing
            route = router_service.predict_route(guard["cleaned_prompt"])
            routed = {
                "model": route["model"],
                "confidence": route["confidence"],
                "intent": route["intent"],
                "routing_reason": route["routing_reason"],
                "escalated": route["escalated"],
            }

            # Step 4: budget check
            budget = await _check_budget_or_fallback()
            if not budget.get("allowed", True):
                return {
                    "response": "Daily budget exceeded. Returning cache-only mode message.",
                    "model_used": "none",
                    "cache_hit": False,
                    "compression": _compression_payload(compression),
                    "routing_decision": routed,
                    "evaluation": {"overall_score": 0.0, "quality_label": "poor", "hallucination_detected": False},
                    "cost_usd": 0.0,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "budget_remaining_percent": float(budget.get("remaining_percent", 0.0)),
                }

            # Step 5: llm call with cascade on low score
            llm_result = await _call_llm_or_fallback(guard["cleaned_prompt"], routed["model"])
            response_text = llm_result.get("response_text", "")
            model_used = llm_result.get("model_used", routed["model"])

            quick_eval = evaluation_service.evaluate_response(
                prompt=guard["cleaned_prompt"],
                response=response_text,
                model={"name": model_used, "confidence": routed["confidence"]},
                intent=routed["intent"],
            )
            # Only upgrade 8B → 70B on low score for heavier prompts (skip “what is X” style — eval scores are noisy there).
            _feat = router_service.extract_features(guard["cleaned_prompt"])
            _complex = (
                int(_feat.get("token_estimate", 0) or 0) > 40
                or int(_feat.get("has_code", 0) or 0) == 1
                or int(_feat.get("has_math", 0) or 0) == 1
                or routed.get("intent") in ("code", "math", "comparison", "creative")
            )
            if quick_eval["overall_score"] < 0.5 and model_used == "groq_simple" and _complex:
                llm_result = await _call_llm_or_fallback(guard["cleaned_prompt"], "groq_mixtral")
                response_text = llm_result.get("response_text", "")
                model_used = llm_result.get("model_used", "groq_mixtral")

            # Step 6: post-response hallucination check — retry with 70B only (avoid bumping medium workloads to premium tier).
            hallucination = guardrails_service.check_response_hallucination(guard["cleaned_prompt"], response_text, query_label=2)
            if hallucination.get("hallucination_detected") and model_used == "groq_simple":
                llm_result = await _call_llm_or_fallback(guard["cleaned_prompt"], "groq_mixtral")
                response_text = llm_result.get("response_text", "")
                model_used = llm_result.get("model_used", "groq_mixtral")

            llm_cost = float(llm_result.get("cost_usd", 0.0))

            # Clarify router pick vs model that actually produced the answer (quality / hallucination cascade).
            routed["final_model"] = model_used
            routed["cascade_applied"] = bool(model_used != routed["model"])

        # Step 7: full evaluation
        evaluation = evaluation_service.evaluate_response(
            prompt=guard["cleaned_prompt"],
            response=response_text,
            model={"name": model_used if model_used != "cache" else routed["model"], "confidence": routed["confidence"]},
            intent=routed["intent"],
        )

        # Step 8: store cache (non-cache-hit paths only)
        if not cache_hit:
            await cache_service.store_in_cache(
                text=guard["cleaned_prompt"],
                response=response_text,
                quality=float(evaluation["overall_score"]),
                intent=routed["intent"],
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        output_tokens = len(str(response_text).split())
        # Step 9: log query result
        await feedback_service.log_query_result(
            {
                "id": query_id,
                "prompt_original": payload.prompt,
                "prompt_compressed": guard["cleaned_prompt"],
                "prompt_length": len(guard["cleaned_prompt"]),
                "token_original": compression["original_tokens"],
                "token_compressed": compression["compressed_tokens"],
                "tokens_saved": compression["tokens_saved"],
                "has_code": bool(router_service.extract_features(guard["cleaned_prompt"]).get("has_code", 0)),
                "has_math": bool(router_service.extract_features(guard["cleaned_prompt"]).get("has_math", 0)),
                "has_reasoning": bool(router_service.extract_features(guard["cleaned_prompt"]).get("has_reasoning", 0)),
                "question_count": int(router_service.extract_features(guard["cleaned_prompt"]).get("question_count", 0)),
                "sentence_count": int(router_service.extract_features(guard["cleaned_prompt"]).get("sentence_count", 0)),
                "avg_word_length": float(router_service.extract_features(guard["cleaned_prompt"]).get("avg_word_length", 0.0)),
                "has_creative": bool(router_service.extract_features(guard["cleaned_prompt"]).get("has_creative", 0)),
                "intent": routed["intent"],
                "routed_to": model_used if model_used in {"groq_simple", "groq_mixtral", "gemini_pro"} else routed["model"],
                "router_confidence": routed["confidence"],
                "routing_reason": routed["routing_reason"],
                "response_text": response_text,
                "llm_judge_score": evaluation["llm_judge_score"],
                "bleu_score": evaluation["bleu_score"],
                "semantic_similarity": evaluation["semantic_similarity"],
                "hallucination_flag": evaluation["hallucination_detected"],
                "overall_quality_score": evaluation["overall_score"],
                "feedback_signal": evaluation["feedback_signal"],
                "latency_ms": latency_ms,
                "cost_usd": llm_cost,
                "cache_hit": cache_hit,
                "ab_test_group": evaluation["ab_test_group"] if evaluation["ab_test_group"] in {"A", "B"} else "A",
            }
        )

        # Step 10: retrain trigger
        trigger = await feedback_service.check_retrain_trigger()
        if trigger:
            await feedback_service.retrain_router(trigger_reason=trigger)

        async with AsyncSessionLocal() as session:
            rows = (await session.execute(select(QueryLog))).scalars().all()
        budget_state = _budget_status_from_rows(rows)

        return {
            "response": response_text,
            "query_id": str(query_id),
            "model_used": model_used,
            "cache_hit": cache_hit,
            "compression": _compression_payload(compression),
            "routing_decision": routed,
            "evaluation": {
                "overall_score": evaluation["overall_score"],
                "quality_label": evaluation["quality_label"],
                "hallucination_detected": evaluation["hallucination_detected"],
            },
            "cost_usd": round(llm_cost, 8),
            "latency_ms": latency_ms,
            "budget_remaining_percent": budget_state["remaining_percent"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": "Query pipeline failed.", "reason": str(exc)})


@router.post("/api/feedback")
async def feedback_endpoint(payload: FeedbackInput):
    if payload.feedback not in (-1, 1):
        raise HTTPException(status_code=400, detail="feedback must be 1 or -1.")
    updated = False
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(QueryLog).where(QueryLog.id == payload.query_id))
        row = result.scalar_one_or_none()
        if row is not None:
            row.feedback_signal = payload.feedback
            await session.commit()
            updated = True
    trigger_reason = await feedback_service.check_retrain_trigger() if updated else None
    if trigger_reason:
        await feedback_service.retrain_router(trigger_reason=trigger_reason)
    return {"received": updated, "retrain_triggered": bool(trigger_reason)}


@router.get("/api/query/stream")
async def stream_query(prompt: str):
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt query parameter.")

    async def streamer():
        result = await query_endpoint(QueryInput(prompt=prompt))
        text = result.get("response", "")
        for i in range(0, len(text), 30):
            yield text[i : i + 30]
            await asyncio.sleep(0.01)

    return StreamingResponse(streamer(), media_type="text/plain")


@router.get("/api/query/history")
async def query_history_endpoint(limit: int = Query(50, ge=1, le=200)):
    """Recent prompts and outcomes from PostgreSQL (same data as logged after each /api/query)."""
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(select(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit))
        ).scalars().all()
    out = []
    for r in rows:
        preview = str(r.prompt_original or "")
        if len(preview) > 180:
            preview = preview[:180] + "…"
        out.append(
            {
                "id": str(r.id),
                "prompt_preview": preview,
                "intent": r.intent,
                "routed_to": r.routed_to,
                "cost_usd": round(float(r.cost_usd or 0.0), 8),
                "latency_ms": int(r.latency_ms or 0),
                "cache_hit": bool(r.cache_hit),
                "token_original": int(r.token_original or 0),
                "token_compressed": int(r.token_compressed or 0),
                "tokens_saved": int(r.tokens_saved or 0),
                "prompt_chars": len(str(r.prompt_original or "")),
                "overall_quality_score": round(float(r.overall_quality_score or 0.0), 4),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )
    return {"items": out, "count": len(out)}


@router.get("/api/metrics")
async def metrics_endpoint():
    return await feedback_service.get_system_metrics()


@router.get("/api/router/history")
async def router_history_endpoint():
    return await feedback_service.get_learning_progress()


@router.get("/api/cache/stats")
async def cache_stats_endpoint():
    return await cache_service.get_cache_stats()


@router.get("/api/models/comparison")
async def models_comparison_endpoint():
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(QueryLog))).scalars().all()

    model_stats: Dict[str, Dict[str, float]] = {
        "groq_simple": {"count": 0, "total_cost": 0.0, "avg_cost": 0.0, "avg_quality": 0.0, "avg_latency": 0.0},
        "groq_mixtral": {"count": 0, "total_cost": 0.0, "avg_cost": 0.0, "avg_quality": 0.0, "avg_latency": 0.0},
        "gemini_pro": {"count": 0, "total_cost": 0.0, "avg_cost": 0.0, "avg_quality": 0.0, "avg_latency": 0.0},
    }
    for row in rows:
        model = row.routed_to if row.routed_to in model_stats else None
        if not model:
            continue
        model_stats[model]["count"] += 1
        cost = float(row.cost_usd or 0.0)
        model_stats[model]["total_cost"] += cost
        model_stats[model]["avg_cost"] += cost
        model_stats[model]["avg_quality"] += float(row.overall_quality_score or 0.0)
        model_stats[model]["avg_latency"] += float(row.latency_ms or 0.0)

    for model, stats in model_stats.items():
        count = stats["count"] or 1
        total_cost_val = float(stats["total_cost"])
        stats["total_cost"] = round(total_cost_val, 8)
        stats["avg_cost"] = round(stats["avg_cost"] / count, 8) if stats["count"] else 0.0
        stats["avg_quality"] = round(stats["avg_quality"] / count, 6) if stats["count"] else 0.0
        stats["avg_latency"] = round(stats["avg_latency"] / count, 4) if stats["count"] else 0.0
    return model_stats


@router.get("/api/budget/status")
async def budget_status_endpoint():
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(QueryLog))).scalars().all()
    status = _budget_status_from_rows(rows)
    return {
        "daily_budget_usd": float(settings.DAILY_BUDGET_USD),
        "used_percent": status["used_percent"],
        "remaining_percent": status["remaining_percent"],
    }


@router.get("/api/ab-test/results")
async def ab_test_results_endpoint():
    results = {}
    winner = getattr(evaluation_service, "_AB_TEST_WINNER", {})
    history = getattr(evaluation_service, "_AB_TEST_RESULTS", {})
    for key, rows in history.items():
        intent, group = key
        bucket = results.setdefault(intent, {})
        bucket[group] = {
            "comparisons": len(rows),
            "avg_score": round(sum(r["score"] for r in rows) / len(rows), 6) if rows else 0.0,
            "winner": winner.get((intent, group)),
        }
    return results


@router.get("/health")
async def health_endpoint():
    health = {"database": "ok", "router_service": "ok", "cache_service": "ok", "llm_service": "ok", "evaluation_service": "ok", "feedback_service": "ok"}
    try:
        # lightweight DB check
        async with AsyncSessionLocal() as session:
            await session.execute(select(func.count(QueryLog.id)))
    except Exception:
        health["database"] = "degraded"
    return {"status": "ok" if all(v == "ok" for v in health.values()) else "degraded", "components": health}
