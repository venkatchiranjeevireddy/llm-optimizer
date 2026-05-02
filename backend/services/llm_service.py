"""
LLM integration: Groq only at inference time.

- groq_simple  -> Llama 3.1 8B Instant
- groq_mixtral -> Llama 3.3 70B Versatile (medium / strong tier)
- gemini_pro   -> Same Groq 70B call; response tagged gemini_pro for product/UI differentiation.
  (No Google Gemini API call — avoids extra key and billing; metrics baseline elsewhere may still
  reference Gemini pricing for savings math.)

API keys: GROQ_API_KEY required. GEMINI_API_KEY optional/unused for completions here.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

import httpx
from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal, QueryLog
from services.cost_service import MODEL_CONFIGS, calculate_real_cost

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


def _groq_openai_model_id(route_name: str) -> str:
    if route_name == "groq_simple":
        return str(MODEL_CONFIGS["groq_simple"]["model_id"])
    if route_name in ("groq_mixtral", "gemini_pro"):
        return str(MODEL_CONFIGS["groq_mixtral"]["model_id"])
    return str(MODEL_CONFIGS["groq_simple"]["model_id"])


async def check_budget() -> Dict[str, Any]:
    """Daily spend vs DAILY_BUDGET_USD."""
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(QueryLog))).scalars().all()
    today = date.today()
    today_cost = sum(
        float(getattr(r, "cost_usd", 0.0) or 0.0)
        for r in rows
        if r.created_at and r.created_at.date() == today
    )
    budget = float(settings.DAILY_BUDGET_USD)
    used_pct = (today_cost / budget * 100.0) if budget > 0 else 0.0
    remaining = max(0.0, 100.0 - used_pct)
    return {
        "allowed": remaining > 0.0,
        "remaining_percent": round(remaining, 4),
        "used_percent": round(used_pct, 4),
        # Legacy flag: gemini_pro now uses Groq 70B too, so no downgrade swap needed.
        "force_groq": False,
    }


async def _groq_complete(
    prompt: str,
    route_for_cost: str,
    *,
    model_used_label: str | None = None,
) -> Dict[str, Any]:
    """Groq chat completion. route_for_cost selects which MODEL_CONFIGS pricing to apply."""
    api_key = settings.GROQ_API_KEY.strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    model_id = _groq_openai_model_id(route_for_cost)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": int(settings.MAX_TOKENS),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(GROQ_CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "").strip()
    usage = data.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    if not text:
        text = "(empty response from Groq)"

    cost = calculate_real_cost(route_for_cost, input_tokens, output_tokens)["total_cost_usd"]
    label = model_used_label if model_used_label else route_for_cost
    return {
        "response_text": text,
        "model_used": label,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": float(cost),
        "provider_model_id": model_id,
    }


async def call_llm(*, prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Execute one completion for the routed tier.
    - groq_simple  -> Llama 3.1 8B Instant
    - groq_mixtral -> Llama 3.3 70B (medium)
    - gemini_pro   -> Llama 3.3 70B (same weights/cost as medium); model_used stays gemini_pro for UI/logs.
    """
    budget = await check_budget()
    if not budget.get("allowed", True):
        return {
            "response_text": "Daily budget exceeded. No LLM call executed.",
            "model_used": "none",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "budget_blocked": True,
        }

    if model_name == "groq_simple":
        return await _groq_complete(prompt, "groq_simple")

    if model_name == "groq_mixtral":
        return await _groq_complete(prompt, "groq_mixtral")

    if model_name == "gemini_pro":
        # Same inference as groq_mixtral; bill at Groq 70B rates; label tier as gemini_pro for dashboards.
        result = await _groq_complete(prompt, "groq_mixtral", model_used_label="gemini_pro")
        return result

    return await _groq_complete(prompt, "groq_mixtral")


def initialize():
    """Optional sync hook for startup; no-op (clients are per-request)."""
    return None
