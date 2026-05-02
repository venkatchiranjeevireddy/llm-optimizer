import json
import random
import re
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

from config import settings

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:  # pragma: no cover
    sentence_bleu = None
    SmoothingFunction = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


MODEL_ORDER = ["groq_simple", "groq_mixtral", "gemini_pro"]
_EMBED_MODEL = None
_AB_TEST_RESULTS = defaultdict(list)
_AB_TEST_WINNER = {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_embedder():
    global _EMBED_MODEL
    if _EMBED_MODEL is None and SentenceTransformer is not None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "your", "have", "what", "when", "where", "which",
        "into", "about", "would", "could", "please", "should", "there", "their",
    }
    return [w for w in words if w not in stop]


def _judge_with_groq(prompt: str, response: str) -> float:
    api_key = settings.GROQ_API_KEY.strip()
    if not api_key:
        return 0.5
    judge_prompt = (
        "Rate this response quality 0 to 10.\n"
        "Return ONLY a number, nothing else.\n"
        f"Question: {prompt}\n"
        f"Response: {response}"
    )
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0,
        "max_tokens": 5,
    }
    req = urllib.request.Request(
        url="https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        match = re.search(r"(\d+(\.\d+)?)", content)
        if not match:
            return 0.5
        score_10 = float(match.group(1))
        return _clamp01(score_10 / 10.0)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError):
        return 0.5


def _bleu_keyword_overlap(prompt: str, response: str) -> float:
    """BLEU-style relevance without NLTK (avoids Py3.12 + nltk 3.8.1 Fraction bug)."""
    keywords = _extract_keywords(prompt)
    if not keywords:
        return 0.0
    candidate = _extract_keywords(response)
    if not candidate:
        return 0.0
    overlap = len(set(candidate) & set(keywords))
    return _clamp01(overlap / max(1, len(set(keywords))))


def _bleu_against_prompt_keywords(prompt: str, response: str) -> float:
    keywords = _extract_keywords(prompt)
    if not keywords:
        return 0.0
    candidate = _extract_keywords(response)
    if not candidate:
        return 0.0
    if sentence_bleu is None or SmoothingFunction is None:
        return _bleu_keyword_overlap(prompt, response)
    # NLTK 3.8.x calls Fraction(..., _normalize=False) which breaks on Python 3.12+.
    try:
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([keywords], candidate, smoothing_function=smoothing)
        return _clamp01(float(score))
    except (TypeError, ValueError, ZeroDivisionError):
        return _bleu_keyword_overlap(prompt, response)


def _semantic_similarity(prompt: str, response: str) -> float:
    model = _load_embedder()
    if model is None or np is None:
        p_set = set(_extract_keywords(prompt))
        r_set = set(_extract_keywords(response))
        if not p_set and not r_set:
            return 1.0
        if not p_set or not r_set:
            return 0.0
        return _clamp01(len(p_set & r_set) / len(p_set | r_set))

    p_emb = model.encode([prompt], normalize_embeddings=True)[0]
    r_emb = model.encode([response], normalize_embeddings=True)[0]
    sim = float(np.dot(p_emb, r_emb))
    return _clamp01(sim)


def _is_complex_query(prompt: str, intent: str) -> bool:
    """Heuristic for 'needs depth' — avoid treating every 'explain …' as complex (false hallucination flags)."""
    if intent in {"math", "comparison", "code"}:
        return True
    words = prompt.split()
    if len(words) >= 22:
        return True
    lowered = prompt.lower()
    strong_markers = [
        "analyze",
        "compare",
        "evaluate",
        "tradeoff",
        "multi-step",
        "multistep",
        "statistics",
        "reasoning",
        "strategy",
        "step by step",
        "prove that",
        "derive",
        "formal proof",
    ]
    return any(marker in lowered for marker in strong_markers)


def _has_self_contradiction(response: str) -> bool:
    lowered = response.lower()
    contradiction_pairs = [
        ("always", "never"),
        ("is true", "is false"),
        ("increase", "decrease"),
    ]
    return any(a in lowered and b in lowered for a, b in contradiction_pairs)


def _numbers_look_fabricated(response: str) -> bool:
    numbers = re.findall(r"\b\d+(\.\d+)?%?\b", response)
    if len(numbers) >= 4 and "[source]" not in response.lower() and "according to" not in response.lower():
        return True
    if re.search(r"\b(exactly|guaranteed|proven)\b", response.lower()) and numbers:
        return True
    return False


def _hallucination_detected(prompt: str, response: str, intent: str) -> bool:
    if _is_complex_query(prompt, intent) and len(response.split()) < 10:
        return True
    if "[source]" in response.lower() or "[citation]" in response.lower():
        return True
    if _has_self_contradiction(response):
        return True
    if _numbers_look_fabricated(response):
        return True
    return False


def _quality_label(score: float) -> str:
    if score > 0.8:
        return "excellent"
    if score > 0.6:
        return "good"
    if score > 0.4:
        return "acceptable"
    return "poor"


def _feedback_signal(score: float) -> int:
    if score > 0.70:
        return 1
    if score < 0.40:
        return -1
    return 0


def _normalize_model(model) -> Tuple[str, float]:
    if isinstance(model, dict):
        name = str(model.get("name", "gemini_pro"))
        confidence = float(model.get("router_confidence", model.get("confidence", 1.0)))
        return (name if name in MODEL_ORDER else "gemini_pro"), confidence
    if isinstance(model, str):
        return (model if model in MODEL_ORDER else "gemini_pro"), 1.0
    return "gemini_pro", 1.0


def _next_model_up(model_name: str) -> str:
    idx = MODEL_ORDER.index(model_name) if model_name in MODEL_ORDER else len(MODEL_ORDER) - 1
    return MODEL_ORDER[min(idx + 1, len(MODEL_ORDER) - 1)]


def _apply_ab_test(model_name: str, intent: str, router_confidence: float, score: float) -> str:
    if 0.50 <= router_confidence <= 0.70:
        group = random.choice(["A", "B"])
        chosen = model_name if group == "A" else _next_model_up(model_name)
        _AB_TEST_RESULTS[(intent, group)].append({"model": chosen, "score": score})
        # "Log both results separately" represented by independent group logs.
        group_scores = _AB_TEST_RESULTS[(intent, group)]
        if len(group_scores) >= 100 and (intent, group) not in _AB_TEST_WINNER:
            by_model = defaultdict(list)
            for row in group_scores:
                by_model[row["model"]].append(row["score"])
            winner = max(by_model.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
            _AB_TEST_WINNER[(intent, group)] = winner
        return group
    return "none"


def evaluate_response(prompt: str, response: str, model, intent: str) -> Dict[str, object]:
    model_name, router_confidence = _normalize_model(model)
    llm_judge_score = _judge_with_groq(prompt, response)
    bleu_score = _bleu_against_prompt_keywords(prompt, response)
    semantic_similarity = _semantic_similarity(prompt, response)
    hallucination_detected = _hallucination_detected(prompt, response, intent)

    overall_score = (
        (llm_judge_score * 0.50)
        + (semantic_similarity * 0.30)
        + (bleu_score * 0.20)
    )
    overall_score = _clamp01(overall_score)

    ab_test_group = _apply_ab_test(model_name, intent, router_confidence, overall_score)

    return {
        "overall_score": round(overall_score, 6),
        "llm_judge_score": round(llm_judge_score, 6),
        "bleu_score": round(bleu_score, 6),
        "semantic_similarity": round(semantic_similarity, 6),
        "hallucination_detected": hallucination_detected,
        "quality_label": _quality_label(overall_score),
        "feedback_signal": _feedback_signal(overall_score),
        "ab_test_group": ab_test_group,
    }
