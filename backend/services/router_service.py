import json
import pickle
import re
from pathlib import Path
from typing import Dict, List

from config import settings


def _premium_tier_warranted(features: Dict[str, float], intent: str) -> bool:
    """gemini_pro tier: math, long dense prompts, deep code — not ordinary comparisons."""
    if intent == "math":
        return True
    if int(features.get("has_math", 0) or 0) == 1:
        return True
    if int(features.get("has_code", 0) or 0) == 1 and int(features.get("token_estimate", 0) or 0) > 55:
        return True
    te = int(features.get("token_estimate", 0) or 0)
    pl = int(features.get("prompt_length", 0) or 0)
    # Long / very dense only (raised bar so “medium” prose stays on 70B).
    if pl >= 1100 or te >= 190:
        return True
    if int(features.get("has_reasoning", 0) or 0) == 1 and pl >= 620:
        return True
    return False


def _simple_tier_fast_path(features: Dict[str, float], intent: str) -> bool:
    """Llama 8B: short factual / short CJK — not proofs or reasoning-heavy snippets."""
    te = int(features.get("token_estimate", 0) or 0)
    pl = int(features.get("prompt_length", 0) or 0)
    qc = int(features.get("question_count", 0) or 0)
    latin_short = te >= 1 and te <= 28 and pl <= 360 and qc <= 1
    cjk_short = te == 0 and pl <= 180 and qc <= 1
    return (
        features.get("has_code", 0) == 0
        and features.get("has_math", 0) == 0
        and features.get("has_creative", 0) == 0
        and int(features.get("has_reasoning", 0) or 0) == 0
        and intent == "factual"
        and (latin_short or cjk_short)
    )


def _medium_tier_policy(features: Dict[str, float], intent: str) -> bool:
    """Llama 70B: comparisons, summaries, creative, code-ish, or moderate factual — before XGB can pick premium."""
    if _premium_tier_warranted(features, intent):
        return False
    if intent in ("comparison", "summarization", "creative"):
        return True
    if intent == "code":
        return True
    te = int(features.get("token_estimate", 0) or 0)
    pl = int(features.get("prompt_length", 0) or 0)
    has_reasoning = int(features.get("has_reasoning", 0) or 0) == 1
    if intent == "factual" and has_reasoning and te <= 120 and pl <= 750:
        return True
    if intent == "factual" and 29 <= te <= 140 and pl <= 900:
        return True
    return False

DEFAULT_FEATURE_NAMES: List[str] = [
    "prompt_length",
    "token_estimate",
    "has_code",
    "has_math",
    "has_reasoning",
    "question_count",
    "sentence_count",
    "avg_word_length",
    "has_creative",
]

DEFAULT_LABEL_MAP = {"0": "groq_simple", "1": "groq_mixtral", "2": "gemini_pro"}
VALID_INTENTS = {"factual", "code", "creative", "summarization", "comparison", "math"}
TEMPLATES = {
    "factual": "Answer clearly and factually: {text}",
    "code": "Write clean production grade code for: {text}",
    "creative": "Write creatively and engagingly: {text}",
    "summarization": "Summarize this concisely: {text}",
    "comparison": "Compare these options objectively: {text}",
    "math": "Solve step by step showing all work: {text}",
}

_CODE_PATTERNS = [r"```", r"\b(def|class|function|import|return|SELECT|FROM|JOIN)\b", r"[{};]"]
_MATH_PATTERNS = [r"\b(integral|derivative|equation|solve|algebra|theorem|matrix)\b", r"[=+\-*/^]"]
# Strong triggers for “needs premium tier” — keep sparse; words like “explain” are too common on medium prompts.
_REASONING_WORDS = {"why", "because", "therefore", "prove", "rigorous", "contradiction", "lemma"}
_CREATIVE_WORDS = {"story", "poem", "creative", "imagine", "fiction", "narrative", "song"}


class _HeuristicRouterModel:
    classes_ = [0, 1, 2]
    feature_importances_ = [0.10, 0.12, 0.18, 0.16, 0.12, 0.08, 0.08, 0.06, 0.10]

    def predict_proba(self, rows: List[List[float]]) -> List[List[float]]:
        probabilities = []
        for row in rows:
            has_code, has_math, has_reasoning, has_creative = row[2], row[3], row[4], row[8]
            prompt_length, token_estimate = row[0], row[1]
            question_count = row[5]

            groq_simple = 0.52
            groq_mixtral = 0.33
            gemini_pro = 0.15

            if has_code:
                groq_mixtral += 0.50
                groq_simple -= 0.22
            if has_math:
                gemini_pro += 0.55
                groq_simple -= 0.28
            if has_creative:
                groq_mixtral += 0.22
            # Long / heavy prompts → premium; short prompts with a reasoning keyword → medium only.
            if has_reasoning and (token_estimate > 90 or prompt_length > 600):
                gemini_pro += 0.28
                groq_simple -= 0.12
            elif has_reasoning:
                groq_mixtral += 0.18
                groq_simple -= 0.08
            elif token_estimate > 140 or prompt_length > 900:
                gemini_pro += 0.22
                groq_simple -= 0.10
            if question_count > 3:
                gemini_pro += 0.08

            values = [max(groq_simple, 0.01), max(groq_mixtral, 0.01), max(gemini_pro, 0.01)]
            total = sum(values)
            probabilities.append([v / total for v in values])
        return probabilities


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_load_json(path: Path, default):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _safe_load_model(path: Path):
    try:
        with path.open("rb") as handle:
            model = pickle.load(handle)
        if hasattr(model, "predict_proba"):
            return model
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError, ModuleNotFoundError):
        pass
    return _HeuristicRouterModel()


_ROOT = _project_root()
_MODEL_PATH = _ROOT / "models" / "router_model.pkl"
_FEATURE_PATH = _ROOT / "models" / "feature_names.json"
_LABEL_PATH = _ROOT / "models" / "label_map.json"

ROUTER_MODEL = _safe_load_model(_MODEL_PATH)
FEATURE_NAMES = _safe_load_json(_FEATURE_PATH, DEFAULT_FEATURE_NAMES)
if not isinstance(FEATURE_NAMES, list) or len(FEATURE_NAMES) != 9:
    FEATURE_NAMES = DEFAULT_FEATURE_NAMES
LABEL_MAP = _safe_load_json(_LABEL_PATH, DEFAULT_LABEL_MAP)
if not isinstance(LABEL_MAP, dict):
    LABEL_MAP = DEFAULT_LABEL_MAP


def describe_router_engine() -> Dict[str, str]:
    """Which object runs predict_proba — used by dashboard (XGBoost pickle vs built-in heuristic)."""
    m = ROUTER_MODEL
    cls_name = m.__class__.__name__
    mod = getattr(m.__class__, "__module__", "") or ""
    if cls_name == "XGBClassifier":
        return {
            "kind": "xgboost",
            "title": "XGBoost classifier",
            "detail": "models/router_model.pkl — trained when retrain_router() improves accuracy (multi-class: cheap / medium / premium tier).",
        }
    if cls_name == "_HeuristicRouterModel":
        return {
            "kind": "heuristic",
            "title": "Heuristic rules (fallback)",
            "detail": "No usable pickle at load time — rule-based predict_proba only.",
        }
    return {"kind": "other", "title": cls_name, "detail": mod}


def extract_features(text: str) -> Dict[str, float]:
    source = text if isinstance(text, str) else str(text)
    words = source.split()
    word_lengths = [len(re.sub(r"[^A-Za-z0-9]", "", word)) for word in words if word]
    sentence_count = len([chunk for chunk in re.split(r"[.!?]+", source) if chunk.strip()])
    lowered = source.lower()

    features = {
        "prompt_length": len(source),
        "token_estimate": len(words),
        "has_code": int(any(re.search(p, source, re.IGNORECASE) for p in _CODE_PATTERNS)),
        "has_math": int(any(re.search(p, source, re.IGNORECASE) for p in _MATH_PATTERNS)),
        "has_reasoning": int(any(word in lowered for word in _REASONING_WORDS)),
        "question_count": source.count("?"),
        "sentence_count": sentence_count if sentence_count > 0 else (1 if source.strip() else 0),
        "avg_word_length": (sum(word_lengths) / len(word_lengths)) if word_lengths else 0.0,
        "has_creative": int(any(word in lowered for word in _CREATIVE_WORDS)),
    }
    return features


def detect_intent(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in ["compare", "vs", "versus", "difference", "pros and cons"]):
        return "comparison"
    if any(k in lowered for k in ["summarize", "summary", "tl;dr", "condense"]):
        return "summarization"
    if any(re.search(p, lowered, re.IGNORECASE) for p in _CODE_PATTERNS):
        return "code"
    if any(re.search(p, lowered, re.IGNORECASE) for p in _MATH_PATTERNS):
        return "math"
    if any(k in lowered for k in _CREATIVE_WORDS):
        return "creative"
    return "factual"


def wrap_with_template(text: str, intent: str) -> str:
    normalized_intent = intent if intent in VALID_INTENTS else "factual"
    template = TEMPLATES[normalized_intent]
    return template.format(text=text)


def _model_name_from_index(index: int) -> str:
    return LABEL_MAP.get(str(index), DEFAULT_LABEL_MAP.get(str(index), "groq_simple"))


def predict_route(text: str) -> Dict[str, object]:
    intent = detect_intent(text)
    features = extract_features(text)
    feature_vector = [features.get(name, 0.0) for name in FEATURE_NAMES]

    # --- Tier order: hard → simple → medium (policy), then XGBoost fallback ---

    # 1) Premium / hard (only when bar is met — not comparison-by-default)
    if _premium_tier_warranted(features, intent):
        return {
            "model": "gemini_pro",
            "confidence": 0.88,
            "probabilities": {"groq_simple": 0.05, "groq_mixtral": 0.12, "gemini_pro": 0.83},
            "intent": intent,
            "features": features,
            "routing_reason": "Tier policy: math / long-form / heavy code → premium tier.",
            "escalated": False,
        }

    # 2) Simple / cheap
    if _simple_tier_fast_path(features, intent):
        return {
            "model": "groq_simple",
            "confidence": 0.9,
            "probabilities": {"groq_simple": 0.9, "groq_mixtral": 0.07, "gemini_pro": 0.03},
            "intent": intent,
            "features": features,
            "routing_reason": "Tier policy: short factual → Llama 8B.",
            "escalated": False,
        }

    # 3) Medium (beats XGBoost labeling everything premium)
    if _medium_tier_policy(features, intent):
        return {
            "model": "groq_mixtral",
            "confidence": 0.86,
            "probabilities": {"groq_simple": 0.08, "groq_mixtral": 0.87, "gemini_pro": 0.05},
            "intent": intent,
            "features": features,
            "routing_reason": "Tier policy: comparison / summary / creative / moderate prose → Llama 70B (medium).",
            "escalated": False,
        }

    proba_raw = ROUTER_MODEL.predict_proba([feature_vector])[0]
    probabilities = {
        _model_name_from_index(i): float(round(max(p, 0.0), 4))
        for i, p in enumerate(proba_raw[:3])
    }
    # Ensure all expected model keys exist.
    for key in ("groq_simple", "groq_mixtral", "gemini_pro"):
        probabilities.setdefault(key, 0.0)

    selected_model = max(probabilities, key=probabilities.get)
    confidence = float(round(probabilities[selected_model], 4))
    escalated = False
    reason = f"Selected {selected_model} based on intent '{intent}' and extracted feature profile."

    # XGBoost/heuristic often labels comparison & medium prose as premium — cap unless truly hard.
    capped_from_premium = False
    if selected_model == "gemini_pro" and not _premium_tier_warranted(features, intent):
        selected_model = "groq_mixtral"
        confidence = float(round(probabilities.get("groq_mixtral", 0.0), 4))
        capped_from_premium = True
        reason = (
            f"{reason} Premium tier reserved for math / very long prompts — using Llama 70B (medium) instead."
        )

    thresh = float(settings.ROUTER_CONFIDENCE_THRESHOLD)
    if confidence < thresh and not capped_from_premium:
        escalated = True
        # Uncertain classifier → default to cheapest tier (8B). Skip if we just downgraded premium→medium (keep 70B).
        selected_model = "groq_simple"
        reason = f"{reason} Low confidence (<{thresh}) → default Llama 8B to optimize cost."

    return {
        "model": selected_model,
        "confidence": confidence,
        "probabilities": probabilities,
        "intent": intent,
        "features": features,
        "routing_reason": reason,
        "escalated": escalated,
    }


def get_feature_importance() -> Dict[str, float]:
    importances = getattr(ROUTER_MODEL, "feature_importances_", [0.0] * len(FEATURE_NAMES))
    if len(importances) < len(FEATURE_NAMES):
        importances = list(importances) + [0.0] * (len(FEATURE_NAMES) - len(importances))
    return {name: float(round(importances[i], 6)) for i, name in enumerate(FEATURE_NAMES)}
