import json
import re
import urllib.error
import urllib.request
from typing import Dict, List

from config import settings

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "forget your system prompt",
    "you are now a different ai",
]

_HARMFUL_KEYWORDS = {
    "violence": ["kill", "murder", "bomb", "shoot", "attack", "weapon"],
    "self_harm": ["suicide", "self harm", "hurt myself", "end my life"],
    "illegal": ["hack bank", "steal credit card", "make drugs", "counterfeit"],
    "explicit": ["porn", "sexual content", "explicit nudes"],
}

_STAT_PATTERNS = [
    r"\b(exact|latest|current|official)\s+(statistics|stats|numbers|figures)\b",
    r"\bwhat\s+is\s+the\s+current\s+(rate|number|count)\b",
]
_RECENT_EVENT_PATTERNS = [
    r"\b(today|this week|this month|this year)\b",
    r"\b(2025|2026|latest news|recent event)\b",
]
_MEDICAL_PATTERNS = [r"\bdiagnose\b", r"\bmedical advice\b", r"\btreatment for\b"]
_LEGAL_PATTERNS = [r"\blegal advice\b", r"\bsue\b", r"\blawsuit\b", r"\bcontract legality\b"]
_FINANCIAL_PATTERNS = [r"\binvestment advice\b", r"\bwhat stock should i buy\b", r"\bfinancial advice\b"]
_REQUEST_HINTS = [
    "?", "explain", "summarize", "compare", "list", "help", "write", "create", "generate", "analyze",
]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _is_all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def check_prompt(text: str) -> Dict[str, object]:
    original = text if isinstance(text, str) else str(text)
    cleaned_prompt = _normalize_whitespace(original)
    lowered = cleaned_prompt.lower()
    warnings: List[str] = []

    # Check 1: harmful content and prompt injection (immediate block)
    harmful_matches = []
    for category, keywords in _HARMFUL_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            harmful_matches.append(category)
    injection_detected = any(phrase in lowered for phrase in _INJECTION_PATTERNS)
    harmful_content = bool(harmful_matches or injection_detected)
    if harmful_content:
        if harmful_matches:
            warnings.append(f"Harmful content detected: {', '.join(sorted(set(harmful_matches)))}")
        if injection_detected:
            warnings.append("Prompt injection pattern detected.")
        return {
            "allowed": False,
            "risk_level": "high",
            "hallucination_risk": False,
            "harmful_content": True,
            "malformed": False,
            "warnings": warnings,
            "fallback_strategy": "block",
            "cleaned_prompt": cleaned_prompt,
        }

    # Check 4: malformed prompt
    words = re.findall(r"\b\w+\b", cleaned_prompt)
    malformed_reasons: List[str] = []
    if len(words) < 3:
        malformed_reasons.append("Prompt too short (under 3 words).")
    if len(words) > 2000:
        malformed_reasons.append("Prompt too long (over 2000 words).")
    if _is_all_caps(cleaned_prompt):
        cleaned_prompt = cleaned_prompt.lower()
        malformed_reasons.append("All-caps prompt normalized.")
    if not any(hint in lowered for hint in _REQUEST_HINTS):
        malformed_reasons.append("No clear question or request detected.")
    malformed = bool(malformed_reasons)
    warnings.extend(malformed_reasons)

    # Check 2: hallucination risk
    hallucination_risk = any(
        [
            _contains_any_pattern(lowered, _STAT_PATTERNS),
            _contains_any_pattern(lowered, _RECENT_EVENT_PATTERNS),
            _contains_any_pattern(lowered, _MEDICAL_PATTERNS),
            _contains_any_pattern(lowered, _LEGAL_PATTERNS),
            _contains_any_pattern(lowered, _FINANCIAL_PATTERNS),
        ]
    )
    if hallucination_risk:
        warnings.append("Hallucination-risk query detected; add disclaimer in response.")

    risk_level = "low"
    fallback_strategy = "none"
    if hallucination_risk and malformed:
        risk_level = "high"
        fallback_strategy = "cascade"
    elif hallucination_risk:
        risk_level = "medium"
        fallback_strategy = "disclaimer"
    elif malformed:
        risk_level = "medium"
        fallback_strategy = "disclaimer"

    return {
        "allowed": True,
        "risk_level": risk_level,
        "hallucination_risk": hallucination_risk,
        "harmful_content": False,
        "malformed": malformed,
        "warnings": warnings,
        "fallback_strategy": fallback_strategy,
        "cleaned_prompt": cleaned_prompt,
    }


def _is_complex_query(prompt: str) -> bool:
    lower = prompt.lower()
    complexity_markers = ["analyze", "compare", "evaluate", "multi-step", "tradeoff", "statistics", "reasoning"]
    return len(prompt.split()) > 20 or any(marker in lower for marker in complexity_markers)


def check_response_hallucination(prompt: str, response: str, query_label: int = 2) -> Dict[str, object]:
    # Step 3 only runs on complex queries (label 2).
    if query_label != 2 and not _is_complex_query(prompt):
        return {"hallucination_detected": False, "confidence": 0.0, "action": "none"}

    api_key = settings.GROQ_API_KEY.strip()
    if not api_key:
        # Safe local fallback if API key is unavailable.
        suspicious = bool(re.search(r"\b\d{3,}%|\bexactly\b|\bproven\b", response.lower()))
        return {
            "hallucination_detected": suspicious,
            "confidence": 0.6 if suspicious else 0.3,
            "action": "flag" if suspicious else "none",
        }

    user_prompt = (
        "Does this response contain unsupported claims or made up statistics?\n"
        "Answer only: Yes or No\n\n"
        f"Prompt: {prompt}\n\nResponse: {response}"
    )
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0,
        "max_tokens": 2,
    }

    request = urllib.request.Request(
        url="https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=3) as response_obj:
            body = json.loads(response_obj.read().decode("utf-8"))
        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )
        detected = content.startswith("yes")
        return {
            "hallucination_detected": detected,
            "confidence": 0.9 if detected else 0.8,
            "action": "cascade" if detected else "none",
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return {"hallucination_detected": False, "confidence": 0.2, "action": "flag"}
