from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Mapping


MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "groq_simple": {
        "model_id": "llama-3.1-8b-instant",
        "provider": "groq",
        "input_cost_per_1m": 0.05,
        "output_cost_per_1m": 0.08,
        "speed_tps": 560,
    },
    "groq_mixtral": {
        "model_id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "input_cost_per_1m": 0.59,
        "output_cost_per_1m": 0.79,
        "speed_tps": 280,
    },
    "gemini_pro": {
        "model_id": "gemini-1.5-flash",
        "provider": "gemini",
        "input_cost_per_1m": 0.075,
        "output_cost_per_1m": 0.30,
        "speed_tps": 150,
    },
}

_ROUND_8 = Decimal("0.00000001")
_ONE_MILLION = Decimal("1000000")


def _to_decimal(value) -> Decimal:
    return Decimal(str(value))


def _round8(value: Decimal) -> float:
    return float(value.quantize(_ROUND_8, rounding=ROUND_HALF_UP))


def _extract_value(log_obj, key: str, default=0):
    if isinstance(log_obj, Mapping):
        return log_obj.get(key, default)
    return getattr(log_obj, key, default)


def calculate_real_cost(model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, object]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_name: {model_name}")

    config = MODEL_CONFIGS[model_name]
    input_tok = int(input_tokens)
    output_tok = int(output_tokens)
    if input_tok < 0 or output_tok < 0:
        raise ValueError("Token counts must be non-negative.")

    input_rate = _to_decimal(config["input_cost_per_1m"])
    output_rate = _to_decimal(config["output_cost_per_1m"])

    input_cost = (_to_decimal(input_tok) / _ONE_MILLION) * input_rate
    output_cost = (_to_decimal(output_tok) / _ONE_MILLION) * output_rate
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "input_cost_usd": _round8(input_cost),
        "output_cost_usd": _round8(output_cost),
        "total_cost_usd": _round8(total_cost),
        "model_id": str(config["model_id"]),
    }


def calculate_baseline_cost(input_tokens: int, output_tokens: int) -> float:
    baseline = calculate_real_cost("gemini_pro", input_tokens, output_tokens)
    return _round8(_to_decimal(baseline["total_cost_usd"]))


def calculate_savings(actual_cost: float, baseline_cost: float) -> Dict[str, float]:
    actual = _to_decimal(actual_cost)
    baseline = _to_decimal(baseline_cost)
    saved = baseline - actual
    if baseline <= Decimal("0"):
        percentage = Decimal("0")
    else:
        percentage = (saved / baseline) * Decimal("100")
    return {
        "saved_usd": _round8(saved),
        "saved_percentage": _round8(percentage),
    }


def get_cost_summary(query_logs_list: List[object]) -> Dict[str, object]:
    total_actual = Decimal("0")
    total_baseline = Decimal("0")
    cost_by_model = {
        "groq_simple": Decimal("0"),
        "groq_mixtral": Decimal("0"),
        "gemini_pro": Decimal("0"),
        "cache_hits": Decimal("0"),
    }

    for log in query_logs_list:
        model_name = str(_extract_value(log, "routed_to", "gemini_pro"))
        if model_name not in MODEL_CONFIGS:
            model_name = "gemini_pro"

        cache_hit = bool(_extract_value(log, "cache_hit", False))
        input_tokens = int(_extract_value(log, "token_compressed", _extract_value(log, "input_tokens", 0)))
        output_tokens = int(_extract_value(log, "output_tokens", 0))
        actual_cost_raw = _extract_value(log, "cost_usd", None)

        if cache_hit:
            actual_cost = Decimal("0")
            model_bucket = "cache_hits"
        elif actual_cost_raw is not None:
            actual_cost = _to_decimal(actual_cost_raw)
            model_bucket = model_name
        else:
            actual_cost = _to_decimal(calculate_real_cost(model_name, input_tokens, output_tokens)["total_cost_usd"])
            model_bucket = model_name

        baseline_cost = _to_decimal(calculate_baseline_cost(input_tokens, output_tokens))

        total_actual += actual_cost
        total_baseline += baseline_cost
        cost_by_model[model_bucket] += actual_cost

    total_saved = total_baseline - total_actual
    total_saved_pct = Decimal("0")
    if total_baseline > Decimal("0"):
        total_saved_pct = (total_saved / total_baseline) * Decimal("100")

    query_count = len(query_logs_list)
    avg_cost = (total_actual / Decimal(str(query_count))) if query_count else Decimal("0")

    return {
        "total_actual_cost": _round8(total_actual),
        "total_baseline_cost": _round8(total_baseline),
        "total_saved_usd": _round8(total_saved),
        "total_saved_percentage": _round8(total_saved_pct),
        "cost_by_model": {
            "groq_simple": _round8(cost_by_model["groq_simple"]),
            "groq_mixtral": _round8(cost_by_model["groq_mixtral"]),
            "gemini_pro": _round8(cost_by_model["gemini_pro"]),
            "cache_hits": _round8(cost_by_model["cache_hits"]),
        },
        "avg_cost_per_query": _round8(avg_cost),
    }
