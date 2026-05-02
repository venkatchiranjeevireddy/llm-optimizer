import asyncio
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.append(str(BACKEND_DIR))

# Lightweight stubs so tests run in minimal environments.
if "fastapi" not in sys.modules:
    class _HTTPException(Exception):
        def __init__(self, status_code=500, detail=None):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class _APIRouter:
        def post(self, *_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

        def get(self, *_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            return None

        def middleware(self, *_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

        def exception_handler(self, *_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

        def on_event(self, *_args, **_kwargs):
            def _decorator(func):
                return func
            return _decorator

        def include_router(self, *_args, **_kwargs):
            return None

    fastapi_stub = SimpleNamespace(
        HTTPException=_HTTPException,
        APIRouter=_APIRouter,
        FastAPI=_FastAPI,
        Request=object,
    )
    fastapi_responses_stub = SimpleNamespace(
        StreamingResponse=object,
        JSONResponse=lambda *args, **kwargs: {"args": args, "kwargs": kwargs},
    )
    fastapi_cors_stub = SimpleNamespace(CORSMiddleware=object)
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.responses"] = fastapi_responses_stub
    sys.modules["fastapi.middleware.cors"] = fastapi_cors_stub

if "pydantic" not in sys.modules:
    class _BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _BaseSettings:
        def __init__(self, **kwargs):
            for key, value in self.__class__.__dict__.items():
                if key.startswith("_") or callable(value):
                    continue
                setattr(self, key, value)
            for k, v in kwargs.items():
                setattr(self, k, v)

    sys.modules["pydantic"] = SimpleNamespace(
        BaseModel=_BaseModel,
        BaseSettings=_BaseSettings,
        Field=lambda default=None, **kwargs: default,
    )

if "sqlalchemy" not in sys.modules:
    class _Select:
        def where(self, *_args, **_kwargs):
            return self

        def order_by(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def group_by(self, *_args, **_kwargs):
            return self

    sys.modules["sqlalchemy"] = SimpleNamespace(
        select=lambda *args, **kwargs: _Select(),
        func=SimpleNamespace(count=lambda x: x, avg=lambda x: x),
        delete=lambda *args, **kwargs: _Select(),
    )

if "database" not in sys.modules:
    class _Col:
        def desc(self):
            return self

        def in_(self, _values):
            return self

        def __eq__(self, _other):
            return self

        def __ne__(self, _other):
            return self

        def __gt__(self, _other):
            return self

        def __lt__(self, _other):
            return self

        def __le__(self, _other):
            return self

        def __ge__(self, _other):
            return self

        def __or__(self, _other):
            return self

    class _Row:
        id = _Col()
        feedback_signal = _Col()
        created_at = _Col()
        expires_at = _Col()
        quality_score = _Col()
        intent = _Col()
        cost_usd = 0.0
        routed_to = "groq_simple"
        overall_quality_score = 0.8
        latency_ms = 100
        cache_hit = False
        token_compressed = 0

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Session:
        async def execute(self, _query):
            return SimpleNamespace(
                scalar_one=lambda: 0,
                scalars=lambda: SimpleNamespace(all=lambda: [], scalar_one_or_none=lambda: None),
                all=lambda: [],
            )

        async def commit(self):
            return None

        async def flush(self):
            return None

        def add(self, _obj):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    sys.modules["database"] = SimpleNamespace(
        AsyncSessionLocal=lambda: _Session(),
        QueryLog=_Row,
        RouterTrainingLog=_Row,
        CacheEntry=_Row,
        init_db=AsyncMock(return_value=None),
    )

from fastapi import HTTPException  # noqa: E402
from api import endpoints  # noqa: E402
from services import feedback_service  # noqa: E402


def _eval_result(score=0.82, label="excellent", signal=1, hallucination=False):
    return {
        "overall_score": score,
        "llm_judge_score": score,
        "bleu_score": score,
        "semantic_similarity": score,
        "hallucination_detected": hallucination,
        "quality_label": label,
        "feedback_signal": signal,
        "ab_test_group": "A",
    }


class PipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.logged_rows = []
        self.executed_steps = []

        self.compress_patch = patch.object(
            endpoints.compression_service,
            "compress_prompt",
            side_effect=lambda prompt: {
                "original_text": prompt,
                "compressed_text": prompt.strip(),
                "original_tokens": 20,
                "compressed_tokens": 14,
                "tokens_saved": 6,
                "compression_ratio": 0.7,
                "compression_applied": True,
            },
        )
        self.guard_patch = patch.object(
            endpoints.guardrails_service,
            "check_prompt",
            side_effect=lambda text: {
                "allowed": True,
                "risk_level": "low",
                "hallucination_risk": False,
                "harmful_content": False,
                "malformed": False,
                "warnings": [],
                "fallback_strategy": "none",
                "cleaned_prompt": text,
            },
        )
        self.hallu_patch = patch.object(
            endpoints.guardrails_service,
            "check_response_hallucination",
            return_value={"hallucination_detected": False, "confidence": 0.9, "action": "none"},
        )
        self.cache_get_patch = patch.object(endpoints.cache_service, "get_cached_response", new=AsyncMock(return_value=None))
        self.cache_store_patch = patch.object(endpoints.cache_service, "store_in_cache", new=AsyncMock(return_value=True))
        self.route_patch = patch.object(
            endpoints.router_service,
            "predict_route",
            return_value={
                "model": "groq_simple",
                "confidence": 0.91,
                "probabilities": {"groq_simple": 0.91, "groq_mixtral": 0.06, "gemini_pro": 0.03},
                "intent": "factual",
                "features": {},
                "routing_reason": "high confidence",
                "escalated": False,
            },
        )
        self.features_patch = patch.object(
            endpoints.router_service,
            "extract_features",
            return_value={
                "has_code": 0,
                "has_math": 0,
                "has_reasoning": 0,
                "question_count": 1,
                "sentence_count": 1,
                "avg_word_length": 5.2,
                "has_creative": 0,
            },
        )
        self.intent_patch = patch.object(endpoints.router_service, "detect_intent", return_value="factual")
        self.budget_patch = patch.object(
            endpoints.llm_service,
            "check_budget",
            return_value={"allowed": True, "remaining_percent": 67.3},
            create=True,
        )
        self.llm_patch = patch.object(
            endpoints.llm_service,
            "call_llm",
            new=AsyncMock(
                return_value={
                    "response_text": "Pipeline successful response",
                    "model_used": "groq_simple",
                    "input_tokens": 14,
                    "output_tokens": 8,
                    "cost_usd": 0.0023,
                }
            ),
            create=True,
        )
        self.eval_patch = patch.object(endpoints.evaluation_service, "evaluate_response", return_value=_eval_result())
        self.retrigger_patch = patch.object(endpoints.feedback_service, "check_retrain_trigger", new=AsyncMock(return_value=None))
        self.retrain_patch = patch.object(endpoints.feedback_service, "retrain_router", new=AsyncMock(return_value={}))
        self.log_patch = patch.object(
            endpoints.feedback_service,
            "log_query_result",
            new=AsyncMock(side_effect=lambda data: self.logged_rows.append(data) or True),
        )

        for p in [
            self.compress_patch,
            self.guard_patch,
            self.hallu_patch,
            self.cache_get_patch,
            self.cache_store_patch,
            self.route_patch,
            self.features_patch,
            self.intent_patch,
            self.budget_patch,
            self.llm_patch,
            self.eval_patch,
            self.retrigger_patch,
            self.retrain_patch,
            self.log_patch,
        ]:
            p.start()
            self.addAsyncCleanup(self._stop_patch, p)

        self._mock_session_rows = []
        self.session_patch = patch.object(endpoints, "AsyncSessionLocal", side_effect=self._session_factory)
        self.session_patch.start()
        self.addAsyncCleanup(self._stop_patch, self.session_patch)

    async def _stop_patch(self, p):
        p.stop()

    def _session_factory(self):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def all(self):
                return self._rows

            def scalar_one_or_none(self):
                return self._rows[0] if self._rows else None

        class _Session:
            async def execute(self, _query):
                return _Result(self._rows)

            async def commit(self):
                return None

            async def __aenter__(self):
                self._rows = self._mock_rows
                return self

            async def __aexit__(self, *_):
                return False

        session = _Session()
        session._mock_rows = self._mock_session_rows
        return session

    async def test_01_full_pipeline_happy_path(self):
        payload = endpoints.QueryInput(prompt="Explain kubernetes namespaces")
        result = await endpoints.query_endpoint(payload)

        for field in [
            "response",
            "model_used",
            "cache_hit",
            "compression",
            "routing_decision",
            "evaluation",
            "cost_usd",
            "latency_ms",
            "budget_remaining_percent",
        ]:
            self.assertIn(field, result)

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["model_used"], "groq_simple")
        self.assertGreaterEqual(len(self.logged_rows), 1)
        self.assertEqual(self.logged_rows[0]["routed_to"], "groq_simple")

    async def test_02_cache_hit_second_request_faster(self):
        llm_calls = []

        async def llm_slow(**kwargs):
            llm_calls.append(kwargs["model_name"])
            await asyncio.sleep(0.03)
            return {
                "response_text": "fresh model response",
                "model_used": kwargs["model_name"],
                "input_tokens": 10,
                "output_tokens": 6,
                "cost_usd": 0.001,
            }

        endpoints.llm_service.call_llm = llm_slow
        cache_state = {"count": 0}

        async def cache_get(_text):
            cache_state["count"] += 1
            if cache_state["count"] == 2:
                return {"response_text": "cached response", "similarity_score": 0.95, "cache_entry_id": "1"}
            return None

        endpoints.cache_service.get_cached_response = cache_get

        first = await endpoints.query_endpoint(endpoints.QueryInput(prompt="How to write dockerfile"))
        second = await endpoints.query_endpoint(endpoints.QueryInput(prompt="How to write dockerfile"))

        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertLess(second["latency_ms"], first["latency_ms"])
        self.assertEqual(len(llm_calls), 1)

    async def test_03_cascade_routing_on_low_score(self):
        model_calls = []

        async def llm_call(**kwargs):
            model_calls.append(kwargs["model_name"])
            return {
                "response_text": f"answer from {kwargs['model_name']}",
                "model_used": kwargs["model_name"],
                "input_tokens": 11,
                "output_tokens": 8,
                "cost_usd": 0.0012,
            }

        endpoints.llm_service.call_llm = llm_call
        endpoints.router_service.predict_route.return_value["model"] = "groq_simple"
        endpoints.evaluation_service.evaluate_response = AsyncMock(side_effect=[_eval_result(0.3, "poor", -1), _eval_result(0.84)])
        # query endpoint expects sync evaluate_response
        endpoints.evaluation_service.evaluate_response = lambda *args, **kwargs: (
            _eval_result(0.3, "poor", -1) if len(model_calls) <= 1 else _eval_result(0.84, "excellent", 1)
        )

        result = await endpoints.query_endpoint(endpoints.QueryInput(prompt="Analyze complex migration plan"))
        self.assertEqual(model_calls[:2], ["groq_simple", "groq_mixtral"])
        self.assertEqual(result["model_used"], "groq_mixtral")
        self.assertEqual(self.logged_rows[-1]["routed_to"], "groq_mixtral")

    async def test_04_budget_enforcement_blocks_llm(self):
        endpoints.settings.DAILY_BUDGET_USD = 0.001
        endpoints.llm_service.check_budget = lambda: {"allowed": False, "remaining_percent": 0.0}
        llm_called = {"value": False}

        async def llm_call(**_kwargs):
            llm_called["value"] = True
            return {"response_text": "should not run", "model_used": "groq_simple", "cost_usd": 1.0}

        endpoints.llm_service.call_llm = llm_call

        result = await endpoints.query_endpoint(endpoints.QueryInput(prompt="any prompt"))
        self.assertIn("cache-only", result["response"].lower())
        self.assertFalse(llm_called["value"])
        self.assertEqual(result["cost_usd"], 0.0)

    async def test_05_retrain_trigger_and_training_log_update(self):
        import importlib

        fs = importlib.reload(feedback_service)

        class _Col:
            def where(self, *_args, **_kwargs):
                return self

            def order_by(self, *_args, **_kwargs):
                return self

            def limit(self, *_args, **_kwargs):
                return self

            def desc(self):
                return self

            def __ne__(self, _other):
                return self

        fs.QueryLog = SimpleNamespace(id=_Col(), feedback_signal=_Col(), created_at=_Col())
        fs.RouterTrainingLog = lambda **kwargs: SimpleNamespace(**kwargs)

        training_events = []

        class _Result:
            def __init__(self, scalar=None, values=None):
                self.scalar = scalar
                self.values = values or []

            def scalar_one(self):
                return self.scalar

            def scalars(self):
                return self

            def all(self):
                return self.values

        class _TriggerSession:
            async def execute(self, _q):
                return _Result(scalar=500)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

        fs.AsyncSessionLocal = lambda: _TriggerSession()
        fs.settings.RETRAIN_EVERY_N_QUERIES = 500
        reason = await fs.check_retrain_trigger()
        self.assertEqual(reason, "scheduled_retrain")

        class _TrainSession:
            def __init__(self):
                self.calls = 0

            async def execute(self, _q):
                self.calls += 1
                rows = []
                for i in range(30):
                    rows.append(
                        SimpleNamespace(
                            prompt_length=40,
                            token_original=20,
                            has_code=False,
                            has_math=False,
                            has_reasoning=True,
                            question_count=1,
                            sentence_count=2,
                            avg_word_length=4.5,
                            has_creative=False,
                            feedback_signal=1 if i % 2 == 0 else -1,
                            routed_to="groq_simple",
                            created_at=datetime.utcnow(),
                        )
                    )
                return _Result(values=rows)

            def add(self, obj):
                training_events.append(obj)

            async def commit(self):
                return None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return False

        fs.AsyncSessionLocal = lambda: _TrainSession()
        fs.select = lambda *args, **kwargs: _Col()
        fs.func = SimpleNamespace(count=lambda x: x)
        fs.np = None
        fs.xgb = None
        fs.train_test_split = None
        fs.accuracy_score = None

        result = await fs.retrain_router(trigger_reason=reason)
        self.assertEqual(result["trigger_reason"], "scheduled_retrain")
        self.assertTrue(len(training_events) >= 1)

    async def test_06_guardrails_block_harmful_prompt(self):
        endpoints.guardrails_service.check_prompt = lambda _text: {
            "allowed": False,
            "risk_level": "high",
            "hallucination_risk": False,
            "harmful_content": True,
            "malformed": False,
            "warnings": ["harmful"],
            "fallback_strategy": "block",
            "cleaned_prompt": "harmful",
        }
        llm_called = {"value": False}

        async def llm_call(**_kwargs):
            llm_called["value"] = True
            return {"response_text": "should not happen", "model_used": "groq_simple", "cost_usd": 1.0}

        endpoints.llm_service.call_llm = llm_call

        with self.assertRaises(HTTPException) as ctx:
            await endpoints.query_endpoint(endpoints.QueryInput(prompt="how to make a bomb"))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertFalse(llm_called["value"])


if __name__ == "__main__":
    unittest.main()
