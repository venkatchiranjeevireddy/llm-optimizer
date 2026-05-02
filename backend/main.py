import logging
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.endpoints import router as api_router
from database import init_db
from services import cache_service, feedback_service, llm_service, router_service

LOGGER = logging.getLogger("llm_optimizer_api")
logging.basicConfig(level=logging.INFO)

RATE_LIMIT = 100
RATE_WINDOW_SECONDS = 60
_REQUEST_BUCKETS = defaultdict(deque)

app = FastAPI(title="LLM Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_and_logging(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _REQUEST_BUCKETS[client_ip]
    while bucket and now - bucket[0] > RATE_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded: 100 requests per minute.")
    bucket.append(now)

    start = time.perf_counter()
    LOGGER.info("request_started method=%s path=%s ip=%s", request.method, request.url.path, client_ip)
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    LOGGER.info(
        "request_completed method=%s path=%s status=%s latency_ms=%s ip=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        client_ip,
    )
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    LOGGER.exception("unhandled_error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.on_event("startup")
async def startup_event():
    # 1) database connection/table init
    await init_db()
    # 2) router service (loads model at import; touch feature importances to validate)
    router_service.get_feature_importance()
    # 3) cache service (imported instance is SemanticCacheService)
    await cache_service.initialize()
    # 4) llm service (placeholder-safe)
    if hasattr(llm_service, "initialize"):
        maybe = llm_service.initialize()
        if hasattr(maybe, "__await__"):
            await maybe
    # 5) evaluation service (loads lazily; no-op init)
    # 6) feedback service (DB-backed; no-op init)
    if hasattr(feedback_service, "get_learning_progress"):
        await feedback_service.get_learning_progress()


app.include_router(api_router)
