# LLM Optimizer
Production-ready adaptive LLM routing system that reduces cost and latency while maintaining response quality.

## Architecture

```text
                          +-----------------------+
User Prompt  -----------> |  FastAPI /api/query   |
                          +-----------+-----------+
                                      |
                                      v
                       +----------------------------+
Step 0                 | compression_service        |
                       | remove filler + shorten    |
                       +-------------+--------------+
                                     |
                                     v
                       +----------------------------+
Step 1                 | guardrails_service         |
                       | safety + malformed checks  |
                       +-------------+--------------+
                                     |
                     blocked? yes ---+----> return safe error
                                     |
                                     v
                       +----------------------------+
Step 2                 | cache_service (FAISS)      |
                       | semantic match + hit path  |
                       +-------------+--------------+
                                     |
                    cache hit? yes --+----> Step 7 evaluate
                                     |
                                     v
                       +----------------------------+
Step 3                 | router_service             |
                       | intent + model selection   |
                       +-------------+--------------+
                                     |
                                     v
                       +----------------------------+
Step 4                 | llm_service.check_budget   |
                       +-------------+--------------+
                                     |
                    budget exceeded? +----> cache-only mode
                                     |
                                     v
                       +----------------------------+
Step 5                 | llm_service.call_llm       |
                       | + cascade on low score     |
                       +-------------+--------------+
                                     |
                                     v
                       +----------------------------+
Step 6                 | guardrails post-check      |
                       | hallucination cascade      |
                       +-------------+--------------+
                                     |
                                     v
                       +----------------------------+
Step 7                 | evaluation_service         |
                       | judge + BLEU + semantic    |
                       +-------------+--------------+
                                     |
                                     v
                       +----------------------------+
Step 8                 | cache_service.store        |
Step 9                 | feedback_service.log       |
Step 10                | feedback retrain trigger   |
                       +-------------+--------------+
                                     |
                                     v
                               API Response
```

## Setup (Local, Step by Step)

1. Clone and enter project:
   - `cd /home/venkat_chiranjeevi/llm-cost-reducing/llm-optimizer`
2. Create environment file:
   - `cp .env.example .env`
3. Backend setup:
   - `cd backend`
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
4. Frontend setup:
   - `cd ../frontend`
   - `npm install`
5. Start backend:
   - `cd ../backend`
   - `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
6. Start frontend:
   - `cd ../frontend`
   - `npm run dev`

## Run Locally

- Backend: [http://localhost:8000](http://localhost:8000)
- Health: [http://localhost:8000/health](http://localhost:8000/health)
- Frontend dashboard: [http://localhost:3000](http://localhost:3000)

## Run With Docker

From repo root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Services:
- Postgres: `localhost:5432`
- Redis: `localhost:6379`
- Backend: `localhost:8000`
- Frontend: `localhost:3000`

## API Documentation

### `POST /api/query`
Runs the full 11-step optimization pipeline.

Request:
```json
{ "prompt": "user text" }
```

Response includes:
- `response`
- `model_used`
- `cache_hit`
- `compression` (`original_tokens`, `compressed_tokens`, `tokens_saved`)
- `routing_decision`
- `evaluation`
- `cost_usd`
- `latency_ms`
- `budget_remaining_percent`

### `POST /api/feedback`
Updates feedback signal for a query and checks retrain trigger.

Request:
```json
{ "query_id": "uuid", "feedback": 1 }
```

### `GET /api/query/stream`
Streaming response endpoint (chunked output).

### `GET /api/metrics`
System-wide metrics snapshot.

### `GET /api/router/history`
Retraining history for router accuracy trend.

### `GET /api/cache/stats`
Cache health and effectiveness metrics.

### `GET /api/models/comparison`
Per-model latency/quality/cost comparison table.

### `GET /api/budget/status`
Daily budget utilization and remaining percentage.

### `GET /api/ab-test/results`
A/B experiment summary by intent group.

### `GET /health`
Component-level health status.

## Benchmark Results (100 Query Run)

Source artifacts:
- `backend/benchmark_results.json`
- `backend/benchmark_chart.png`

Measured results:
- Total cost (optimizer): **$0.00006484**
- Baseline cost (all Gemini): **$0.00032570**
- Cost saved: **$0.00026086 (80.09%)**
- Avg latency (optimizer): **60.82 ms**
- Avg latency (baseline): **150.40 ms**
- Latency improvement: **2.47x**
- Cache hit rate: **80.00%**
- Router accuracy (test set): **80.00%**
- Average quality score: **1.0000**
- Cascade triggers: **0**
- Hallucinations caught: **0**
- Token savings: **180 tokens (26.47%)**

Run benchmark:

```bash
cd backend
python3 benchmark.py
```

## Online Learning Loop

1. Each query is logged with features, routing, cost, quality, and feedback.
2. `check_retrain_trigger()` fires retrain when:
   - query count is divisible by `RETRAIN_EVERY_N_QUERIES`, or
   - last-100 bad-response rate exceeds threshold.
3. `retrain_router()` trains on latest labeled samples.
4. Negative feedback examples are relabeled to a better model class.
5. New model replaces current model only if holdout accuracy improves.
6. Training event is appended to `router_training_log`.

## Dashboard Screenshots (Description)

- **Top bar**: service health + refresh status.
- **Cost Tracker**: total savings, baseline comparison, per-model spend.
- **Model Performance**: table of calls, latency, quality, cost, success rate.
- **Cache Metrics**: hit rate, entries, average quality, cache savings.
- **Router Accuracy Chart**: retrain event trend (key interview chart).
- **Latency Analytics**: cache hit vs miss area + model latency bars.
- **A/B Results**: intent-wise model winner analysis.
- **Budget Tracker**: utilization gauge, trend line, warning state.

## Tech Stack (Pinned)

- Python `3.11`
- FastAPI `0.104.1`
- Uvicorn `0.24.0`
- SQLAlchemy `2.0.23`
- AsyncPG `0.29.0`
- Redis `5.0.1`
- XGBoost `2.0.2`
- LightGBM `4.1.0`
- FAISS CPU `1.7.4`
- Sentence Transformers `2.2.2`
- NLTK `3.8.1`
- NumPy `1.26.2`
- Pandas `2.1.3`
- React `18.3.1`
- Recharts `2.15.3`
- Tailwind CSS `3.4.17`
- Vite `5.4.16`
- Docker / Docker Compose

## Interview Talking Points

- **Cost engineering**: dynamic router + semantic cache + compression reduced token spend.
- **Safety architecture**: pre-guardrails and post-response hallucination checks.
- **Adaptive intelligence**: online feedback loop retrains router from production signals.
- **Resilience**: graceful fallbacks for missing dependencies and low-confidence routes.
- **Observability**: per-query logging, budget controls, and real-time dashboarding.
- **Experimentation**: built-in A/B layer to validate route strategy improvements.




##One Liner

"Built a self improving LLM optimization platform 
routing across three real model providers with 
semantic caching, prompt compression and online 
learning — reducing API costs by 80% and improving 
latency 2.47x measured on real benchmark traffic, 
with a router that retrains automatically on 
production feedback signals."

## Key Questions

Q: How do you optimize LLM costs?
A: Three layers — prompt compression cuts tokens 26%, 
   semantic cache eliminates 80% of redundant calls, 
   smart routing sends simple queries to cheap models.

Q: How does your system improve over time?
A: Online learning loop logs every response quality. 
   Bad responses flip the routing label. Every 500 
   queries router retrains automatically on real data.

Q: How do you measure quality?
A: Four signal evaluation — LLM judge 50%, semantic 
   similarity 30%, BLEU 20%, plus hallucination 
   detection with automatic cascade to stronger model.

Q: Why not just use the best model always?
A: Benchmarks show 80% of queries need only Llama 3.1 
   8B at a fraction of the cost with same quality score. 
   Router learns which queries need the stronger model.

running "

backend:

uvicorn main:app --reload


frontend :


npm run dev
