# LLM Evaluation Dashboard

> Compare and benchmark LLM outputs across quality metrics — latency, lexical diversity, readability, coherence, and ROUGE-L.

## What it Does
Submit a prompt, select 2–3 models, and get side-by-side metric scores with a radar chart comparison. Optionally provide a reference answer to enable ROUGE-L scoring.

## Skills Demonstrated
- **NLP metrics**: ROUGE-L (LCS-based), Flesch Reading Ease, lexical diversity
- **REST API design**: FastAPI with Pydantic validation, CORS, typed responses
- **Frontend data viz**: Chart.js radar chart, real-time metric bars
- **Software engineering**: Modular metric functions, unit-tested with clear assertions

## Tech Stack
| Layer | Tech |
|-------|------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Frontend | Vanilla JS, Chart.js |
| Testing | Python unittest (stdlib) |
| Deploy | Railway / Render (backend) + Vercel (frontend) |

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Tests
cd tests
python test_metrics.py

# Frontend
# Open frontend/index.html in a browser
# (works in demo mode even without the backend)
```

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/evaluate` | Run evaluation across selected models |
| GET | `/models` | List available models |
| GET | `/metrics/schema` | Describe what each metric measures |
| GET | `/health` | Health check |

## To Make it Production-Ready
1. Replace `MOCK_RESPONSES` in `main.py` with real OpenAI/Anthropic SDK calls
2. Add a `.env` file with API keys (use `python-dotenv`)
3. Deploy backend to Render, frontend to Vercel
4. Add a `/history` endpoint with SQLite persistence (use `aiosqlite`)

## Interview Talking Points
- "I implemented ROUGE-L from scratch using dynamic programming (LCS) rather than a library — it helped me understand what the metric actually measures."
- "The frontend works in demo mode without a backend, which made it easy to iterate on UX independently."
- "I kept metric logic pure (no side effects) so unit testing is trivial."
