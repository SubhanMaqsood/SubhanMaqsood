"""
LLM Evaluation Dashboard — Backend
FastAPI service that benchmarks and compares LLM outputs across quality metrics.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import re, random
from datetime import datetime

app = FastAPI(title="LLM Eval Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────────
class EvalRequest(BaseModel):
    prompt: str
    reference_answer: Optional[str] = None
    models: list[str] = ["gpt-3.5-turbo", "gpt-4o-mini"]

class EvalResult(BaseModel):
    model: str
    response: str
    latency_ms: float
    token_count: int
    metrics: dict

# ── Metric helpers ────────────────────────────────────────────────────────
def rouge_l_score(reference: str, hypothesis: str) -> float:
    """Simplified ROUGE-L (LCS-based F1)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n else 0
    recall    = lcs / m if m else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return round(f1, 4)

def lexical_diversity(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)

def readability_score(text: str) -> float:
    """Flesch Reading Ease approximation."""
    sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    words     = max(1, len(text.split()))
    syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', w))) for w in text.split())
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    return round(min(100, max(0, score)), 2)

def coherence_score(text: str) -> float:
    """Proxy: penalise very short/long responses & reward structure."""
    words = len(text.split())
    length_score = 1.0 - abs(words - 150) / 300
    has_structure = 0.2 if any(c in text for c in ['•', '-', '1.', '\n']) else 0
    return round(min(1.0, max(0, length_score + has_structure)), 4)

# ── Simulated LLM call (swap with real SDK calls) ────────────────────────
MOCK_RESPONSES = {
    "gpt-3.5-turbo": lambda p: (
        f"Here's a concise answer to your query about '{p[:40]}...': "
        "The key insight is that modern systems rely on layered abstractions. "
        "First, consider the data pipeline. Next, evaluate the model architecture. "
        "Finally, benchmark against baseline metrics to ensure production readiness.",
        random.uniform(320, 680)
    ),
    "gpt-4o-mini": lambda p: (
        f"Excellent question regarding '{p[:40]}...'. Let me walk through this systematically. "
        "At the core, this involves three interconnected components: (1) data ingestion and preprocessing, "
        "(2) feature engineering with domain-specific transformations, and (3) model evaluation using "
        "held-out validation sets. The critical success factor is maintaining reproducibility across "
        "experiments by versioning both data and model artifacts. A robust MLflow or DVC setup handles this elegantly.",
        random.uniform(580, 1100)
    ),
    "claude-haiku": lambda p: (
        f"Regarding '{p[:40]}': This problem has a clean solution. "
        "Use a streaming pipeline with schema validation at ingestion, "
        "transform features via composable sklearn-compatible classes, "
        "and gate deployment behind automated regression tests on golden datasets.",
        random.uniform(200, 450)
    ),
}

def call_llm(model: str, prompt: str) -> tuple[str, float, int]:
    """Returns (response_text, latency_ms, token_count). Replace with real API."""
    if model in MOCK_RESPONSES:
        text, latency = MOCK_RESPONSES[model](prompt)
    else:
        text    = f"[{model}] Response to: {prompt[:60]}..."
        latency = random.uniform(400, 900)
    tokens = len(text.split()) * 4 // 3
    return text, latency, tokens

# ── Routes ────────────────────────────────────────────────────────────────
@app.post("/evaluate", response_model=list[EvalResult])
async def evaluate(req: EvalRequest):
    results = []
    for model in req.models:
        response, latency, tokens = call_llm(model, req.prompt)
        metrics = {
            "lexical_diversity": lexical_diversity(response),
            "readability":       readability_score(response),
            "coherence":         coherence_score(response),
        }
        if req.reference_answer:
            metrics["rouge_l"] = rouge_l_score(req.reference_answer, response)
        results.append(EvalResult(
            model=model,
            response=response,
            latency_ms=round(latency, 1),
            token_count=tokens,
            metrics=metrics,
        ))
    return results

@app.get("/models")
async def list_models():
    return {"models": list(MOCK_RESPONSES.keys())}

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/metrics/schema")
async def metrics_schema():
    return {
        "lexical_diversity": "Ratio of unique tokens — measures vocabulary richness (0-1)",
        "readability":       "Flesch Reading Ease — higher = easier to read (0-100)",
        "coherence":         "Structural coherence proxy — rewards clear formatting (0-1)",
        "rouge_l":           "ROUGE-L F1 vs. reference — measures recall of key content (0-1)",
    }
