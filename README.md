# RAG: Bi-Encoder Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines compared over the same chunked corpus, the
[RFC 6749 — OAuth 2.0 Authorization Framework](https://www.rfc-editor.org/rfc/rfc6749) (76-page PDF):

|          | Pipeline 1 — Baseline                                                                                      | Pipeline 2 — Re-ranking                                                           |
| -------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Retrieve | `sentence-transformers/all-MiniLM-L6-v2` bi-encoder + FAISS (inner product on normalised vectors) → top 10 | same top 10                                                                       |
| Select   | top 3 by cosine similarity                                                                                 | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs → top 3 |
| Generate | local `qwen2.5:7b-instruct` (Ollama), shared prompt → Answer 1                                             | same LLM, identical prompt → Answer 2                                             |

Everything — parsing, chunking, both pipelines, the failure-mode evaluation (34 labelled queries),
LLM answer comparison, latency profiling (GPU and CPU) and the scaling experiment — is in
**[`rag_rerank.ipynb`](rag_rerank.ipynb)**, committed with its outputs.

## Repository layout

```
rag_rerank.ipynb         # the whole assignment, commented step by step
data/rfc6749_oauth2.pdf  # source document
results/                 # CSV/JSON outputs written by the notebook
  rank_shifts.csv        #   gold-chunk rank per query, before/after re-ranking
  retrieval_summary.csv  #   Hit@1, Hit@3, Recall@10, MRR@10 for both pipelines
  by_category.csv        #   the same broken down by failure category
  case_studies.json      #   top-10 tables + Answer 1 vs Answer 2 for the analysed queries
  latency*.csv           #   per-stage latency (median / p95), GPU and CPU
  scaling*.csv           #   cross-encoder cost over the whole corpus + extrapolation
  llm_answers.json       #   cached LLM answers + generation times (lets the notebook re-run without Ollama)
  figures/rank_shift.png
report/report.pdf        # final report (architecture, rank shifts, latency, scaling)
requirements.txt
```

## Running it

```bash
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# local LLM (https://ollama.com) - only needed to regenerate answers; cached ones are used otherwise
ollama serve &
ollama pull qwen2.5:7b-instruct        # ~4.7 GB, 4-bit; fits in 8 GB of VRAM

.venv/bin/jupyter lab rag_rerank.ipynb
```

Everything runs locally: no API keys or cloud services. A CUDA GPU is used when available; retrieval
latency is profiled on both GPU and CPU.
