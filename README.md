---
title: AI JEE Study Agent
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI JEE Study Agent

A bilingual (**Hindi / English**) AI study assistant for JEE preparation, built as a
**retrieval-augmented generation (RAG)** service. Ask about the JEE Main/Advanced exam
pattern, eligibility, physics/chemistry/maths formulas or preparation strategy, and the
agent answers **only from its study material**, refusing instead of guessing when it has
no reliable source.

**Stack:** Python · FastAPI · LangChain · FAISS · Sentence-Transformers · Docker · Hugging Face Spaces

## How it works

```
 question ──► multilingual embedding ──► FAISS top-k search
                                              │
                          drop chunks with distance > SCORE_THRESHOLD   ◄── weak-match filter
                                              │
                     ┌────────────────────────┴───────────────┐
              no chunks left                           chunks remain
        "I couldn't find reliable info"        LLM answers from the context
        (in the question's language)           (or extractive mode with no API key)
```

- **Chunking:** `RecursiveCharacterTextSplitter` (700 chars, 120 overlap) over `.txt` / `.pdf` files in `data/`.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2`, so Hindi and English share one index.
- **Weak-match filter:** FAISS L2 distance is thresholded, so low-relevance chunks never reach the user.
- **Language:** replies in the language of the question (Devanagari detection).

## Project structure

```
app/
  config.py     settings (env-overridable)
  ingest.py     loads data/, chunks, embeds, saves FAISS index
  rag.py        JEEAgent: retrieval + filtering + answer generation
  main.py       FastAPI service (+ minimal web UI at /)
data/           knowledge base (.txt / .pdf)  ← add your own material here
evaluation/     questions.json (30-question evaluation set)
evaluate.py     coverage / correctness / rejection report
analyze_scores.py   helps choose SCORE_THRESHOLD
tests/          offline unit tests
Dockerfile
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optionally add OPENAI_API_KEY
python -m app.ingest            # build the FAISS index
uvicorn app.main:app --port 7860
```

Open http://localhost:7860 for the UI or http://localhost:7860/docs for the API docs.

```bash
curl -X POST localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"question": "JEE Main में नेगेटिव मार्किंग कितनी है?"}'
```

## Run with Docker

```bash
docker build -t jee-agent .
docker run -p 7860:7860 --env-file .env jee-agent
```

## Evaluation

```bash
python evaluate.py          # coverage, correctness and out-of-scope rejection
python analyze_scores.py    # score distribution to calibrate SCORE_THRESHOLD
pip install -r requirements-dev.txt && pytest   # offline unit tests
```

`evaluation/questions.json` has 30 questions (English + Hindi, plus out-of-scope ones that
should be rejected). Re-run these after changing the data, chunking or threshold.

## Deploy to Hugging Face Spaces

1. Create a new Space → SDK **Docker**.
2. Push this repository to the Space (the YAML block at the top of this README configures it).
3. (Optional) Add `OPENAI_API_KEY` (and `OPENAI_BASE_URL` / `LLM_MODEL`) under *Settings → Secrets*.

## Extending the knowledge base

Drop more `.txt` or `.pdf` files into `data/` and rebuild with `python -m app.ingest`.
The bundled notes are concise study summaries; always verify exam dates and pattern on
[jeemain.nta.nic.in](https://jeemain.nta.nic.in) and [jeeadv.ac.in](https://jeeadv.ac.in).
