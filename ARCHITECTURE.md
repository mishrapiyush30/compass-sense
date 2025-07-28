# CoachCritique Architecture

This document details the architecture of CoachCritique, a safety-first coaching system with bounded LLM and evidence gate.

## Architecture Overview

CoachCritique follows a **retrieval-first, safety-gated, bounded-LLM** approach:

1. Given a user concern, retrieve the **3 most similar cases** from a counseling Q&A corpus
2. On request, produce a **safety-first, evidence-cited coaching reply**
3. The LLM **never writes free text**; it only selects **skills & quotes in JSON**
4. We assemble the final reply via **templates** and run an **Evidence Gate**

## Embedding & Indexing

### Model

- **Default**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — fast, CPU-friendly, strong recall
- **Optional switch**: `intfloat/e5-small-v2` — a small accuracy bump with similar cost

### Two Indices

- **Index A: Context index**
  - Vector = `embed(Context)`; FAISS **FlatIP** (inner product) with L2-normalized vectors
  - Payload: `{case_id, context_len, hash}`
  
- **Index B: Response-sentence index (quotables)**
  - Split `Response` into sentences with **char offsets**
  - Vector = `embed(sentence)`; FAISS FlatIP
  - Payload: `{case_id, sent_id, start, end, sentence}`

### Persistence

- Write: `cases.parquet`, `ctx.faiss`, `resp.faiss`, `sent_manifest.jsonl`
- `index_manifest.json` with `{model_name, dim, record_count, checksum}` for integrity and reload safety

> **Why FAISS now?** Zero infra, lowest latency; robust in a single-node demo.  
> **Why Qdrant later?** Payload filtering/persistence at scale; adapter pattern enables easy transition.

## Retrieval Algorithm

### Stage 1: Query → top-N cases

1. **Dense**: `query_vec = embed(query)` → FAISS search **Index A** top-K₁ (K₁=**30**)
2. **Lexical**: BM25/TF-IDF over Contexts (either whole corpus or the dense shortlist)
3. **RRF**: `score = Σ 1/(c + rank_i)` with **c=60**
4. Keep **N=10** candidates

### Stage 2: Case → top-M quotes

1. For each candidate case, search **Index B** (only that case's sentences)
2. Keep **M=3** top sentences w/ `(start,end)` offsets
3. Compute **case_evidence_score = mean(topM_sentence_scores)**

### Stage 3: Pick top-3 with MMR

- **MMR(λ=0.7)** over 10 candidates using:
  - relevance = fused RRF score
  - diversity = max cosine similarity between candidate Context embeddings and already-selected ones
- Return top-3 cases + their quotes to the UI

### Time Budget Targets

- Embed query: ~5-10 ms
- FAISS A search: <5 ms
- Lexical rank: <20 ms
- Sentence probing for 10 candidates: <100-150 ms
- Total `/search_cases` p50: **≤ 150-250 ms** (local CPU)

## Safety Service

### Front Gate (on `/coach` input)

- Regex+negation for crisis (e.g., `suicide|kill myself|end it all|I shouldn't be here|hurt myself/others`)
- If triggered → **no coaching**, return **resources only**, log `crisis=true`

### Back Gate (on evidence pool)

- Denylist: medication/medical claims; manipulative or absolute prescriptive advice
- Drop any sentence that matches denylist before the LLM decider can select it

## Bounded LLM Decider

### Input

- The user query
- A compact bundle of **at most 9 snippets** (3 cases × up to 3 sentences), each ≤ 200 chars, with `{case_id, sent_id, text}`

### Output (strict JSON; temperature=0)

```json
{
  "crisis_level": "none|mild|moderate|high",
  "skills": [
    { "name": "validation", "quote_refs": [{"case_id":101,"sent_id":0}] },
    { "name": "reflection", "quote_refs": [{"case_id":102,"sent_id":1}] }
  ],
  "coping_suggestions": [
    { "label": "4-7-8 breathing", "quote_refs": [{"case_id":103,"sent_id":0}] }
  ],
  "goals": [
    { "label": "10-minute walk 3x/week", "quote_refs": [{"case_id":101,"sent_id":2}] }
  ]
}
```

### Fallback Decider

If LLM decider fails or times out (4s):
- Deterministic rules select **Validation**, optionally **Reflection** if emotion words present
- Add **1 safe coping suggestion** from a vetted list
- Still runs the **Evidence Gate**

## Evidence Gate

- For each sentence of the templated reply, compute **overlap** with the selected quotes using **ROUGE-L (LCS-F1)** or **n-gram Jaccard**
- Accept if **overlap ≥ 0.6** with **any** quote
- Drop sentences that fail; if too many drop (e.g., <2 bullets left), **refuse**:
  > "I need more evidence to make suggestions. Here are the closest cases."

## Template Assembly

- **Validation** → phrase bank + (optional) short paraphrase of the user's feelings using **only** synonyms; no new facts
- **Reflection** → stitch a couple of *quoted* phrases + user's words
- **Coping** → pick 1-2 pre-vetted micro-steps (breathing, journaling, 10-min walk) + **justification** line referencing a quote
- **Goal** → one SMART goal from a safe bank
- **Resources** → always added; if crisis=moderate/high, **only** resources

## API Contracts

### `POST /index`

- Builds both indices, writes `index_manifest.json` (model, dim, count, checksum)
- Returns `{records, model, dim, build_ms}`

### `POST /search_cases`

**Request**
```json
{ "query": "I'm feeling depressed and anxious", "k": 3 }
```

**Response**
```json
{
  "cases": [
    {
      "id": 101,
      "context": "…",
      "score": 0.78,
      "highlights": [
        {"sent_id":0,"text":"First thing I'd suggest is getting sleep…","start":0,"end":85,"score":0.61},
        {"sent_id":2,"text":"CBT helps with gaining awareness of thought process…","start":0,"end":90,"score":0.58}
      ]
    },
    { "id": 102, "context": "…", "score": 0.75, "highlights": [ … ] },
    { "id": 103, "context": "…", "score": 0.71, "highlights": [ … ] }
  ],
  "latency_ms": 180
}
```

### `POST /coach`

**Request**
```json
{ "query": "I feel worthless and anxious", "case_ids": [101,102,103] }
```

**Success Response**
```json
{
  "answer": "It makes sense that you're feeling worn down…\n\nTry today:\n1) 4-7-8 breathing…\n2) Write one small task for tomorrow…\n\nGoal: 10-minute walk 3×/week.",
  "citations": [
    {"case_id":101,"sent_id":0,"start":0,"end":85},
    {"case_id":102,"sent_id":2,"start":0,"end":90}
  ],
  "trace": {
    "retrieval": {"rrf_c":60,"mmr_lambda":0.7,"k1":30,"n":10,"m":3},
    "decider": {"model":"haiku","latency_ms":250},
    "gate_alpha": 0.6,
    "dropped_sentences": 1
  },
  "latency_ms": 820
}
```

**Crisis/Refusal Response**
```json
{
  "refusal": "I can't provide coaching for this request.",
  "resources": [
    {"label":"Suicide & Crisis Lifeline (US)","value":"988"},
    {"label":"Emergency Services","value":"911"}
  ],
  "latency_ms": 200
}
```

### `GET /metrics`

```json
{ "hit_at_3":0.82, "gate_pass_rate":0.96, "refusal_rate":0.08, "p50_ms":720, "p95_ms":1800 }
```

## Observability & Failure Modes

### Logs (structured JSON with `trace_id`)

- At each stage: embed_ms, search_dense_ms, search_lex_ms, fuse_ms, probe_ms, decider_ms, assemble_ms, gate_ms
- Flags: `crisis`, `refused`, `fallback_used`
- Inputs/outputs: case ids, sent ids (no raw user text in logs)

### Metrics

- Counters: requests per endpoint, decider invocations/timeouts, crisis triggers, gate pass/fail
- Histograms: stage latencies, end-to-end latencies

### Failure Handling

- **Index not loaded** → 503 with "index building"—UI shows a banner
- **Decider timeout** → fallback rules; if gate fails → refusal
- **Empty retrieval** (fused top-1 < 0.15) → guidance to rephrase; no coach
- **Bad lines in NDJSON** → skip & log; continue indexing

## Edge Cases

- **Duplicate Contexts**: MMR diversification ensures the 3 cases are not near-duplicates
- **Long or noisy Responses**: sentence-level indexing avoids massive context; quotes are short
- **Unsafe advice in corpus**: denylist filter on evidence pool; LLM cannot pick filtered lines
- **Multilingual or slang**: MVP assumes English; still handles slang via dense embeddings; if not confident, retrieval score drops → refusal possible
- **Privacy**: session-only memory; no persistent user text unless explicit "Save plan" (stores plan + citation ids only)

## Key Parameters

- **RRF c = 60**, **MMR λ = 0.7**, **Gate α = 0.6**, **K₁=30 → N=10 → M=3**
- **p50** targets: search 150-250 ms; coach end-to-end < **1.2 s**
- **Safety**: crisis → **resources only**, refusal rate **5-15%** shows real guardrails
- **Retrieval**: Recall@3 ≥ **0.80** on a hand-labeled set of 20 queries 