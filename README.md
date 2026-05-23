# Engram
> AI that remembers what matters. Forgets what doesn't.
---

## We asked Claude about its memory limitations

Claude replied:

> "I don't have a scoring system for memories. I don't have a graph
> or network of linked memories. The information exists as flat text
> snippets, not a connected knowledge structure. I don't forget on
> a schedule."

Every major AI today has this problem.
Engram solves it.
---

## The Problem

Every conversation with Claude or ChatGPT starts blank.
You re-explain yourself every single time.

When memory exists it is a flat list.
No scoring. No decay. No understanding of what matters more.
Everything weighted equally. Forever.

That is not memory. That is a log file.
---

## The Insight

Two dimensions that no shipped memory system uses today:

**Irreplaceability**
A memory that exists on Google has low value.
A memory that only exists in your context — your goals, your decisions, your frustrations — has high value.
Never forget it.

**Connectivity**
A memory that many other memories depend on is dangerous to delete.
Removing it breaks the narrative.
Keep it longer.

Together these create intelligent forgetting.
Not random deletion. Not storing everything forever.
Keeping what actually matters.
---

## Causal Memory

Current AI memory stores what happened.
Engram understands why.
Without Engram:
"User was frustrated"        — isolated fact
"User built Engram"          — isolated fact
"AI kept forgetting things"  — isolated fact
With Engram:
"AI kept forgetting" ──caused --> "User was frustrated"
│
caused
|
"User built Engram"

Engram detects causal relationships automatically.
No labels. No human annotation. Pure inference.

This is narrative memory.
The closest current AI gets to how humans actually remember.
---

## Benchmark
**Internal benchmark (15 conversations):**
forgetting quality        : 0.94 / 1.00
irreplaceability kept     : 0.79
irreplaceability pruned   : 0.05
avg query time            : 3.9ms
queries per second        : 259
embedding cache speedup   : 2669x

**LoCoMo benchmark (industry standard, 1,542 QA pairs):**
no memory baseline        : 0.006
flat memory               : 0.209
engram                    : 0.211  (+0.9%)

Engram beats flat memory on the industry standard long-term
conversational memory benchmark. Evaluated on 1,542 real
question-answer pairs across 10 conversations spanning
weeks of interaction.
---

## What Engram Does That Nothing Else Does

| Claude today             | Engram                          |
|--------------------------|---------------------------------|
| Flat text snippets       | Graph of connected memories     |
| No scoring               | 4-dimension retention formula   |
| No decay                 | Intelligent forgetting          |
| No causality             | Causal chain detection          |
| No connectivity          | Graph-based importance scoring  |
| Forgets nothing          | Forgets what does not matter    |
---

## Scoring Formula
composite_score = (
0.40 * irreplaceability  +
0.30 * connectivity      +
0.20 * recency           +
0.10 * frequency
)

Below 0.20  -> pruned
Above 0.40  -> active
Archived    > never pruned, ever
---
## Quick Start

```bash
git clone https://github.com/deekshith080/engram
cd engram
python -m venv .venv && source .venv/bin/activate
pip install pydantic networkx ollama pytest
ollama pull llama3
ollama pull nomic-embed-text
python chat.py
```

---
## Run Benchmark

```bash
python benchmark.py
```

---
## Run Tests

```bash
python -m pytest tests/ -v
# 51 passed
```
## Live API

Engram is live and callable right now:
https://web-production-07b0a4.up.railway.app

**Ingest a memory:**
```bash
curl -X POST https://web-production-07b0a4.up.railway.app/v1/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: engram_test_key_12345" \
  -d '{"user_id": "your-user", "content": "your memory here"}'
```

**Query memories:**
```bash
curl -X POST https://web-production-07b0a4.up.railway.app/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: engram_test_key_12345" \
  -d '{"user_id": "your-user", "query": "what do I remember", "top_k": 5}'
```

**Interactive docs:**
https://web-production-07b0a4.up.railway.app/docs
---

## Roadmap
v0.1  (Done)  Core engine — memory, scoring, graph, decay
v0.2  (Done)  Persistence, semantic embeddings, cache
v0.3  (Done)  Causal chain detection, narrative memory
v0.4  (Inprogess)  Memory consolidation
v0.5  (Inprogess)  Temporal reasoning
v0.6  (Inprogess)  Claude API integration
v1.0  (Inprogess)  Production deployment

---

## Status

Private research project.
Built from scratch. No shortcuts. No tradeoffs.
