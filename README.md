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
"AI kept forgetting" ──caused──▶ "User was frustrated"
│
caused
▼
"User built Engram"

Engram detects causal relationships automatically.
No labels. No human annotation. Pure inference.

This is narrative memory.
The closest current AI gets to how humans actually remember.

---

## Benchmark Results

**Personalization Benchmark — what Engram is actually built for:**
metric                no memory   flat memory   engram
overall accuracy         0.40        1.00        1.00
personalization          0.29        1.00        1.00
causal understanding     0.50        1.00        1.00
noise rejection          1.00        1.00        1.00
memory retained           100%        100%        28%

Engram matches flat memory on every accuracy dimension
while retaining only 28% of memories.

Same intelligence. 72% less storage. 72% faster retrieval.

**LoCoMo benchmark — industry standard, 1,542 QA pairs:**
no memory baseline  : 0.006
flat memory         : 0.209
engram              : 0.211  (+0.9%)

**Internal forgetting quality:**
forgetting quality      : 0.94 / 1.00
irreplaceability kept   : 0.79
irreplaceability pruned : 0.05
avg query time          : 3.9ms
queries per second      : 259
embedding cache speedup : 2669x

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

Below 0.20  → pruned
Above 0.40  → active
Archived    → never pruned, ever

---

## Quick Start

```bash
git clone https://github.com/deekshith080/engram-public
cd engram-public
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python chat.py
```

---

## Run Benchmarks

```bash
python benchmark_personalization.py
python benchmark.py
python benchmark_locomo.py
```

---

## Run Tests

```bash
python -m pytest tests/ -v
# 51 passed
```

---

## Live API

Engram is live and callable right now:
https://web-production-07b0a4.up.railway.app

## Live Demo

**Dashboard:** https://web-production-07b0a4.up.railway.app
**API:** https://web-production-07b0a4.up.railway.app/v1
**Interactive docs:** https://web-production-07b0a4.up.railway.app/docs

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
v0.1  ✅  Core engine — memory, scoring, graph, decay
v0.2  ✅  Persistence, semantic embeddings, cache
v0.3  ✅  Causal chain detection, narrative memory
v0.4  ✅  Conversation-aware irreplaceability detection
v0.5  ⬜  Memory consolidation
v0.6  ⬜  Temporal reasoning
v0.7  ⬜  Significance scoring — detect moments of decision and emotion
v0.8  ⬜  Reconstructive retrieval
v0.9  ⬜  Sleep consolidation
v1.0  ⬜  Production deployment

---

## Mission

Memory belongs to the user.
Engram is the guardian, not the owner.
Built to make the world better.

---

## Status

Active research project.
Built from scratch. No shortcuts. No tradeoffs.
Contact: cdeekshith1@gmail.com