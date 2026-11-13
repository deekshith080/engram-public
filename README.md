# Engram
> AI that remembers what matters. Forgets what doesn't.

Engram is a memory API for AI applications. Send it what your users say;
ask it what matters about them later. It keeps what's important, links
related memories, understands cause and effect, and lets the rest fade —
the way human memory works.

**Live API:** https://engram-c5m6.onrender.com/v1
**Interactive docs:** https://engram-c5m6.onrender.com/docs

---

## We asked Claude about its memory limitations

Claude replied:

> "I don't have a scoring system for memories. I don't have a graph
> or network of linked memories. The information exists as flat text
> snippets, not a connected knowledge structure. I don't forget on
> a schedule."

Every major AI today has this problem. Engram solves it.

---

## The problem

Every conversation with an AI assistant starts blank. You re-explain
yourself every single time. When memory exists, it's a flat list — no
scoring, no decay, no sense of what matters more. Everything weighted
equally, forever.

That is not memory. That is a log file.

## The idea

**Irreplaceability.** A fact you could look up anywhere has low value. A
memory that only exists in your context — your goals, your decisions,
your frustrations — has high value. Never forget it.

**Connectivity.** A memory that many others depend on is dangerous to
delete; removing it breaks the story. Keep it longer.

Together they give **intelligent forgetting**: not deleting at random,
not storing everything forever — keeping what actually matters.

**Causal memory.** Engram doesn't just store what happened; it links why.
"The AI kept forgetting" → *caused* → "the user was frustrated" →
*caused* → "the user built Engram". Detected automatically, no labels.

---

## Results

**Personalization benchmark**

| metric | no memory | flat memory | Engram |
|---|---|---|---|
| overall accuracy | 0.40 | 1.00 | 1.00 |
| personalization | 0.29 | 1.00 | 1.00 |
| causal understanding | 0.50 | 1.00 | 1.00 |
| noise rejection | 1.00 | 1.00 | 1.00 |
| memories retained | 100% | 100% | **28%** |

Same accuracy as keeping everything, while keeping only 28% of memories.

**LoCoMo** (industry standard, 1,542 QA pairs): no memory 0.006 · flat
memory 0.209 · Engram 0.211.

**Speed:** ~3.9 ms per query, ~259 queries per second.

---

## Try it

### 1. Get an API key

```bash
curl -X POST https://engram-c5m6.onrender.com/v1/request-key \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com"}'
```

Your key arrives by email. Keep it safe — keys can't be recovered, only
re-requested. (The free server sleeps when idle, so the first request
can take up to a minute.)

### 2. Store a memory

```bash
curl -X POST https://engram-c5m6.onrender.com/v1/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"user_id": "alice", "content": "I am training for a marathon in April"}'
```

`user_id` is *your* app's end user — each one gets their own memory, and
only your key can see your users.

### 3. Ask what matters

```bash
curl -X POST https://engram-c5m6.onrender.com/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"user_id": "alice", "query": "what is Alice working towards?", "top_k": 5}'
```

More in [`examples/`](examples/) — a curl walkthrough and a short Python
script. Every endpoint is documented at
https://engram-c5m6.onrender.com/docs.

---

## Endpoints

| method | path | what it does |
|---|---|---|
| POST | `/v1/request-key` | email yourself an API key |
| POST | `/v1/ingest` | store what a user said |
| POST | `/v1/query` | the memories most relevant to a question |
| POST | `/v1/recall` | memories surfaced by association — one memory bringing up related ones |
| GET | `/v1/memories/{user_id}` | everything kept for a user |
| GET | `/v1/timeline/{user_id}` | a user's memories summarized by time period |
| POST | `/v1/predict` | memories likely to be relevant to the current conversation |
| POST | `/v1/decay` | run a forgetting cycle for a user now — weak memories are pruned |

---

## Mission

Memory belongs to the user. Engram is the guardian, not the owner.

Questions or feedback: cdeekshith1@gmail.com
