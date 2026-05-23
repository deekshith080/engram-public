# AI memory should forget. Here's how I built intelligent decay.

I've been frustrated with AI memory for a while.

Every conversation with Claude or ChatGPT starts completely blank. 
You explain yourself again and again. Your name, your project, your 
preferences — gone after every session.

When memory does exist, it's a flat list. No scoring. No decay. 
No understanding of what matters more than something else. 
Everything weighted equally. Forever.

So I asked Claude directly about its own memory limitations. It said:

> "I don't have a scoring system for memories. I don't have a graph 
> or network of linked memories. The information exists as flat text 
> snippets, not a connected knowledge structure. I don't forget on 
> a schedule."

That answer became the spec for what I built next.

---

## The insight that changed everything

Human memory isn't a database. It's a narrative.

We remember things because they matter. We forget things because they 
don't. Nobody remembers every email they've ever read. But everyone 
remembers the day they decided to quit their job.

Current AI memory treats both equally. That's the problem.

I came up with two dimensions that no shipped memory system uses:

**Irreplaceability**

Can this memory be Googled? A definition of Python exists everywhere 
on the internet — low value, forget it fast. But "I'm building an AI 
memory system because I was frustrated with AI amnesia" — that exists 
nowhere else. That's personal context. Never forget it.

**Connectivity**

What breaks if we delete this memory? A memory that five other memories 
depend on is more dangerous to delete than an isolated one. Remove it 
and you break the narrative. This pushed the architecture toward a 
knowledge graph — not just a list.

Combined, these two dimensions create intelligent forgetting. Not random 
deletion. Not storing everything forever. Keeping what actually matters.

---

## The scoring formula

Every memory gets a composite score:
score = 0.40 × irreplaceability
+ 0.30 × connectivity
+ 0.20 × recency
+ 0.10 × frequency

Memories below 0.20 get pruned. Memories above 0.40 stay active. 
Archived memories — ones the user pins — never get pruned regardless 
of score.

The weights aren't arbitrary. Irreplaceability gets the highest weight 
because it captures something the other dimensions can't — uniqueness. 
A memory accessed yesterday but findable on Google is less valuable than 
a personal decision made six months ago that nobody else knows about.

---

## Causal chains — the part I'm most excited about

Current AI memory stores what happened. I wanted to understand why.

When someone says "I was frustrated because AI kept forgetting everything, 
so I decided to build Engram" — there's a causal chain buried in that sentence:
"AI kept forgetting"  ──caused──▶  "frustration"
│
caused
▼
"decided to build Engram"

I built two detection methods. The first looks for explicit causal language 
— words like "because", "so I", "which led to", "decided to". The second 
uses semantic embeddings to detect causal flow between temporally ordered 
memories even when no causal words are present.

No labels. No human annotation. Pure inference from the text.

This is the closest current AI gets to how humans actually remember — 
not just facts, but the narrative connecting them.

---

## The honest benchmark results

I ran against LoCoMo — the industry standard long-term conversational 
memory benchmark. 1,542 real question-answer pairs across 10 conversations 
spanning weeks of interaction.
no memory baseline  : 0.006
flat memory         : 0.209
engram              : 0.211  (+0.9%)

Small margin. I'm not going to pretend otherwise.

What I am confident about is forgetting quality — 0.94 on internal 
benchmarks. Engram consistently keeps irreplaceable personal memories 
and prunes generic re-fetchable ones. That's the core thesis working.

The retrieval gap is what I'm working on next. The intelligent forgetting 
is real. Getting it to surface the right memories at query time is the 
next frontier.

---

## The live API

Engram is live right now. Three endpoints:

```bash
# Store a memory
POST /v1/ingest
{"user_id": "alice", "content": "I prefer dark mode and hate meetings"}

# Retrieve relevant memories  
POST /v1/query
{"user_id": "alice", "query": "what does alice prefer"}

# Run intelligent forgetting
POST /v1/decay
{"user_id": "alice"}
```

Any AI app gets persistent intelligent memory with three API calls.

Email **cdeekshith1@gmail.com** for API access.

---

## What's next

Better irreplaceability detection — rule-based works but misses subtlety. 
Claude API integration for richer semantic understanding. Improving the 
retrieval to use graph edges more aggressively.

The benchmark number is small but the direction is right. The pieces 
are in place. Now I need real users breaking it in ways I haven't 
thought of.

If you're building AI apps and want memory that actually forgets 
intelligently — try it. Break it. Tell me what's wrong.

**GitHub:** https://github.com/deekshith080/engram

Built this solo. Feedback welcome.