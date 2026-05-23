"""
Tests Engram against the LoCoMo long-term conversational memory benchmark.
Uses LLM answer extraction with temperature=0 for deterministic results.
"""

import json
import random
import string
from dataclasses import dataclass

import numpy as np
import ollama

from engram.core.ingestion import IngestionPipeline
from engram.core.memory import MemoryNode
from engram.core.retrieval import RetrievalEngine
from engram.graph.auto_edge import AutoEdgeCreator
from engram.graph.causal import CausalChainBuilder
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayScheduler
from engram.scoring.engine import ScoringConfig
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.store import InMemoryStore

# Deterministic results
random.seed(42)
np.random.seed(42)

DB_PATH = "locomo_benchmark.db"


# Data structures

@dataclass
class QAPair:
    question: str
    answer:   str


@dataclass
class Session:
    date:  str
    turns: list[dict]


@dataclass
class Conversation:
    sample_id: str
    sessions:  list[Session]
    qa_pairs:  list[QAPair]


# Loader

def load_locomo(path: str = "locomo10.json") -> list[Conversation]:
    with open(path) as f:
        raw = json.load(f)

    conversations = []
    for item in raw:
        conv      = item["conversation"]
        conv_keys = list(conv.keys())

        session_keys = sorted(
            [k for k in conv_keys
             if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1])
        )

        sessions = []
        for key in session_keys:
            date  = conv.get(f"{key}_date_time", "unknown")
            turns = conv[key] if isinstance(conv[key], list) else []
            sessions.append(Session(date=date, turns=turns))

        qa_pairs = [
            QAPair(
                question = str(qa.get("question", "")).strip(),
                answer   = str(qa.get("answer",   "")).strip(),
            )
            for qa in item.get("qa", [])
            if qa.get("question") and qa.get("answer")
        ]

        conversations.append(Conversation(
            sample_id = str(item.get("sample_id", "")),
            sessions  = sessions,
            qa_pairs  = qa_pairs,
        ))

    return conversations[:2]  # subset for fast iteration


# Scoring

def normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def answer_matches(predicted: str, ground_truth: str) -> bool:
    pred  = normalise(predicted)
    truth = normalise(ground_truth)
    return truth in pred or pred in truth


# LLM answer extraction — temperature=0 for determinism

def extract_answer(question: str, memory_context: str) -> str:
    """Use Llama3 with temperature=0 for deterministic answers."""
    try:
        response = ollama.chat(
            model   = "llama3",
            options = {"temperature": 0},
            messages = [
                {
                    "role":    "system",
                    "content": (
                        "You are answering questions about conversations. "
                        "Use only the provided memories to answer. "
                        "Be concise — answer in as few words as possible. "
                        "If the answer is not in the memories, say 'unknown'."
                    ),
                },
                {
                    "role":    "user",
                    "content": (
                        f"Memories:\n{memory_context}\n\n"
                        f"Question: {question}\n\n"
                        f"Answer in as few words as possible:"
                    ),
                },
            ],
        )
        return response["message"]["content"].strip()
    except Exception:
        return ""


# Pre-warm cache

def prewarm_cache(conversations: list[Conversation]) -> None:
    cache    = EmbeddingCache(DB_PATH)
    pipeline = IngestionPipeline(use_llm=False)
    total    = 0

    print("pre-warming embedding cache...")

    for conv in conversations:
        for session in conv.sessions:
            for turn in session.turns:
                text = turn.get("text", "").strip()
                if not text:
                    continue
                result = pipeline.ingest(text)
                for node in result.nodes:
                    if not cache.exists(node.content):
                        cache.get(node.content)
                        total += 1

        for qa in conv.qa_pairs:
            if not cache.exists(qa.question):
                cache.get(qa.question)
                total += 1

    print(f"cached {total} new embeddings")
    cache.close()


# No memory baseline

def run_no_memory_baseline(conversations: list[Conversation]) -> dict:
    total   = 0
    correct = 0
    for conv in conversations:
        for qa in conv.qa_pairs:
            total    += 1
            predicted = extract_answer(qa.question, "No memories available.")
            if answer_matches(predicted, qa.answer):
                correct += 1
    return {
        "name":     "no memory",
        "total":    total,
        "correct":  correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }


# Flat memory baseline

def run_flat_memory_baseline(conversations: list[Conversation]) -> dict:
    pipeline = IngestionPipeline(use_llm=False)
    total    = 0
    correct  = 0

    for conv in conversations:
        engine:    RetrievalEngine  = RetrievalEngine(top_k=10, db_path=DB_PATH)
        all_nodes: list[MemoryNode] = []

        for session in conv.sessions:
            for turn in session.turns:
                text = turn.get("text", "").strip()
                if not text:
                    continue
                result = pipeline.ingest(text)
                all_nodes.extend(result.nodes)

        for qa in conv.qa_pairs:
            total   += 1
            results  = engine.query(qa.question, all_nodes)
            if not results:
                predicted = extract_answer(qa.question, "No memories available.")
            else:
                context   = "\n".join(r.node.content for r in results)
                predicted = extract_answer(qa.question, context)
            if answer_matches(predicted, qa.answer):
                correct += 1

    return {
        "name":     "flat memory",
        "total":    total,
        "correct":  correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }


# Engram

def run_engram(conversations: list[Conversation]) -> dict:
    pipeline = IngestionPipeline(use_llm=False)
    config   = ScoringConfig(prune_threshold=0.05, decay_threshold=0.15)
    total    = 0
    correct  = 0

    for conv in conversations:
        engine  = RetrievalEngine(top_k=10, db_path=DB_PATH)
        store   = InMemoryStore()
        graph   = GraphManager()
        creator = AutoEdgeCreator(graph, db_path=DB_PATH)
        builder = CausalChainBuilder(graph, db_path=DB_PATH)

        all_nodes: list[MemoryNode] = []

        for session in conv.sessions:
            for turn in session.turns:
                text = turn.get("text", "").strip()
                if not text:
                    continue
                result = pipeline.ingest(text)
                for node in result.nodes:
                    store.save(node)
                    graph.add_node(node)
                    creator.connect(node, all_nodes)
                    builder.process(node, all_nodes)
                    all_nodes.append(node)

        scheduler = DecayScheduler(store, graph, config)
        scheduler.run()
        surviving = store.get_all()

        active = [n for n in surviving if n.status.value == "active"]
        pruned = [n for n in surviving if n.status.value == "pruned"]
        print(f"  memories: {len(all_nodes)} total, "
              f"{len(active)} active, {len(pruned)} pruned")

        for qa in conv.qa_pairs:
            total   += 1
            results  = engine.query(qa.question, active, graph)
            if not results:
                predicted = extract_answer(qa.question, "No memories available.")
            else:
                context   = "\n".join(r.node.content for r in results)
                predicted = extract_answer(qa.question, context)
            if answer_matches(predicted, qa.answer):
                correct += 1

    return {
        "name":     "engram",
        "total":    total,
        "correct":  correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }


# Main

def main() -> None:
    print()
    print("=== Engram LoCoMo Benchmark ===")

    print("loading dataset...")
    conversations = load_locomo()
    print(f"conversations : {len(conversations)}")
    print(f"sessions      : {sum(len(c.sessions) for c in conversations)}")
    print(f"qa pairs      : {sum(len(c.qa_pairs) for c in conversations)}")
    print()

    prewarm_cache(conversations)
    print()

    print("running no memory baseline...")
    no_memory = run_no_memory_baseline(conversations)
    print(f"accuracy : {no_memory['accuracy']:.3f}")
    print()

    print("running flat memory baseline...")
    flat = run_flat_memory_baseline(conversations)
    print(f"accuracy : {flat['accuracy']:.3f}")
    print()

    print("running Engram...")
    engram = run_engram(conversations)
    print(f"accuracy : {engram['accuracy']:.3f}")
    print()

    print("=== Results ===")
    print(f"no memory baseline : {no_memory['accuracy']:.3f}")
    print(f"flat memory        : {flat['accuracy']:.3f}")
    print(f"engram             : {engram['accuracy']:.3f}")
    print()

    if engram["accuracy"] > flat["accuracy"]:
        improvement = (
            (engram["accuracy"] - flat["accuracy"])
            / flat["accuracy"] * 100
        )
        print(f"Engram beats flat memory by : +{improvement:.1f}%")
    else:
        gap = (
            (flat["accuracy"] - engram["accuracy"])
            / max(flat["accuracy"], 0.001) * 100
        )
        print(f"Engram trails flat memory by : -{gap:.1f}%")
        print("this tells us exactly where to improve next")

    print(f"total qa pairs evaluated : {engram['total']}")
    print("================================")


if __name__ == "__main__":
    main()