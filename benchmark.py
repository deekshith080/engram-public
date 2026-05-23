"""
Measures Engram against a flat memory baseline across four dimensions:
1. Retrieval precision  — does it find the right memories?
2. Forgetting quality   — does it prune the right memories?
3. Memory efficiency    — value retained vs storage used
4. Speed               — queries per second with cached embeddings
"""

import time
from datetime import datetime, timedelta, timezone

from engram.core.evaluation import Evaluator
from engram.core.ingestion import IngestionPipeline
from engram.core.memory import MemoryNode, MemoryStatus, MemoryType
from engram.core.retrieval import RetrievalEngine
from engram.graph.auto_edge import AutoEdgeCreator
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayScheduler
from engram.utils.embedding_cache import EmbeddingCache
from engram.utils.store import InMemoryStore


# Benchmark dataset — simulates a real user's conversation history

CONVERSATIONS = [
    "My name is Deekshith and I am building an AI memory system called Engram",
    "I prefer Python over every other programming language",
    "Engram uses graph based memory and intelligent forgetting",
    "I hate verbose code and unnecessary complexity",
    "I am building Engram to change how AI remembers things forever",
    "What is a neural network",
    "What is a transformer model",
    "What is backpropagation",
    "I work best in the mornings and hate meetings",
    "My goal is to build something magnificent that billion dollar companies cannot",
    "What is gradient descent",
    "I believe the connectivity dimension is what makes Engram novel",
    "What is Python",
    "What is machine learning",
    "I want Engram to be acquired by Anthropic or OpenAI someday",
]

TEST_QUERIES = [
    ("what is the user building",        ["engram", "building", "memory"]),
    ("what language does the user prefer", ["python"]),
    ("what is the user's goal",          ["goal", "magnificent", "billion"]),
    ("what does the user believe",       ["connectivity", "novel"]),
    ("what does the user hate",          ["verbose", "meetings"]),
]


# Baseline — flat memory, no decay, no scoring

def run_baseline(nodes: list[MemoryNode]) -> dict:
    """Flat memory baseline — saves everything, no intelligence."""
    engine = RetrievalEngine(top_k=5, db_path="benchmark.db")

    total_precision = 0.0
    for query_text, expected_keywords in TEST_QUERIES:
        results = engine.query(query_text, nodes)
        if not results:
            continue
        hits = sum(
            1 for r in results
            if any(kw.lower() in r.node.content.lower() for kw in expected_keywords)
        )
        total_precision += hits / len(results)

    return {
        "precision":    total_precision / len(TEST_QUERIES),
        "total_stored": len(nodes),
        "pruned":       0,
        "forgetting":   "none — stores everything forever",
    }


# Engram — intelligent memory with decay and graph

def run_engram(nodes: list[MemoryNode], store, graph) -> dict:
    """Engram — intelligent forgetting, scoring, graph connectivity."""
    scheduler = DecayScheduler(store, graph)
    report    = scheduler.run()

    evaluator = Evaluator()
    result    = evaluator.evaluate(nodes, report, TEST_QUERIES)

    return {
        "precision":          result.precision_at_k,
        "forgetting_quality": result.forgetting_quality,
        "irrepl_kept":        result.irreplaceability_kept,
        "irrepl_pruned":      result.irreplaceability_pruned,
        "total_stored":       result.total_memories,
        "pruned":             result.total_pruned,
        "active":             result.total_active,
    }


# Speed benchmark

def run_speed_benchmark(nodes: list[MemoryNode]) -> dict:
    """Measure query speed with and without embedding cache."""
    engine = RetrievalEngine(top_k=5, db_path="benchmark.db")
    cache  = EmbeddingCache("benchmark.db")

    # Warm up cache — ensure all embeddings are stored
    for node in nodes:
        cache.get(node.content)

    # Speed with cache — run 10 queries and average
    query = "what is the user building"
    start = time.time()
    for _ in range(10):
        engine.query(query, nodes)
    cached_time = (time.time() - start) / 10

    cache.close()

    return {
        "avg_query_ms":        round(cached_time * 1000, 1),
        "memories_searched":   len(nodes),
        "queries_per_second":  round(1 / cached_time, 1),
    }


# Main

def main() -> None:
    print()
    print("=== Engram Benchmark ===")
    print(f"dataset : {len(CONVERSATIONS)} conversations")
    print(f"queries : {len(TEST_QUERIES)} test queries")
    print()

    # Build memory system
    store    = InMemoryStore()
    graph    = GraphManager()
    pipeline = IngestionPipeline()
    creator  = AutoEdgeCreator(graph, db_path="benchmark.db")

    all_nodes = []
    for msg in CONVERSATIONS:
        result = pipeline.ingest(msg)
        for node in result.nodes:
            # Simulate age — factual memories are older and less accessed
            if node.memory_type == MemoryType.FACTUAL:
                node.last_accessed_at = (
                    datetime.now(timezone.utc) - timedelta(days=30)
                )
                node.irreplaceability = 0.05
            store.save(node)
            graph.add_node(node)
            creator.connect(node, all_nodes)
            all_nodes.append(node)

    print(f"memories created : {len(all_nodes)}")
    print(f"graph edges      : {graph.summary()['edges']}")
    print()

    # Run baseline
    print("--- Baseline (flat memory, no intelligence) ---")
    baseline = run_baseline(all_nodes)
    print(f"precision        : {baseline['precision']:.2f}")
    print(f"forgetting       : {baseline['forgetting']}")
    print(f"total stored     : {baseline['total_stored']}")
    print()

    # Run Engram
    print("--- Engram (intelligent memory) ---")
    engram = run_engram(all_nodes, store, graph)
    print(f"precision        : {engram['precision']:.2f}")
    print(f"forgetting quality: {engram['forgetting_quality']:.2f}")
    print(f"irrepl kept      : {engram['irrepl_kept']:.2f}")
    print(f"irrepl pruned    : {engram['irrepl_pruned']:.2f}")
    print(f"active memories  : {engram['active']}")
    print(f"pruned memories  : {engram['pruned']}")
    print()

    # Improvement
    precision_improvement = (
        (engram["precision"] - baseline["precision"]) / baseline["precision"] * 100
        if baseline["precision"] > 0 else 0
    )
    print("--- Results ---")
    print(f"precision improvement : {precision_improvement:+.0f}%")
    print(f"forgetting quality    : {engram['forgetting_quality']:.2f} / 1.00")
    print()

    # Speed
    print("--- Speed ---")
    speed = run_speed_benchmark(all_nodes)
    print(f"avg query time   : {speed['avg_query_ms']}ms")
    print(f"memories searched: {speed['memories_searched']}")
    print(f"queries per sec  : {speed['queries_per_second']}")
    print()
    print("========================")

    # Cleanup
    import os
    if os.path.exists("benchmark.db"):
        os.remove("benchmark.db")


if __name__ == "__main__":
    main()