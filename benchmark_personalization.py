"""
Engram Personalization Benchmark
Run: python benchmark_personalization.py
"""

import random
import string
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import ollama

from engram.core.ingestion import IngestionPipeline
from engram.core.memory import MemoryNode, MemoryStatus
from engram.core.retrieval import RetrievalEngine
from engram.graph.auto_edge import AutoEdgeCreator
from engram.graph.causal import CausalChainBuilder
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayScheduler
from engram.scoring.engine import ScoringConfig, ScoringWeights
from engram.scoring.significance import SignificanceScorer
from engram.utils.store import InMemoryStore

random.seed(42)
np.random.seed(42)

DB_PATH = "benchmark_personalization.db"


@dataclass
class PersonalizationTest:
    question:          str
    expected_keywords: list[str]
    dimension:         str
    description:       str


@dataclass
class BenchmarkResult:
    name:         str
    total:        int  = 0
    correct:      int  = 0
    by_dimension: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def record(self, dimension: str, correct: bool) -> None:
        self.total += 1
        if correct:
            self.correct += 1
        if dimension not in self.by_dimension:
            self.by_dimension[dimension] = {"total": 0, "correct": 0}
        self.by_dimension[dimension]["total"] += 1
        if correct:
            self.by_dimension[dimension]["correct"] += 1

    def dimension_accuracy(self, dimension: str) -> float:
        d     = self.by_dimension.get(dimension, {})
        total = d.get("total", 0)
        return d.get("correct", 0) / total if total > 0 else 0.0


PERSONAL_CONTEXT = [
    "My name is Alex and I am a software engineer.",
    "I strongly prefer Python over every other programming language.",
    "I am building an AI memory system called Engram.",
    "I hate verbose code and unnecessary complexity.",
    "I was frustrated because AI kept forgetting everything about me.",
    "So I decided to build Engram to solve the memory problem.",
    "I work best in the mornings and hate afternoon meetings.",
    "My goal is to change how AI remembers things forever.",
    "I believe connectivity and irreplaceability are the key dimensions.",
    "I live in San Francisco and work remotely.",
]

GENERIC_NOISE = [
    "What is the capital of France?",
    "How does gradient descent work?",
    "What is the difference between Python 2 and Python 3?",
    "Explain the transformer architecture.",
    "What is backpropagation?",
    "Who invented the internet?",
    "What is the speed of light?",
    "How does HTTPS work?",
]

TESTS = [
    PersonalizationTest(
        question          = "What programming language should I use for my project?",
        expected_keywords = ["python"],
        dimension         = "personalization",
        description       = "Remembers language preference",
    ),
    PersonalizationTest(
        question          = "What is my name?",
        expected_keywords = ["alex"],
        dimension         = "personalization",
        description       = "Remembers personal identity",
    ),
    PersonalizationTest(
        question          = "What am I building?",
        expected_keywords = ["engram", "memory"],
        dimension         = "personalization",
        description       = "Remembers current project",
    ),
    PersonalizationTest(
        question          = "What is my goal?",
        expected_keywords = ["memory", "ai", "change"],
        dimension         = "personalization",
        description       = "Remembers stated goal",
    ),
    PersonalizationTest(
        question          = "When should we schedule our meeting?",
        expected_keywords = ["morning"],
        dimension         = "personalization",
        description       = "Remembers work preferences",
    ),
    PersonalizationTest(
        question          = "What coding style do I prefer?",
        expected_keywords = ["concise", "simple", "verbose"],
        dimension         = "personalization",
        description       = "Remembers coding preferences",
    ),
    PersonalizationTest(
        question          = "Where do I live?",
        expected_keywords = ["san francisco", "francisco"],
        dimension         = "personalization",
        description       = "Remembers location",
    ),
    PersonalizationTest(
        question          = "Why did I start building Engram?",
        expected_keywords = ["frustrated", "forgetting", "memory"],
        dimension         = "causal",
        description       = "Understands causal motivation",
    ),
    PersonalizationTest(
        question          = "What problem does Engram solve?",
        expected_keywords = ["memory", "forget", "ai"],
        dimension         = "causal",
        description       = "Understands causal chain",
    ),
    PersonalizationTest(
        question          = "What is the capital of France?",
        expected_keywords = ["paris"],
        dimension         = "noise",
        description       = "Generic question — should answer from knowledge",
    ),
]


def normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def answer_matches(predicted: str, keywords: list[str]) -> bool:
    pred = normalise(predicted)
    return any(kw.lower() in pred for kw in keywords)


def ask_llm(question: str, memory_context: str) -> str:
    try:
        response = ollama.chat(
            model   = "llama3",
            options = {"temperature": 0},
            messages = [
                {
                    "role":    "system",
                    "content": (
                        "You are a helpful personal assistant. "
                        "Use the provided memories to give personalized answers. "
                        "Be concise and direct. "
                        "If memories are relevant use them. "
                        "If not answer from general knowledge."
                    ),
                },
                {
                    "role":    "user",
                    "content": (
                        f"Memories about this user:\n{memory_context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )
        return response["message"]["content"].strip()
    except Exception:
        return ""


def build_engram_memory(
    personal_context: list[str],
    generic_noise:    list[str],
    days_ago:         int = 7,
) -> tuple[list[MemoryNode], GraphManager, InMemoryStore]:
    """Build Engram memory with significance scoring wired in.

    Significance is computed for every memory and stored in metadata.
    DecayScheduler reads significance from metadata during decay.
    This is the full brain-inspired pipeline — all 5 dimensions active.
    """
    pipeline   = IngestionPipeline(use_llm=False)
    sig_scorer = SignificanceScorer(db_path=DB_PATH)
    store      = InMemoryStore()
    graph      = GraphManager()
    creator    = AutoEdgeCreator(graph, db_path=DB_PATH)
    builder    = CausalChainBuilder(graph, db_path=DB_PATH)
    nodes      = []

    all_content = personal_context + generic_noise

    for text in all_content:
        result = pipeline.ingest(text)
        for node in result.nodes:
            # Score significance and store in metadata
            node.metadata["significance"] = sig_scorer.score(node.content)
            # Simulate aging
            node.last_accessed_at -= timedelta(days=days_ago)
            node.created_at       -= timedelta(days=days_ago)
            store.save(node)
            graph.add_node(node)
            creator.connect(node, nodes)
            builder.process(node, nodes)
            nodes.append(node)

    config = ScoringConfig(
        weights = ScoringWeights(
            irreplaceability = 0.35,
            connectivity     = 0.25,
            significance     = 0.20,
            recency          = 0.12,
            frequency        = 0.08,
        ),
        recency_half_life_days = 14.0,
        prune_threshold        = 0.20,
        decay_threshold        = 0.40,
    )

    scheduler = DecayScheduler(store, graph, config)
    scheduler.run()

    from engram.core.consolidation import MemoryConsolidator
    active    = [n for n in store.get_all() if n.status == MemoryStatus.ACTIVE]
    consolidator = MemoryConsolidator(db_path=DB_PATH)
    surviving, consolidation = consolidator.consolidate(active)
    surviving = [n for n in surviving if n.status == MemoryStatus.ACTIVE]

    return surviving, graph, store


def run_no_memory(tests: list[PersonalizationTest]) -> BenchmarkResult:
    result = BenchmarkResult(name="no memory")
    for test in tests:
        answer  = ask_llm(test.question, "No memories available.")
        correct = answer_matches(answer, test.expected_keywords)
        result.record(test.dimension, correct)
    return result


def run_flat_memory(
    tests:            list[PersonalizationTest],
    personal_context: list[str],
    generic_noise:    list[str],
) -> BenchmarkResult:
    pipeline  = IngestionPipeline(use_llm=False)
    engine    = RetrievalEngine(top_k=10, db_path=DB_PATH)
    all_nodes = []
    result    = BenchmarkResult(name="flat memory")

    for text in personal_context + generic_noise:
        ingested = pipeline.ingest(text)
        all_nodes.extend(ingested.nodes)

    for test in tests:
        retrieved = engine.query(test.question, all_nodes)
        context   = "\n".join(r.node.content for r in retrieved) if retrieved else "No memories."
        answer    = ask_llm(test.question, context)
        correct   = answer_matches(answer, test.expected_keywords)
        result.record(test.dimension, correct)

    return result


def run_engram(
    tests:            list[PersonalizationTest],
    personal_context: list[str],
    generic_noise:    list[str],
    days_ago:         int = 7,
) -> BenchmarkResult:
    nodes, graph, _ = build_engram_memory(personal_context, generic_noise, days_ago)
    engine          = RetrievalEngine(top_k=10, db_path=DB_PATH)
    result          = BenchmarkResult(name="engram")

    total = len(personal_context) + len(generic_noise)
    print(f"  engram kept {len(nodes)} of ~{total * 2} memories after decay")

    for test in tests:
        retrieved = engine.query(test.question, nodes, graph)
        context   = "\n".join(r.node.content for r in retrieved) if retrieved else "No memories."
        answer    = ask_llm(test.question, context)
        correct   = answer_matches(answer, test.expected_keywords)
        result.record(test.dimension, correct)

    return result


def print_results(
    no_memory:   BenchmarkResult,
    flat_memory: BenchmarkResult,
    engram:      BenchmarkResult,
) -> None:
    dimensions = ["personalization", "causal", "noise"]

    print()
    print("=== Engram Personalization Benchmark ===")
    print(f"{'metric':<30} {'no memory':>10} {'flat memory':>12} {'engram':>8}")
    print("-" * 62)
    print(f"{'overall accuracy':<30} "
          f"{no_memory.accuracy:>10.2f} "
          f"{flat_memory.accuracy:>12.2f} "
          f"{engram.accuracy:>8.2f}")

    for dim in dimensions:
        nm  = no_memory.dimension_accuracy(dim)
        fm  = flat_memory.dimension_accuracy(dim)
        eng = engram.dimension_accuracy(dim)
        print(f"{dim:<30} {nm:>10.2f} {fm:>12.2f} {eng:>8.2f}")

    print()
    if engram.accuracy > flat_memory.accuracy:
        improvement = (
            (engram.accuracy - flat_memory.accuracy)
            / flat_memory.accuracy * 100
        )
        print(f"Engram beats flat memory by : +{improvement:.1f}%")
    else:
        gap = (
            (flat_memory.accuracy - engram.accuracy)
            / max(flat_memory.accuracy, 0.001) * 100
        )
        print(f"Engram trails flat memory by : -{gap:.1f}%")

    print()
    print(f"total questions evaluated : {engram.total}")
    print("=========================================")


def main() -> None:
    print()
    print("=== Engram Personalization Benchmark ===")
    print(f"personal context  : {len(PERSONAL_CONTEXT)} statements")
    print(f"generic noise     : {len(GENERIC_NOISE)} statements")
    print(f"test questions    : {len(TESTS)}")
    print()

    print("running no memory baseline...")
    no_memory = run_no_memory(TESTS)
    print(f"accuracy : {no_memory.accuracy:.2f}")
    print()

    print("running flat memory baseline...")
    flat = run_flat_memory(TESTS, PERSONAL_CONTEXT, GENERIC_NOISE)
    print(f"accuracy : {flat.accuracy:.2f}")
    print()

    print("running Engram (memories aged 7 days)...")
    engram = run_engram(TESTS, PERSONAL_CONTEXT, GENERIC_NOISE, days_ago=7)
    print(f"accuracy : {engram.accuracy:.2f}")
    print()

    print_results(no_memory, flat, engram)


if __name__ == "__main__":
    main()