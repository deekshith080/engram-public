from __future__ import annotations

from dataclasses import dataclass

import ollama

from engram.core.ingestion import IngestionPipeline
from engram.core.memory import MemoryNode
from engram.core.retrieval import RetrievalEngine
from engram.graph.auto_edge import AutoEdgeCreator
from engram.graph.manager import GraphManager
from engram.scheduler.decay import DecayScheduler
from engram.utils.sqlite_store import SQLiteStore


@dataclass
class ConversationTurn:
    """One exchange between user and the assistant."""
    user:      str
    assistant: str


class EngramOllama:
    """Llama3 with persistent intelligent memory powered by Engram.

    Every conversation is ingested into memory.
    Every response is informed by relevant past memories.
    Memories decay intelligently over time.
    Embeddings are cached — each memory embedded once, reused forever.
    Runs entirely on your local machine — no API key, no cost.

    Usage
    -----
        client   = EngramOllama()
        response = client.chat("I prefer Python and I am building Engram")
        print(response)
    """

    MODEL = "llama3"

    def __init__(
        self,
        db_path: str = "engram.db",
        top_k:   int = 5,
    ) -> None:
        self._store     = SQLiteStore(db_path)
        self._graph     = GraphManager()
        self._pipeline  = IngestionPipeline()
        self._creator   = AutoEdgeCreator(self._graph, db_path)
        self._retrieval = RetrievalEngine(top_k=top_k, db_path=db_path)
        self._history:  list[ConversationTurn] = []
        self._nodes:    list[MemoryNode]       = []

        self._bootstrap()

    def chat(self, user_message: str) -> str:
        """Send a message and get a memory-informed response.

        Steps
        -----
        1. Retrieve relevant memories using cached embeddings
        2. Build prompt with memory context
        3. Get Llama3 response
        4. Ingest user message and response into memory
        5. Return response

        Parameters
        ----------
        user_message: The user's message.

        Returns
        -------
        Assistant response as a string.
        """
        context       = self._retrieval.query_as_context(
            user_message, self._nodes
        )
        system_prompt = self._build_system_prompt(context)

        response = ollama.chat(
            model    = self.MODEL,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        assistant_message = response["message"]["content"]

        self._ingest(user_message)
        self._ingest(assistant_message)

        self._history.append(ConversationTurn(
            user      = user_message,
            assistant = assistant_message,
        ))

        return assistant_message

    def run_decay(self) -> None:
        """Run one decay cycle — prune weak memories."""
        scheduler = DecayScheduler(self._store, self._graph)
        scheduler.run()

    def memory_summary(self) -> str:
        """Return a human readable summary of current memory state."""
        active   = sum(1 for n in self._nodes if n.status.value == "active")
        decaying = sum(1 for n in self._nodes if n.status.value == "decaying")
        pruned   = sum(1 for n in self._nodes if n.status.value == "pruned")
        return (
            f"memories : {len(self._nodes)} total — "
            f"{active} active, {decaying} decaying, {pruned} pruned"
        )

    def close(self) -> None:
        """Close all database connections cleanly."""
        self._store.close()

    # Private helpers

    def _bootstrap(self) -> None:
        """Load existing memories from SQLite into graph on startup."""
        existing = self._store.get_all()
        for node in existing:
            self._graph.add_node(node)
            self._nodes.append(node)

    def _ingest(self, text: str) -> None:
        """Ingest a message into memory and connect edges."""
        result = self._pipeline.ingest(text)
        for node in result.nodes:
            self._store.save(node)
            self._graph.add_node(node)
            self._creator.connect(node, self._nodes)
            self._nodes.append(node)

    def _build_system_prompt(self, memory_context: str) -> str:
        return (
            "You are a helpful assistant with persistent memory.\n\n"
            "The following memories are relevant to this conversation:\n\n"
            f"{memory_context}\n\n"
            "Use these memories to give personalized, context-aware responses.\n"
            "If a memory is directly relevant, use it naturally in your response.\n"
            "Never mention that you are reading from a memory system."
        )