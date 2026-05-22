"""Knowledge Graph for GraphRAG.

Entities are nodes; co-occurrence in the same chunk creates weighted edges.
Implemented without NetworkX (pure-dict adjacency) for zero extra dependencies.

Retrieval strategy:
  1. Extract entities from query via regex
  2. Expand entity set by following edges (multi-hop)
  3. Score chunks by total mention count of expanded entities
  4. Return top-n ranked chunks

This catches multi-hop questions like "How does [A] relate to [B]?" where A and
B never co-occur in a single chunk but share an intermediate entity C.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .extractor import Entity, extract_entities


@dataclass
class KGNode:
    """Graph node representing a named entity."""

    entity_text: str
    entity_type: str
    chunk_ids: list[str] = field(default_factory=list)
    mention_count: int = 0


@dataclass
class KGEdge:
    """Undirected edge: two entities co-occurred in the same chunk."""

    source: str  # entity key (lowercase)
    target: str
    co_occurrence_count: int = 1
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class KGStats:
    """Summary statistics for monitoring and debugging."""

    n_nodes: int
    n_edges: int
    n_chunks: int
    top_entities: list[dict]

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_chunks": self.n_chunks,
            "top_entities": self.top_entities,
        }


class KnowledgeGraph:
    """Entity co-occurrence graph built from indexed document chunks."""

    def __init__(self) -> None:
        self._nodes: dict[str, KGNode] = {}
        # Edge key: tuple(sorted([src_key, tgt_key])) — undirected
        self._edges: dict[tuple[str, str], KGEdge] = {}
        self._n_chunks: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_chunks(self, chunks: list[dict]) -> KGStats:
        """Build graph from a list of document chunks.

        Resets state before building so multiple calls are idempotent.
        """
        self._nodes.clear()
        self._edges.clear()
        self._n_chunks = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            text = chunk.get("text", "")
            entities = extract_entities(text, chunk_id)
            entity_keys = self._register_entities(entities, chunk_id)
            self._register_edges(entity_keys, chunk_id)
            self._n_chunks += 1

        return self._make_stats()

    def _register_entities(self, entities: list[Entity], chunk_id: str) -> list[str]:
        keys: list[str] = []
        for ent in entities:
            key = ent.text.lower()
            if key not in self._nodes:
                self._nodes[key] = KGNode(
                    entity_text=ent.text,
                    entity_type=ent.entity_type,
                    chunk_ids=[chunk_id],
                    mention_count=1,
                )
            else:
                node = self._nodes[key]
                if chunk_id not in node.chunk_ids:
                    node.chunk_ids.append(chunk_id)
                node.mention_count += 1
            keys.append(key)
        return keys

    def _register_edges(self, entity_keys: list[str], chunk_id: str) -> None:
        for a_idx, src_key in enumerate(entity_keys):
            for tgt_key in entity_keys[a_idx + 1 :]:
                if src_key == tgt_key:
                    continue
                edge_key: tuple[str, str] = tuple(sorted([src_key, tgt_key]))  # type: ignore[assignment]
                if edge_key not in self._edges:
                    self._edges[edge_key] = KGEdge(
                        source=edge_key[0],
                        target=edge_key[1],
                        co_occurrence_count=1,
                        chunk_ids=[chunk_id],
                    )
                else:
                    edge = self._edges[edge_key]
                    edge.co_occurrence_count += 1
                    if chunk_id not in edge.chunk_ids:
                        edge.chunk_ids.append(chunk_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_neighbors(self, entity_key: str, max_hops: int = 1) -> list[str]:
        """Return entity keys reachable within max_hops via co-occurrence edges."""
        visited = {entity_key}
        frontier = {entity_key}

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for src, tgt in self._edges:
                if src in frontier and tgt not in visited:
                    next_frontier.add(tgt)
                elif tgt in frontier and src not in visited:
                    next_frontier.add(src)
            visited |= next_frontier
            frontier = next_frontier

        visited.discard(entity_key)
        return list(visited)

    def query_graph(
        self,
        query: str,
        chunks: list[dict],
        n_results: int = 5,
        max_hops: int = 1,
    ) -> list[dict]:
        """Graph-based retrieval: query → entities → graph expansion → ranked chunks.

        Returns empty list if no entities found in query (caller should fall back
        to semantic or hybrid search to avoid empty responses).
        """
        query_entities = extract_entities(query, chunk_id="__query__")
        if not query_entities:
            return []

        query_keys = [e.text.lower() for e in query_entities]

        # Expand via co-occurrence neighbors
        expanded_keys: set[str] = set(query_keys)
        for key in list(query_keys):
            for neighbor in self.get_neighbors(key, max_hops=max_hops):
                expanded_keys.add(neighbor)

        # Gather candidate chunks and score by entity mention count
        chunk_scores: dict[str, int] = defaultdict(int)
        chunk_set: set[str] = set()
        for key in expanded_keys:
            node = self._nodes.get(key)
            if node:
                for cid in node.chunk_ids:
                    chunk_scores[cid] += node.mention_count
                    chunk_set.add(cid)

        if not chunk_set:
            return []

        # Build chunk_id → chunk lookup
        chunk_map = {f"chunk_{i}": c for i, c in enumerate(chunks)}

        sorted_ids = sorted(chunk_set, key=lambda x: chunk_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:n_results]:
            chunk = chunk_map.get(cid)
            if chunk:
                results.append(
                    {
                        "text": chunk.get("text", ""),
                        "metadata": chunk.get("metadata", {}),
                        "score": chunk_scores[cid],
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_entity_subgraph(self, entity_key: str) -> dict:
        """Return D3.js-compatible subgraph centred on an entity (1-hop)."""
        neighbors = self.get_neighbors(entity_key, max_hops=1)
        all_keys = [entity_key, *neighbors]

        nodes_out = []
        for key in all_keys:
            node = self._nodes.get(key)
            if node:
                nodes_out.append(
                    {
                        "id": key,
                        "text": node.entity_text,
                        "type": node.entity_type,
                        "mention_count": node.mention_count,
                        "n_chunks": len(node.chunk_ids),
                    }
                )

        edges_out = []
        for src, tgt in self._edges:
            if src in all_keys and tgt in all_keys:
                edge = self._edges[(src, tgt)]
                edges_out.append(
                    {
                        "source": src,
                        "target": tgt,
                        "co_occurrence": edge.co_occurrence_count,
                    }
                )

        return {
            "center": entity_key,
            "nodes": nodes_out,
            "edges": edges_out,
            "found": entity_key in self._nodes,
        }

    def _make_stats(self) -> KGStats:
        top_entities = sorted(
            [
                {
                    "text": n.entity_text,
                    "type": n.entity_type,
                    "mentions": n.mention_count,
                }
                for n in self._nodes.values()
            ],
            key=lambda x: x["mentions"],
            reverse=True,
        )[:10]
        return KGStats(
            n_nodes=len(self._nodes),
            n_edges=len(self._edges),
            n_chunks=self._n_chunks,
            top_entities=top_entities,
        )

    @property
    def is_built(self) -> bool:
        return self._n_chunks > 0

    def stats(self) -> KGStats:
        return self._make_stats()
