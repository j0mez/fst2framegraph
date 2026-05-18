from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from fst2framegraph.graph.builder import FrameGraphBuilder


@dataclass(frozen=True)
class _AgentFrameEvent:
    agent_text: str
    frame_type: str


class AnalysisBase:
    """Read-only analysis helpers over a FrameGraphBuilder MultiDiGraph."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        agentive_roles: list[str] | None = None,
        normalize_filler: Callable[[object], str] | None = None,
    ) -> None:
        self.graph = graph
        self.agentive_roles = tuple(agentive_roles or ["Agent", "Cause", "Cognizer"])
        self.normalize_filler = normalize_filler or FrameGraphBuilder.default_normalize

    def _filler_id_from_text(self, filler_text: str) -> str:
        normalized = self.normalize_filler(filler_text)
        if not normalized:
            return ""
        return f"filler:{FrameGraphBuilder.filler_hash(normalized)}"

    def get_filler_neighbors(self, filler_id: str, role: str | None = None) -> list[tuple[str, str]]:
        if filler_id not in self.graph:
            return []
        neighbors: list[tuple[str, str]] = []
        for frame_id, _, key, data in self.graph.in_edges(filler_id, keys=True, data=True):
            if self.graph.nodes.get(frame_id, {}).get("node_type") != "FrameInstance":
                continue
            edge_role = str(data.get("role") or key or "")
            if role is not None and edge_role != role:
                continue
            neighbors.append((frame_id, edge_role))
        return neighbors

    def get_frame_instances_by_type(self, frame_type: str) -> list[str]:
        return [
            node_id
            for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") == "FrameInstance" and data.get("frame_type") == frame_type
        ]

    def frames_for_filler(self, filler_text: str, role: str | None = None) -> list[tuple[str, str]]:
        filler_id = self._filler_id_from_text(filler_text)
        return self.get_filler_neighbors(filler_id, role=role)

    def trace_paths(
        self,
        start_filler_text: str,
        max_depth: int = 3,
        role_filters: list[str] | None = None,
    ) -> list[list[str]]:
        start_id = self._filler_id_from_text(start_filler_text)
        if not start_id or start_id not in self.graph or max_depth <= 0:
            return []

        allowed_roles = set(role_filters or [])
        use_filter = bool(allowed_roles)

        paths: list[list[str]] = []
        queue: deque[tuple[str, list[str], int]] = deque([(start_id, [start_id], 0)])

        while queue:
            node_id, path, depth = queue.popleft()
            if depth >= max_depth:
                if len(path) > 1:
                    paths.append(path)
                continue

            node_type = self.graph.nodes[node_id].get("node_type")
            if node_type == "Filler":
                for frame_id, edge_role in self.get_filler_neighbors(node_id):
                    if use_filter and edge_role not in allowed_roles:
                        continue
                    if frame_id in path:
                        continue
                    queue.append((frame_id, path + [frame_id], depth + 1))
                continue

            if node_type == "FrameInstance":
                expanded = False
                for _, next_filler, key, data in self.graph.out_edges(node_id, keys=True, data=True):
                    if self.graph.nodes.get(next_filler, {}).get("node_type") != "Filler":
                        continue
                    edge_role = str(data.get("role") or key or "")
                    if use_filter and edge_role not in allowed_roles:
                        continue
                    if next_filler in path:
                        continue
                    queue.append((next_filler, path + [next_filler], depth + 1))
                    expanded = True
                if not expanded and len(path) > 1:
                    paths.append(path)

        deduped: list[list[str]] = []
        seen = set()
        for path in paths:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    def _iter_agent_frame_events(self) -> list[_AgentFrameEvent]:
        events: list[_AgentFrameEvent] = []
        allowed_roles = set(self.agentive_roles)
        for frame_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "FrameInstance":
                continue
            frame_type = str(data.get("frame_type") or "")
            if not frame_type:
                continue
            for _, filler_id, key, edge_data in self.graph.out_edges(frame_id, keys=True, data=True):
                if self.graph.nodes.get(filler_id, {}).get("node_type") != "Filler":
                    continue
                role = str(edge_data.get("role") or key or "")
                if role not in allowed_roles:
                    continue
                agent_text = str(self.graph.nodes[filler_id].get("text") or "")
                if not agent_text:
                    continue
                events.append(_AgentFrameEvent(agent_text=agent_text, frame_type=frame_type))
        return events

    def agent_frame_lift(
        self,
        top_n_frames: int = 20,
        top_n_agents: int = 30,
        min_count: int = 2,
    ) -> pd.DataFrame:
        columns = ["agent", "frame_type", "count", "lift"]
        events = self._iter_agent_frame_events()
        if not events:
            return pd.DataFrame(columns=columns)

        frame_totals_all = Counter(event.frame_type for event in events)
        top_frames = {frame for frame, _ in frame_totals_all.most_common(top_n_frames)}
        if not top_frames:
            return pd.DataFrame(columns=columns)

        filtered = [event for event in events if event.frame_type in top_frames]
        if not filtered:
            return pd.DataFrame(columns=columns)

        agent_totals = Counter(event.agent_text for event in filtered)
        top_agents = {agent for agent, _ in agent_totals.most_common(top_n_agents)}
        if not top_agents:
            return pd.DataFrame(columns=columns)

        filtered = [event for event in filtered if event.agent_text in top_agents]
        if not filtered:
            return pd.DataFrame(columns=columns)

        pair_counts = Counter((event.agent_text, event.frame_type) for event in filtered)
        frame_totals = Counter(event.frame_type for event in filtered)
        agent_totals = Counter(event.agent_text for event in filtered)
        overall = len(filtered)
        effective_min_count = max(1, min(int(min_count or 1), max(pair_counts.values())))

        rows: list[dict[str, Any]] = []
        for (agent, frame_type), observed in pair_counts.items():
            if observed < effective_min_count:
                continue
            expected = (agent_totals[agent] * frame_totals[frame_type]) / overall if overall else 0.0
            if expected <= 0:
                continue
            rows.append(
                {
                    "agent": agent,
                    "frame_type": frame_type,
                    "count": int(observed),
                    "lift": float(observed / expected),
                }
            )
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns).sort_values(
            by=["lift", "count", "agent", "frame_type"],
            ascending=[False, False, True, True],
            ignore_index=True,
        )

    def agent_frame_communities(self, n_communities: int = 5) -> dict[str, Any]:
        events = self._iter_agent_frame_events()
        if not events:
            return {"assignments": {}, "top_terms": {}}

        weights = Counter((event.agent_text, event.frame_type) for event in events)
        bipartite = nx.Graph()
        for (agent_text, frame_type), weight in weights.items():
            agent_node = f"agent:{agent_text}"
            frame_node = f"frame_type:{frame_type}"
            bipartite.add_node(agent_node, kind="agent", label=agent_text)
            bipartite.add_node(frame_node, kind="frame_type", label=frame_type)
            bipartite.add_edge(agent_node, frame_node, weight=float(weight))

        if bipartite.number_of_nodes() == 0:
            return {"assignments": {}, "top_terms": {}}

        communities = list(nx.algorithms.community.greedy_modularity_communities(bipartite, weight="weight"))
        communities = communities[: max(0, n_communities)]

        assignments: dict[str, int] = {}
        top_terms: dict[int, dict[str, list[str]]] = {}

        for index, community_nodes in enumerate(communities):
            frame_scores: defaultdict[str, float] = defaultdict(float)
            agent_scores: defaultdict[str, float] = defaultdict(float)
            for node in community_nodes:
                assignments[node] = index
                node_data = bipartite.nodes[node]
                label = str(node_data.get("label") or "")
                score = float(bipartite.degree(node, weight="weight"))
                if node_data.get("kind") == "agent":
                    agent_scores[label] += score
                elif node_data.get("kind") == "frame_type":
                    frame_scores[label] += score
            top_terms[index] = {
                "agents": [term for term, _ in sorted(agent_scores.items(), key=lambda item: -item[1])[:5]],
                "frames": [term for term, _ in sorted(frame_scores.items(), key=lambda item: -item[1])[:5]],
            }

        return {"assignments": assignments, "top_terms": top_terms}
