from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from fst2framegraph.adapters import from_frame_elements_long_csv
from fst2framegraph.analysis import AnalysisBase
from fst2framegraph.graph.builder import FrameGraphBuilder


def build_event_graph(
    *,
    run_dir: str | Path,
    graph_out_dir: str | Path,
    include_sentence_nodes: bool = True,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    run_dir = Path(run_dir)
    graph_out_dir = Path(graph_out_dir)
    graph_out_dir.mkdir(parents=True, exist_ok=True)

    source_csv = run_dir / "frame_elements_long.csv"
    documents = from_frame_elements_long_csv(source_csv)
    builder = FrameGraphBuilder(include_sentence_nodes=include_sentence_nodes)
    graph = builder.build_graph(documents)

    builder.save_graph(graph, graph_out_dir / "graph.gpickle")
    builder.save_graph(graph, graph_out_dir / "graph.graphml")

    node_type_counts = Counter(
        str(data.get("node_type") or "unknown") for _, data in graph.nodes(data=True)
    )
    report = {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "node_type_counts": dict(node_type_counts),
        "graph_pickle": str(graph_out_dir / "graph.gpickle"),
        "graph_graphml": str(graph_out_dir / "graph.graphml"),
    }
    (graph_out_dir / "graph_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return graph, report


def run_analysis_outputs(
    *,
    graph: nx.MultiDiGraph,
    analysis_out_dir: str | Path,
    top_n_frames: int = 20,
    top_n_agents: int = 30,
    min_count: int = 2,
    n_communities: int = 5,
) -> dict[str, Any]:
    analysis_out_dir = Path(analysis_out_dir)
    analysis_out_dir.mkdir(parents=True, exist_ok=True)
    analysis = AnalysisBase(graph)

    frame_counter = Counter(
        str(data.get("frame_type") or "")
        for _, data in graph.nodes(data=True)
        if data.get("node_type") == "FrameInstance" and data.get("frame_type")
    )
    top_frames_df = pd.DataFrame(
        [{"frame_type": frame_type, "count": int(count)} for frame_type, count in frame_counter.most_common(top_n_frames)]
    )
    top_frames_df.to_csv(analysis_out_dir / "top_frame_types.csv", index=False)

    agent_counter = Counter()
    for frame_id, data in graph.nodes(data=True):
        if data.get("node_type") != "FrameInstance":
            continue
        for _, filler_id, key, edge_data in graph.out_edges(frame_id, keys=True, data=True):
            role = str(edge_data.get("role") or key or "")
            if role != "Agent":
                continue
            filler_text = str(graph.nodes[filler_id].get("text") or "")
            if filler_text:
                agent_counter[filler_text] += 1
    top_agents_df = pd.DataFrame(
        [{"agent": agent, "count": int(count)} for agent, count in agent_counter.most_common(top_n_agents)]
    )
    top_agents_df.to_csv(analysis_out_dir / "top_agent_fillers.csv", index=False)

    lift_df = analysis.agent_frame_lift(
        top_n_frames=top_n_frames,
        top_n_agents=top_n_agents,
        min_count=min_count,
    )
    lift_df.to_csv(analysis_out_dir / "agent_frame_lift.csv", index=False)

    community_payload = analysis.agent_frame_communities(n_communities=n_communities)
    (analysis_out_dir / "agent_frame_communities.json").write_text(
        json.dumps(community_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    path_seed = top_agents_df.iloc[0]["agent"] if not top_agents_df.empty else "we"
    traces = analysis.trace_paths(
        str(path_seed),
        max_depth=2,
        role_filters=["Agent", "Goal", "Theme", "Cause", "Entity", "Event"],
    )
    trace_rows: list[dict[str, Any]] = []
    for index, path in enumerate(traces[:50]):
        trace_rows.append({"path_index": index, "path": " -> ".join(path)})
    trace_df = pd.DataFrame(trace_rows)
    trace_df.to_csv(analysis_out_dir / "sample_path_traces.csv", index=False)

    summary = {
        "top_frame_types_rows": int(len(top_frames_df)),
        "top_agent_fillers_rows": int(len(top_agents_df)),
        "lift_rows": int(len(lift_df)),
        "trace_rows": int(len(trace_df)),
        "community_count": int(len(community_payload.get("top_terms", {}))),
        "path_seed": str(path_seed),
    }
    (analysis_out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
