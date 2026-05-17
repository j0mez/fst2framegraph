from __future__ import annotations

import random

import pandas as pd

from fst2framegraph import AnalysisBase, FrameGraphBuilder


def analysis_documents() -> list[dict]:
    return [
        {
            "doc_id": "d1",
            "text": "We invest in clean energy. We report progress.",
            "metadata": {"source": "test"},
            "frames": [
                {
                    "frame_type": "Investing",
                    "trigger": "invest",
                    "sent_idx": 0,
                    "frame_elements": [
                        {"role": "Agent", "text": "We"},
                        {"role": "Goal", "text": "clean energy"},
                    ],
                },
                {
                    "frame_type": "Reporting",
                    "trigger": "report",
                    "sent_idx": 1,
                    "frame_elements": [
                        {"role": "Agent", "text": "we."},
                        {"role": "Topic", "text": "progress"},
                    ],
                },
            ],
        },
        {
            "doc_id": "d2",
            "text": "Companies invest in efficiency.",
            "metadata": {"source": "test"},
            "frames": [
                {
                    "frame_type": "Investing",
                    "trigger": "invest",
                    "sent_idx": 0,
                    "frame_elements": [
                        {"role": "Agent", "text": "Companies"},
                        {"role": "Goal", "text": "efficiency"},
                    ],
                }
            ],
        },
        {
            "doc_id": "d3",
            "text": "Storms cause outages.",
            "metadata": {"source": "test"},
            "frames": [
                {
                    "frame_type": "Causation",
                    "trigger": "cause",
                    "sent_idx": 0,
                    "frame_elements": [
                        {"role": "Cause", "text": "Storms"},
                        {"role": "Effect", "text": "outages"},
                    ],
                }
            ],
        },
    ]


def test_frames_for_filler_and_neighbors() -> None:
    graph = FrameGraphBuilder().build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    matches = analysis.frames_for_filler("we", role="Agent")
    frame_ids = {frame_id for frame_id, _ in matches}
    assert frame_ids == {"frame:d1:0", "frame:d1:1"}

    filler_id = "filler:" + FrameGraphBuilder.filler_hash("we")
    neighbors = analysis.get_filler_neighbors(filler_id)
    assert len(neighbors) == 2
    assert all(role == "Agent" for _, role in neighbors)


def test_get_frame_instances_by_type() -> None:
    graph = FrameGraphBuilder().build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    investing_frames = analysis.get_frame_instances_by_type("Investing")
    assert set(investing_frames) == {"frame:d1:0", "frame:d2:0"}


def test_trace_paths_finds_expected_chain() -> None:
    graph = FrameGraphBuilder().build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    paths = analysis.trace_paths("we", max_depth=2, role_filters=["Agent", "Goal"])
    clean_energy_id = "filler:" + FrameGraphBuilder.filler_hash("clean energy")
    assert any(path[0].startswith("filler:") and clean_energy_id in path for path in paths)


def test_agent_frame_lift_columns_and_signal() -> None:
    graph = FrameGraphBuilder().build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    result = analysis.agent_frame_lift(top_n_frames=10, top_n_agents=10, min_count=1)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["agent", "frame_type", "count", "lift"]
    assert not result.empty
    assert ((result["agent"] == "we") & (result["frame_type"] == "Investing")).any()


def test_agent_frame_communities_shape() -> None:
    graph = FrameGraphBuilder().build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    communities = analysis.agent_frame_communities(n_communities=5)
    assert "assignments" in communities
    assert "top_terms" in communities
    assert isinstance(communities["assignments"], dict)
    assert isinstance(communities["top_terms"], dict)


def test_missing_filler_returns_empty_structures() -> None:
    graph = FrameGraphBuilder().build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    assert analysis.frames_for_filler("not-in-graph", role="Agent") == []
    assert analysis.trace_paths("not-in-graph", max_depth=3) == []

    empty_graph = FrameGraphBuilder().build_graph(
        [{"doc_id": "d0", "text": "No frames.", "metadata": {}, "frames": []}]
    )
    empty_analysis = AnalysisBase(empty_graph)
    assert empty_analysis.agent_frame_lift().empty
    assert empty_analysis.agent_frame_communities() == {"assignments": {}, "top_terms": {}}


def test_analysis_works_with_sentence_nodes_disabled() -> None:
    builder = FrameGraphBuilder(include_sentence_nodes=False)
    graph = builder.build_graph(analysis_documents())
    analysis = AnalysisBase(graph)

    assert analysis.frames_for_filler("we", role="Agent")
    assert analysis.trace_paths("we", max_depth=2, role_filters=["Agent", "Goal"])


def test_methods_complete_quickly_on_medium_graph() -> None:
    rng = random.Random(7)
    docs: list[dict] = []
    agents = ["we", "companies", "regulators", "consumers", "industry"]
    goals = ["emissions", "efficiency", "innovation", "resilience", "jobs"]
    frame_types = ["Investing", "Reporting", "Planning", "Using", "Causation"]

    for idx in range(900):
        frames = []
        for frame_idx in range(5):
            agent = agents[rng.randrange(len(agents))]
            goal = goals[rng.randrange(len(goals))]
            frame_type = frame_types[frame_idx % len(frame_types)]
            frames.append(
                {
                    "frame_type": frame_type,
                    "trigger": frame_type.lower(),
                    "sent_idx": frame_idx,
                    "frame_elements": [
                        {"role": "Agent", "text": agent},
                        {"role": "Goal", "text": goal},
                    ],
                }
            )
        docs.append({"doc_id": f"bulk{idx}", "text": "", "metadata": {}, "frames": frames})

    graph = FrameGraphBuilder().build_graph(docs)
    analysis = AnalysisBase(graph)
    assert graph.number_of_nodes() >= 5000

    lift_df = analysis.agent_frame_lift(top_n_frames=5, top_n_agents=5, min_count=3)
    paths = analysis.trace_paths("we", max_depth=2, role_filters=["Agent", "Goal"])

    assert not lift_df.empty
    assert paths
