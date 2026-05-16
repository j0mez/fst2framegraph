import pandas as pd

from fst2framegraph.graph.build_nested import build_nested_edges


def test_nested_target_text_containment():
    frames = pd.DataFrame([
        {"frame_instance_id": "f1", "sentence_id": "s1", "document_id": "d1", "frame_name": "Assistance", "target_text": "help"},
        {"frame_instance_id": "f2", "sentence_id": "s1", "document_id": "d1", "frame_name": "Cause_change", "target_text": "reduce"},
    ])
    elements = pd.DataFrame([
        {"frame_instance_id": "f1", "sentence_id": "s1", "document_id": "d1", "frame_name": "Assistance", "fe_name": "Goal", "filler_text": "reduce emissions"},
    ])
    nested = build_nested_edges(frames, elements)
    assert len(nested) == 1
    assert nested.iloc[0]["target"] == "f2"


def test_nested_assistance_goal_contains_reduce_emissions_frame_by_span():
    frames = pd.DataFrame(
        [
            {
                "frame_instance_id": "s1::f0::Assistance::15",
                "sentence_id": "s1",
                "document_id": "doc1",
                "frame_name": "Assistance",
                "target_text": "help",
                "target_start": 15,
                "target_end": 19,
            },
            {
                "frame_instance_id": "s1::f1::Cause_change_of_position_on_a_scale::36",
                "sentence_id": "s1",
                "document_id": "doc1",
                "frame_name": "Cause_change_of_position_on_a_scale",
                "target_text": "reduce",
                "target_start": 36,
                "target_end": 42,
            },
        ]
    )
    elements = pd.DataFrame(
        [
            {
                "frame_instance_id": "s1::f0::Assistance::15",
                "sentence_id": "s1",
                "document_id": "doc1",
                "frame_name": "Assistance",
                "fe_name": "Goal",
                "filler_text": "consumers reduce emissions",
                "filler_start": 20,
                "filler_end": 52,
            }
        ]
    )

    nested = build_nested_edges(frames, elements)

    assert len(nested) == 1
    edge = nested.iloc[0]
    assert edge["source"] == "s1::f0::Assistance::15"
    assert edge["target"] == "s1::f1::Cause_change_of_position_on_a_scale::36"
    assert edge["parent_frame_name"] == "Assistance"
    assert edge["parent_fe_name"] == "Goal"
    assert edge["child_frame_name"] == "Cause_change_of_position_on_a_scale"
    assert edge["nesting_method"] == "span_containment"
