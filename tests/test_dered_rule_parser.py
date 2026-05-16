from pathlib import Path

from fst2framegraph.framebase.parse_dered_rules import parse_dered_rules


def test_parse_simple_rule(tmp_path: Path):
    rule = """
    CONSTRUCT { ?s <http://framebase.org/dbp/Causation.causes> ?o }
    WHERE {
      ?f a <http://framebase.org/frame/Causation> .
      ?f <http://framebase.org/fe/Causation.has_cause> ?s .
      ?f <http://framebase.org/fe/Causation.has_effect> ?o .
    }
    """
    p = tmp_path / "rules.rq"
    p.write_text(rule)
    rules = parse_dered_rules(p)
    assert len(rules) == 1
    assert rules[0].frame_name == "Causation"
    assert rules[0].subject_fe_name.lower() == "cause"
    assert rules[0].object_fe_name.lower() == "effect"


def test_parse_prefixed_old_style_rule(tmp_path: Path):
    rule = """
    PREFIX : <http://framebase.org/>
    CONSTRUCT { ?s :dbp.Separating.isPartitionedIntoParts ?o }
    WHERE {
      ?f a :Microframe.Separating.verb.partition .
      ?f :fe.Separating.Whole ?s .
      ?f :fe.Separating.Parts ?o .
    }
    """
    p = tmp_path / "rules.rq"
    p.write_text(rule)
    rules = parse_dered_rules(p)
    assert len(rules) == 1
    assert rules[0].frame_name == "Separating"
    assert rules[0].subject_fe_name == "Whole"
    assert rules[0].object_fe_name == "Parts"
