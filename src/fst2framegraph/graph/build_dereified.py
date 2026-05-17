from __future__ import annotations

from itertools import product

import pandas as pd

from fst2framegraph.framebase.rule_index import RuleIndex, normalise_match_text
from fst2framegraph.normalise.ids import short_hash
from fst2framegraph.schema import FrameBaseRule


def _rows_for_fe(group: pd.DataFrame, fe_name: str | None) -> list:
    if not fe_name:
        return []
    wanted = normalise_match_text(fe_name)
    rows = [
        row
        for row in group.itertuples(index=False)
        if normalise_match_text(getattr(row, "fe_name", "")) == wanted
    ]
    seen = set()
    unique = []
    for row in rows:
        key = (row.frame_instance_id, row.fe_name, row.filler_id, row.filler_text)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _target_matches(rule: FrameBaseRule, target_text: str) -> bool:
    target = normalise_match_text(target_text)
    if not target:
        return False
    for candidate in (rule.target_lemma_or_lu, rule.microframe_name):
        if candidate and normalise_match_text(candidate) == target:
            return True
    return False


def _candidate_rules(
    rules: list[FrameBaseRule],
    frame_name: str,
    frame_iri: str,
    subject_fe: str,
    object_fe: str,
    target_text: str,
) -> tuple[str, list[FrameBaseRule]]:
    pair_rules = [
        rule
        for rule in rules
        if normalise_match_text(rule.subject_fe_name) == normalise_match_text(subject_fe)
        and normalise_match_text(rule.object_fe_name) == normalise_match_text(object_fe)
    ]
    if not pair_rules:
        return "unmatched", []

    allow_exact = "." in (frame_name or "")
    exact_rules = [rule for rule in pair_rules if allow_exact and frame_iri and rule.frame_iri == frame_iri]
    if len(exact_rules) == 1:
        return "exact_frame_iri", exact_rules
    if len(exact_rules) > 1:
        return "ambiguous", exact_rules

    target_rules = [rule for rule in pair_rules if _target_matches(rule, target_text)]
    if len(target_rules) == 1:
        return "frame_target_fe_unique", target_rules
    if len(target_rules) > 1:
        return "ambiguous", target_rules

    if len(pair_rules) == 1:
        return "broad_frame_fe_unique", pair_rules
    return "ambiguous", pair_rules


def build_dereified_edges(
    frame_instances: pd.DataFrame,
    frame_elements: pd.DataFrame,
    rule_index: RuleIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if frame_instances.empty or frame_elements.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "dereification_rules_matched": 0,
            "dereification_rule_match_ambiguous": 0,
            "dereification_rule_match_unmatched": 0,
            "dereification_opportunities": 0,
        }

    instance_lookup = {
        row.frame_instance_id: row
        for row in frame_instances.itertuples(index=False)
    }
    direct_edges: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    matched = 0
    ambiguous = 0
    unmatched = 0
    opportunities = 0

    for frame_id, group in frame_elements.groupby("frame_instance_id"):
        instance = instance_lookup.get(frame_id)
        if instance is None:
            continue
        frame_name = str(instance.frame_name)
        rules = rule_index.get(frame_name)
        if not rules:
            continue

        fe_names = {str(value) for value in group["fe_name"].dropna().unique()}
        candidate_pairs = {
            (str(rule.subject_fe_name), str(rule.object_fe_name))
            for rule in rules
            if rule.subject_fe_name and rule.object_fe_name
        }
        for subject_fe, object_fe in candidate_pairs:
            if subject_fe not in fe_names or object_fe not in fe_names or subject_fe == object_fe:
                continue
            subject_rows = _rows_for_fe(group, subject_fe)
            object_rows = _rows_for_fe(group, object_fe)
            if not subject_rows or not object_rows:
                continue

            opportunities += 1
            tier, candidates = _candidate_rules(
                rules,
                frame_name=frame_name,
                frame_iri=str(getattr(instance, "framebase_frame_iri", "") or ""),
                subject_fe=subject_fe,
                object_fe=object_fe,
                target_text=str(getattr(instance, "target_text", "") or ""),
            )

            if tier == "ambiguous":
                ambiguous += 1
                diagnostics.append(
                    {
                        "sentence_id": getattr(instance, "sentence_id", ""),
                        "doc_id": getattr(instance, "document_id", ""),
                        "frame_instance_id": frame_id,
                        "frame_name": frame_name,
                        "target_text": getattr(instance, "target_text", ""),
                        "subject_fe": subject_fe,
                        "object_fe": object_fe,
                        "status": "ambiguous",
                        "candidate_rule_count": len(candidates),
                        "candidate_rule_ids": "|".join(rule.rule_id for rule in candidates),
                        "message": "Multiple dereification rules matched; no official edge emitted.",
                    }
                )
                continue

            if not candidates:
                unmatched += 1
                diagnostics.append(
                    {
                        "sentence_id": getattr(instance, "sentence_id", ""),
                        "doc_id": getattr(instance, "document_id", ""),
                        "frame_instance_id": frame_id,
                        "frame_name": frame_name,
                        "target_text": getattr(instance, "target_text", ""),
                        "subject_fe": subject_fe,
                        "object_fe": object_fe,
                        "status": "unmatched",
                        "candidate_rule_count": 0,
                        "candidate_rule_ids": "",
                        "message": "No official FrameBase dereification rule matched safely.",
                    }
                )
                continue

            matched += 1
            rule = candidates[0]
            for sub, obj in product(subject_rows, object_rows):
                if sub.filler_id == obj.filler_id and sub.fe_name == obj.fe_name:
                    continue
                direct_edges.append(
                    {
                        "edge_id": "dered_" + short_hash(frame_id, sub.filler_id, rule.dbp_predicate_iri, obj.filler_id),
                        "source": sub.filler_id,
                        "target": obj.filler_id,
                        "label": rule.dbp_predicate_name or rule.dbp_label or rule.dbp_predicate_iri,
                        "dbp_iri": rule.dbp_predicate_iri,
                        "layer": "dereified",
                        "sentence_id": getattr(instance, "sentence_id", ""),
                        "doc_id": getattr(instance, "document_id", ""),
                        "frame_instance_id": frame_id,
                        "edge_type": "official_framebase_reder_edge",
                        "subject_filler": sub.filler_text,
                        "predicate_iri": rule.dbp_predicate_iri,
                        "predicate_label": rule.dbp_predicate_name or rule.dbp_label or rule.dbp_predicate_iri,
                        "object_filler": obj.filler_text,
                        "frame_name": frame_name,
                        "target_text": getattr(instance, "target_text", ""),
                        "subject_fe": subject_fe,
                        "object_fe": object_fe,
                        "source_rule_id": rule.rule_id,
                        "source_format": rule.source_format or "spin",
                        "match_tier": tier,
                        "match_status": "matched",
                    }
                )

    direct_df = pd.DataFrame(direct_edges).drop_duplicates("edge_id") if direct_edges else pd.DataFrame()
    diagnostics_df = pd.DataFrame(diagnostics)
    stats = {
        "dereification_rules_matched": matched,
        "dereification_rule_match_ambiguous": ambiguous,
        "dereification_rule_match_unmatched": unmatched,
        "dereification_opportunities": opportunities,
    }
    return direct_df, diagnostics_df, stats
