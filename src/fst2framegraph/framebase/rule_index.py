from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import unquote_plus

from fst2framegraph.schema import FrameBaseRule


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_POS_SUFFIX = re.compile(r"\.(verb|noun|adjective|adverb|v|n|a)$")


def normalise_match_text(value: object) -> str:
    text = unquote_plus("" if value is None else str(value)).strip().lower()
    text = text.replace("_", " ")
    text = _POS_SUFFIX.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


@dataclass
class RuleIndex:
    by_frame: dict[str, list[FrameBaseRule]] = field(default_factory=dict)

    @classmethod
    def from_rules(cls, rules: list[FrameBaseRule]) -> "RuleIndex":
        grouped: dict[str, list[FrameBaseRule]] = defaultdict(list)
        for rule in rules:
            if rule.frame_name:
                grouped[normalise_match_text(rule.frame_name)].append(rule)
        return cls(by_frame=dict(grouped))

    def get(self, frame_name: str) -> list[FrameBaseRule]:
        return self.by_frame.get(normalise_match_text(frame_name), [])
