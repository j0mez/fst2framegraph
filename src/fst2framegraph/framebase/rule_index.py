from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from fst2framegraph.normalise.text import clean_text
from fst2framegraph.schema import FrameBaseRule


@dataclass
class RuleIndex:
    by_frame: dict[str, list[FrameBaseRule]] = field(default_factory=dict)

    @classmethod
    def from_rules(cls, rules: list[FrameBaseRule]) -> "RuleIndex":
        d: dict[str, list[FrameBaseRule]] = defaultdict(list)
        for rule in rules:
            if rule.frame_name:
                d[clean_text(rule.frame_name)].append(rule)
        return cls(by_frame=dict(d))

    def get(self, frame_name: str) -> list[FrameBaseRule]:
        return self.by_frame.get(clean_text(frame_name), [])
