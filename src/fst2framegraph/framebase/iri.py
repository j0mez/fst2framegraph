from __future__ import annotations

from urllib.parse import quote


def encode_block(value: str) -> str:
    """Approximate FrameBase path-block encoding for practical IRI minting.

    The official FrameBase encoder handles rare dot/colon cases. For converter output,
    this keeps common FrameNet frame and FE names readable and stable.
    """
    value = value.strip().replace(" ", "+")
    value = value.replace(".", ":")
    return quote(value, safe="_:+-")


def frame_iri(frame_name: str) -> str:
    return f"http://framebase.org/frame/{encode_block(frame_name)}"


def fe_iri(frame_name: str, fe_name: str) -> str:
    # FrameBase 2 commonly uses has_ lower-case role names in FE properties.
    return f"http://framebase.org/fe/{encode_block(frame_name)}.has_{encode_block(fe_name.lower())}"


def dbp_label_from_iri(iri: str) -> str:
    tail = iri.rstrip("/").split("/")[-1]
    if "." in tail:
        tail = tail.split(".")[-1]
    return tail.replace("+", " ")
