"""Convert FrameNet-style parser output into FrameBase-compatible research graphs."""

from .adapters import from_frame_elements_long_csv, from_fst_output, from_legacy_pickle
from .analysis import AnalysisBase
from .fst import FSTGraphWriter, build_graph_from_clean, encode_with_fst, materialise_run
from .graph.builder import FrameGraphBuilder
from .io.inspect_outputs import convert_fst_outputs, inspect_fst_outputs

__version__ = "0.4.0-alpha"


def setup_framebase(*args, **kwargs):
    from .framebase.index import setup_framebase as _setup_framebase

    return _setup_framebase(*args, **kwargs)

__all__ = [
    "FSTGraphWriter",
    "AnalysisBase",
    "FrameGraphBuilder",
    "build_graph_from_clean",
    "encode_with_fst",
    "convert_fst_outputs",
    "from_frame_elements_long_csv",
    "from_fst_output",
    "from_legacy_pickle",
    "inspect_fst_outputs",
    "materialise_run",
    "setup_framebase",
]
