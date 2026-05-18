from __future__ import annotations

from scripts.install_colab_fst import FST_RUNTIME_REQUIREMENTS


def test_colab_installer_installs_fst_runtime_dependencies() -> None:
    joined = " ".join(FST_RUNTIME_REQUIREMENTS)

    assert "protobuf>=3.20.1,<4.0.0" in FST_RUNTIME_REQUIREMENTS
    assert "transformers" in joined
    assert "nlpaug" in joined
    assert "nltk" in joined
    assert "pytorch-lightning" in joined
    assert "tqdm" in joined
