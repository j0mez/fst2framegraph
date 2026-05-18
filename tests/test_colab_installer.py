from __future__ import annotations

from scripts.install_colab_fst import (
    COLAB_BACKEND_ENV,
    FST_RUNTIME_REQUIREMENTS,
    apply_colab_backend_env,
    write_colab_backend_env_guard,
)


def test_colab_installer_installs_fst_runtime_dependencies() -> None:
    joined = " ".join(FST_RUNTIME_REQUIREMENTS)

    assert "protobuf>=3.20.1,<4.0.0" in FST_RUNTIME_REQUIREMENTS
    assert "transformers" in joined
    assert "nlpaug" in joined
    assert "nltk" in joined
    assert "pytorch-lightning" in joined
    assert "tqdm" in joined


def test_colab_installer_sets_tensorflow_disable_environment() -> None:
    env: dict[str, str] = {}

    apply_colab_backend_env(env)

    assert env == COLAB_BACKEND_ENV
    assert env["USE_TF"] == "0"
    assert env["TRANSFORMERS_NO_TF"] == "1"
    assert env["USE_FLAX"] == "0"


def test_colab_installer_writes_python_startup_env_guard(tmp_path) -> None:
    path = write_colab_backend_env_guard(tmp_path)

    content = path.read_text(encoding="utf-8")
    assert path.name == "fst2framegraph_colab_env.pth"
    for name, value in COLAB_BACKEND_ENV.items():
        assert f"os.environ.setdefault({name!r}, {value!r})" in content
