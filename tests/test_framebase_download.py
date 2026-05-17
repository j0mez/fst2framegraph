from __future__ import annotations

from pathlib import Path

from fst2framegraph.framebase.download import (
    DownloadError,
    FrameBaseDownload,
    _download_with_fallbacks,
)


def test_download_with_fallbacks_uses_secondary_url(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_download(url: str, path: Path, overwrite: bool = False, timeout: int = 60) -> None:
        calls.append(url)
        if "primary" in url:
            raise DownloadError("primary failed")
        path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("fst2framegraph.framebase.download._download_url", fake_download)

    item = FrameBaseDownload(
        key="x",
        file_name="x.dat",
        url="https://primary.example/x.dat",
        fallback_urls=("https://fallback.example/x.dat",),
    )
    out = tmp_path / "x.dat"
    selected = _download_with_fallbacks(item, out)

    assert selected == "https://fallback.example/x.dat"
    assert calls == ["https://primary.example/x.dat", "https://fallback.example/x.dat"]
    assert out.read_text(encoding="utf-8") == "ok"


def test_download_with_fallbacks_raises_if_all_fail(monkeypatch, tmp_path: Path) -> None:
    def fake_download(url: str, path: Path, overwrite: bool = False, timeout: int = 60) -> None:
        raise DownloadError(f"failed {url}")

    monkeypatch.setattr("fst2framegraph.framebase.download._download_url", fake_download)

    item = FrameBaseDownload(
        key="x",
        file_name="x.dat",
        url="https://primary.example/x.dat",
        fallback_urls=("https://fallback.example/x.dat",),
    )

    try:
        _download_with_fallbacks(item, tmp_path / "x.dat")
    except DownloadError as exc:
        text = str(exc)
        assert "Could not download x.dat from any configured source." in text
        assert "https://primary.example/x.dat" in text
        assert "https://fallback.example/x.dat" in text
    else:
        raise AssertionError("Expected DownloadError when all download sources fail.")
