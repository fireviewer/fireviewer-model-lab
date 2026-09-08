from pathlib import Path

import pytest
from fireviewer_model_lab.training import romav2_offline


def test_verify_assets_fails_closed_on_weight_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roma = tmp_path / "roma"
    dino = tmp_path / "dino"
    (roma / "src/romav2").mkdir(parents=True)
    (roma / "src/romav2/romav2.py").write_text("", encoding="utf-8")
    dino.mkdir()
    (dino / "hubconf.py").write_text("", encoding="utf-8")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"wrong")

    revisions = iter((romav2_offline.ROMAV2_REVISION, romav2_offline.DINOV3_TORCHHUB_REVISION))
    monkeypatch.setattr(romav2_offline, "git_revision", lambda _path: next(revisions))
    with pytest.raises(RuntimeError, match="weights SHA-256 mismatch"):
        romav2_offline.verify_assets(
            romav2_source=roma,
            dinov3_source=dino,
            weights=weights,
        )
