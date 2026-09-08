from pathlib import Path

import pytest
from fireviewer_model_lab.training.hls_burn_scars_corpus import paired_scene_files


def test_pairs_hls_scene_and_mask_by_stem(tmp_path: Path) -> None:
    scene = tmp_path / "subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4_merged.tif"
    mask = tmp_path / "subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4.mask.tif"
    scene.touch()
    mask.touch()

    assert paired_scene_files(tmp_path) == [(scene, mask)]


def test_rejects_orphan_hls_mask(tmp_path: Path) -> None:
    (tmp_path / "subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4.mask.tif").touch()

    with pytest.raises(ValueError, match="Masks without matching scenes"):
        paired_scene_files(tmp_path)
