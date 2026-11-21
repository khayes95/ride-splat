import os
import struct

import pytest
from PIL import Image

from ridesplat.data import prepare as prep


def write_images_bin(path, names):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names):
            f.write(struct.pack("<i", i + 1) + b"\0" * 56 + struct.pack("<i", 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 1) + b"\0" * 24)


def make_scene(root, train, held_out):
    scene = root / "rs1"
    for split, names in (("train", train), ("eval", held_out)):
        (scene / split).mkdir(parents=True)
        for n in names:
            Image.new("RGB", (4, 4)).save(scene / split / n)
    (scene / "sparse" / "0").mkdir(parents=True)
    write_images_bin(scene / "sparse" / "0" / "images.bin", train + held_out)
    return scene


def test_colmap_names_and_images_folder(tmp_path):
    scene = make_scene(tmp_path, ["train_r00000.png", "train_r00001.png"], ["eval_c00000.png"])
    assert prep.colmap_image_names(scene / "sparse" / "0" / "images.bin") == [
        "train_r00000.png", "train_r00001.png", "eval_c00000.png"]
    prep.prepare_scene(str(scene))
    assert sorted(os.listdir(scene / "images")) == ["eval_c00000.png", "train_r00000.png", "train_r00001.png"]
    assert Image.open(scene / "images" / "eval_c00000.png").size == (4, 4)


def test_missing_frame_is_an_error(tmp_path):
    scene = make_scene(tmp_path, ["train_r00000.png"], ["eval_c00000.png"])
    os.remove(scene / "eval" / "eval_c00000.png")
    with pytest.raises(SystemExit):
        prep.prepare_scene(str(scene))


def test_existing_links_are_kept_and_copy_mode_copies(tmp_path):
    scene = make_scene(tmp_path, ["train_r00000.png"], ["eval_c00000.png"])
    prep.prepare_scene(str(scene))
    prep.prepare_scene(str(scene))                                  # second run is a no-op
    assert os.path.islink(scene / "images" / "train_r00000.png")
    other = make_scene(tmp_path / "copy", ["train_r00000.png"], ["eval_c00000.png"])
    prep.prepare_scene(str(other), copy=True)
    assert not os.path.islink(other / "images" / "train_r00000.png")


def test_main_requires_a_scene(tmp_path):
    with pytest.raises(SystemExit):
        prep.main(["--root", str(tmp_path)])
