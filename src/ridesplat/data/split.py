"""Train/held-out split: frames named eval_* are held out, every other frame is used for training.

RIDE-Splat scenes name the reflective training frames train_* and the frames from the separate
reflection-free drive eval_*. Stock 3D Gaussian Splatting holds out every 8th image instead, which
would put evaluation frames in the training set, so install_prefix_split() replaces that split in
the 3DGS COLMAP loader.
"""
import os
import sys

EVAL_PREFIX = "eval_"


def frame_stem(image_name):
    """'eval_c00000.png' -> 'eval_c00000'. 3DGS keeps the file extension in a camera's image_name."""
    root, ext = os.path.splitext(image_name)
    return root if ext.lower() in (".png", ".jpg", ".jpeg") else image_name


def split_by_prefix(cams, prefix=EVAL_PREFIX):
    """Split camera infos into (train, test): names starting with `prefix` are held out.

    Repeated image names are kept once, and both lists are sorted by name.
    """
    seen, unique = set(), []
    for cam in cams:
        if cam.image_name in seen:
            continue
        seen.add(cam.image_name)
        unique.append(cam)
    unique.sort(key=lambda cam: cam.image_name)
    train = [cam for cam in unique if not cam.image_name.startswith(prefix)]
    test = [cam for cam in unique if cam.image_name.startswith(prefix)]
    return train, test


def install_prefix_split(prefix=EVAL_PREFIX):
    """With --eval, make the 3DGS COLMAP loader hold out exactly the images named `prefix`*.

    Call this once the 3DGS code is importable and before constructing a Scene.
    """
    import scene.dataset_readers as readers

    original = readers.readColmapSceneInfo
    if getattr(original, "_ridesplat_split", False):
        return

    def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
        info = original(path, images, depths, eval, train_test_exp, llffhold=llffhold)
        if not eval:
            return info
        train, test = split_by_prefix(list(info.train_cameras) + list(info.test_cameras), prefix)
        if not test:
            sys.exit(f"--eval was given but no image in {path} is named '{prefix}*'. RIDE-Splat holds "
                     f"out the {prefix}* frames; rename the held-out frames or drop --eval.")
        print(f"[split] {len(train)} training views, {len(test)} held-out views ({prefix}*)")
        return info._replace(train_cameras=train, test_cameras=test)

    readColmapSceneInfo._ridesplat_split = True
    readers.readColmapSceneInfo = readColmapSceneInfo
    readers.sceneLoadTypeCallbacks["Colmap"] = readColmapSceneInfo
