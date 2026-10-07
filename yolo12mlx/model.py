"""Build YOLO12 on top of the patched yolo-mlx and load PyTorch (Ultralytics) weights."""
import math
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from yolo26mlx.nn.tasks import DetectionModel

CFG = str(Path(__import__("yolo26mlx").__file__).parent / "cfg" / "models" / "12" / "yolo12.yaml")
HEAD = 21  # index of the Detect layer in yolo12.yaml


def build(nc: int = 80, scale: str = "s") -> DetectionModel:
    return DetectionModel(CFG, nc=nc, scale=scale, verbose=False)


def map_name(k: str) -> str:
    """Ultralytics state_dict key -> yolo-mlx parameter name."""
    k = re.sub(r"^model\.(\d+)\.", r"model.layers.\1.", k)
    k = re.sub(rf"layers\.{HEAD}\.cv2\.(\d+)\.(\d+)\.", rf"layers.{HEAD}.cv2.layer\1.layers.\2.", k)
    k = re.sub(
        rf"layers\.{HEAD}\.cv3\.(\d+)\.(\d+)\.(\d+)\.",
        lambda m: f"layers.{HEAD}.cv3.layer{m[1]}.layers.{int(m[2]) * 2 + int(m[3])}.",
        k,
    )
    k = re.sub(
        rf"layers\.{HEAD}\.cv3\.(\d+)\.(\d+)\.",
        lambda m: f"layers.{HEAD}.cv3.layer{m[1]}.layers.{int(m[2]) * 2}.",
        k,
    )
    k = re.sub(r"\.m\.(\d+)\.(\d+)\.(attn|mlp)", r".m.block\1.ab\2.\3", k)
    k = re.sub(r"(layers\.(?:11|14|17)\.m\.)(\d+)\.", r"\1block\2.", k)
    k = re.sub(r"\.mlp\.0\.", ".mlp.cv1.", k)
    k = re.sub(r"\.mlp\.1\.", ".mlp.cv2.", k)
    return k


def load_npz(model, path: str):
    """Strict-ish load of an npz made by `convert.py`. Returns (missing, extra)."""
    mapped = [(map_name(k), v) for k, v in mx.load(path).items()]
    names = {k for k, _ in tree_flatten(model.parameters())}
    ok = [(k, v) for k, v in mapped if k in names]
    missing = sorted(names - {k for k, _ in ok})
    extra = sorted({k for k, _ in mapped} - names)
    model.load_weights(ok, strict=False)
    return missing, extra


def load_npz_partial(model, path: str):
    """Load only tensors whose shape matches (e.g. COCO weights into an nc != 80 model).
    Returns the names that were skipped because of a shape mismatch."""
    cur = dict(tree_flatten(model.parameters()))
    ok, skipped = [], []
    for k, v in mx.load(path).items():
        k = map_name(k)
        if k in cur and tuple(cur[k].shape) == tuple(v.shape):
            ok.append((k, v))
        elif k in cur:
            skipped.append(k)
    model.load_weights(ok, strict=False)
    return skipped


def init_cls_bias(model, nc: int, strides=(8, 16, 32)) -> None:
    """Prior-probability init of the final classification convs. Required after a
    partial load: without it the first loss values are ~200x too large."""
    upd = [
        (f"model.layers.{HEAD}.cv3.layer{i}.layers.4.bias", mx.full((nc,), math.log(5 / nc / (640 / s) ** 2)))
        for i, s in enumerate(strides)
    ]
    model.load_weights(upd, strict=False)
