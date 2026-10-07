"""Fine-tune YOLO12 from converted COCO weights (needs the patched yolo-mlx).

    python -m yolo12mlx.train --data data.yaml --weights yolo12s.npz --imgsz 896 --batch 6 --epochs 3

Env: YOLO_PRECISION=bf16|fp16|fp32 (default bf16), YOLO_COMPILE=1|0, YOLO_LOADER_WORKERS=6.
"""
import argparse

from yolo26mlx.engine.model import YOLO

from .model import build, init_cls_bias, load_npz_partial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--nc", type=int, required=True); ap.add_argument("--scale", default="s", choices=list("nsmlx")); ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--project", default="runs"); ap.add_argument("--name", default="train")
    a = ap.parse_args()
    y = YOLO(verbose=False)
    y.model = build(a.nc, a.scale)
    print("skipped (shape mismatch):", load_npz_partial(y.model, a.weights))
    init_cls_bias(y.model, a.nc)
    y._setup_metadata()
    y.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.project,
            name=a.name, exist_ok=True, val=False)


if __name__ == "__main__":
    main()
