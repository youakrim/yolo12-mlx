"""Convert an Ultralytics YOLO12 .pt checkpoint to an .npz readable by `load_npz`.

Needs torch + ultralytics (not MLX):  python -m yolo12mlx.convert best.pt best.npz
"""
import sys

import numpy as np
from ultralytics import YOLO


def main(src: str, dst: str) -> None:
    out = {}
    for k, v in YOLO(src).model.state_dict().items():
        if k.endswith("num_batches_tracked"):
            continue
        v = v.float().cpu().numpy()
        if v.ndim == 4:  # OIHW -> OHWI (MLX layout)
            v = v.transpose(0, 2, 3, 1)
        out[k] = v
    np.savez(dst, **out)
    print(f"{len(out)} tensors -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
