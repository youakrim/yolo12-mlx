"""Inference: letterbox, forward, class-aware NMS, back to original pixels.

    python -m yolo12mlx.predict weights.npz page.jpg --nc 6 --imgsz 896 --bf16
"""
import argparse
import time

import cv2
import mlx.core as mx
import numpy as np

from .model import build, load_npz


def letterbox(im, size, stride=32):
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (stride - nw % stride) % stride / 2, (stride - nh % stride) % stride / 2
    top, bot, left, right = round(ph - 0.1), round(ph + 0.1), round(pw - 0.1), round(pw + 0.1)
    im = cv2.copyMakeBorder(im, top, bot, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, left, top


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[i, 0], boxes[rest, 0]); y0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[i, 2], boxes[rest, 2]); y1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        a = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        order = rest[inter / (a(boxes[i]) + a(boxes[rest]) - inter + 1e-9) <= iou_thr]
    return keep


class Detector:
    def __init__(self, weights: str, nc: int, bf16: bool = False, imgsz: int = 896):
        self.model = build(nc)
        missing, _ = load_npz(self.model, weights)
        self.model.eval()
        self.bf16, self.imgsz, self.nc = bf16, imgsz, nc
        if bf16:
            self.model.apply(lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a)

    def __call__(self, path, conf=0.25, iou=0.7, max_det=300):
        """Returns an (n, 6) array: x0, y0, x1, y1, confidence, class (original pixels)."""
        im0 = cv2.imread(path)
        h, w = im0.shape[:2]
        im, r, left, top = letterbox(im0, self.imgsz)
        x = mx.array(im[..., ::-1][None].astype(np.float32) / 255.0)
        out = self.model(x.astype(mx.bfloat16) if self.bf16 else x).astype(mx.float32)
        out = np.array(out)[0]  # (N, 4 + nc), boxes as cx cy w h
        cls_scores = out[:, 4:]
        score, cls = cls_scores.max(1), cls_scores.argmax(1)
        m = score > conf
        xywh, score, cls = out[m, :4], score[m], cls[m]
        xyxy = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                         xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        keep = []
        for c in np.unique(cls):  # class-aware NMS
            idx = np.where(cls == c)[0]
            keep += list(idx[nms(xyxy[idx], score[idx], iou)])
        keep = sorted(keep, key=lambda i: -score[i])[:max_det]
        b = xyxy[keep].copy()
        b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w)
        b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h)
        return np.concatenate([b, score[keep, None], cls[keep, None]], 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights"); ap.add_argument("image")
    ap.add_argument("--nc", type=int, default=80); ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    det = Detector(a.weights, a.nc, a.bf16, a.imgsz)
    det(a.image)  # warm-up
    t = time.perf_counter(); res = det(a.image, a.conf); dt = time.perf_counter() - t
    for r in res:
        print(" ".join(f"{v:.1f}" for v in r[:4]), f"conf={r[4]:.2f} class={int(r[5])}")
    print(f"{len(res)} detections in {dt * 1000:.1f} ms")
