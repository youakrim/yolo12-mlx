# yolo12-mlx

YOLO12 (A2C2f / area attention, DFL head) for Apple silicon, running natively on [MLX](https://github.com/ml-explore/mlx). Weights trained with [Ultralytics](https://github.com/ultralytics/ultralytics) load directly; training and inference both run on MLX.

The model code builds on [thewebAI/yolo-mlx](https://github.com/thewebAI/yolo-mlx), which implements YOLO26. This repository adds the YOLO12 architecture config and the fixes needed to train a non-end-to-end (DFL + NMS) head there, plus the tooling around it: weight conversion, inference with NMS, and a benchmark.

## What is in here

| Path | Purpose |
|---|---|
| `patches/yolo-mlx-yolo12.patch` | Patch against yolo-mlx `abbcb79` (config, loss fixes, faster depthwise conv, compiled bf16 training step, threaded data loader) |
| `yolo12mlx/model.py` | Build YOLO12, map Ultralytics weight names, partial load, classification-bias init |
| `yolo12mlx/convert.py` | Ultralytics `.pt` to `.npz` (needs `torch` and `ultralytics`) |
| `yolo12mlx/predict.py` | Letterbox, forward, NMS, boxes back in original pixels |
| `yolo12mlx/train.py` | Fine-tuning launcher |
| `yolo12mlx/bench.py` | Forward + backward timing, fp32 / fp16 / bf16, eager vs `mx.compile` |

## Install

```sh
scripts/setup.sh    # clones yolo-mlx at the right commit, applies the patch, installs both
```

Python 3.10+, `mlx>=0.30.3,<0.31`. Apple silicon only.

## Pretrained weights

The official YOLO12 COCO weights for all five scales (n, s, m, l, x), already converted: [huggingface.co/youakrim/yolo12-mlx](https://huggingface.co/youakrim/yolo12-mlx).

```sh
hf download youakrim/yolo12-mlx yolo12s-coco.npz --local-dir .
```

## Use

```sh
# 1. Convert your own Ultralytics checkpoint (any size, any number of classes)
python -m yolo12mlx.convert best.pt best.npz

# 2. Detect
python -m yolo12mlx.predict best.npz page.jpg --nc 6 --imgsz 896 --bf16

# 3. Fine-tune from converted COCO weights
python -m yolo12mlx.train --data data.yaml --weights yolo12s-coco.npz --nc 6 --imgsz 896 --batch 6 --epochs 3
```

Pass `--scale n|s|m|l|x` to `predict`, `train` and `bench` (default `s`). Training data uses the YOLO layout (`images/<split>`, `labels/<split>`, `data.yaml` with `nc` and `names`).

Environment variables read by the trainer: `YOLO_PRECISION` (`bf16` default, `fp16`, `fp32`), `YOLO_COMPILE` (`1` default), `YOLO_LOADER_WORKERS` (default 6).

## Results

Measured on a MacBook Pro M5 (10-core GPU, 32 GB), YOLO12s, 896 px.

**Fidelity.** On a document layout test set (200 pages, 1771 boxes, 6 classes, IoU > 0.5), the same trained weights give identical results in Ultralytics (PyTorch MPS), MLX fp32 and MLX bf16: 1701 / 1771 correct boxes in each case. On COCO weights the raw outputs match Ultralytics to 0.0016 on scores and about 1 px on boxes.

**Scales.** All five scales (n, s, m, l, x) were checked against the official COCO weights at 640 px: parameter counts match Ultralytics (2.6M, 9.3M, 20.3M, 26.6M, 59.4M) and outputs agree to within 0.011 on scores and 3 px on boxes. Forward and backward also run for `m` and `l` (bf16, `mx.compile`: 0.55 s/it for `m` at batch 4 / 640 px, 0.40 s/it for `l` at batch 2 / 640 px). Only `s` has been trained and evaluated end to end. `l` and `x` need a lot of memory: `l` in fp32 at batch 4 / 896 px exhausted 32 GB.

**Inference**, per page, model only: PyTorch MPS 25.3 ms, MLX fp32 30.9 ms, MLX bf16 23.1 ms.

**Training step** (batch 6, 896 px, forward + backward with a simplified loss):

| | s / iteration |
|---|---|
| fp32, eager | 1.79 |
| fp32, `mx.compile` | 1.44 |
| bf16, `mx.compile` | 0.91 |

End to end with the real loss, optimizer and data loader: about 1.15 to 1.2 s/it, against 1.2 to 1.4 s/it for PyTorch MPS (measured in a separate session, so the two ranges overlap; not a controlled comparison).

## What the patch changes

- `cfg/models/12/yolo12.yaml`: YOLO12 with `end2end: False`, `reg_max: 16`.
- Trainer uses the plain `v8DetectionLoss` for non-end-to-end models.
- `DFLoss` returns the right shape; `tal.py` gather rewritten to avoid a `(B, M, N, C)` broadcast.
- `Conv` and `AAttn.pe` accept a bias, as in the Ultralytics checkpoints.
- `A2C2f` gets the residual layer-scale (`gamma`) and `mlp_ratio` 1.2 for the `l` and `x` scales, as in Ultralytics.
- Depthwise convolutions use a custom VJP: MLX's native weight gradient for depthwise convolutions is about 30x its forward pass.
- The training step is `mx.compile`d, with bf16 compute on fp32 master weights and the loss in fp32.
- The data loader is lazy and threaded; upstream loaded the whole epoch into memory before the first step.

## Limitations

- Training quality has not been validated against PyTorch. Inference parity is verified; a full training recipe on MLX (and its accuracy) is not.
- bf16 training is only checked for a decreasing loss over a short run.
- Training and the end-to-end layout evaluation were only done with YOLO12s; m, l and x are verified for weight loading and forward parity only.
- Detection only.

## License

AGPL-3.0, like both Ultralytics YOLO12 and yolo-mlx. See `LICENSE`.
