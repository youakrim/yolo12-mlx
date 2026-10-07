"""Forward+backward timing: fp32/bf16/fp16, eager vs mx.compile (synthetic loss).

    python -m yolo12mlx.bench --batch 6 --imgsz 896
"""
import argparse
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from .model import build, init_cls_bias


def timeit(f, n=6):
    f(); f(); mx.synchronize()
    t = time.time()
    for _ in range(n):
        f()
    mx.synchronize()
    return (time.time() - t) / n


def run(label, batch, size, dtype=None, compile_=False, nc=6):
    m = build(nc); init_cls_bias(m, nc); m.train()
    x = mx.random.uniform(shape=(batch, size, size, 3))
    if dtype is not None:
        m.apply(lambda a: a.astype(dtype)); x = x.astype(dtype)

    def f(mod, x):
        p = mod(x)
        return mx.mean(p["scores"].astype(mx.float32)) + mx.mean(p["boxes"].astype(mx.float32))

    vg = nn.value_and_grad(m, f)
    if compile_:
        st = [m.state]

        @partial(mx.compile, inputs=st, outputs=st)
        def step(x):
            return vg(m, x)
    else:
        step = lambda x: vg(m, x)

    def go():
        l, g = step(x); mx.eval(l, g)

    print(f"{label:16s} {timeit(go):.3f} s/it", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--imgsz", type=int, default=896)
    a = ap.parse_args()
    run("fp32 eager", a.batch, a.imgsz)
    run("fp32 compile", a.batch, a.imgsz, None, True)
    run("fp16 compile", a.batch, a.imgsz, mx.float16, True)
    run("bf16 compile", a.batch, a.imgsz, mx.bfloat16, True)
