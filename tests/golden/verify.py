#!/usr/bin/env python3
"""把 Simulator 产出的 out.raw 与 PyTorch fp16 参考（a@b，可选 +bias）对拍。

用法:
    verify.py <c.raw> <a.raw> <b.raw> [--M M --N N --K K] [--bias bias.raw]

raw 都是打包的 fp16、不带 shape（M/N/K 默认 128）。c 的形状是 [M, N]。
"""
import argparse
import os

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

TOL = 1e-2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("c", help="simulator 产出的输出 raw [M,N]")
    ap.add_argument("a", help="输入 A raw [M,K]")
    ap.add_argument("b", help="输入 B raw [K,N]")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--bias", default=None, help="可选 bias raw [M,N]")
    return ap.parse_args()


def main():
    a = parse_args()
    A = np.fromfile(a.a, dtype=np.float16).reshape(a.M, a.K)
    B = np.fromfile(a.b, dtype=np.float16).reshape(a.K, a.N)
    C = np.fromfile(a.c, dtype=np.float16).reshape(a.M, a.N)

    if HAS_TORCH:
        golden = (torch.from_numpy(A).float() @ torch.from_numpy(B).float()).half()
        if a.bias:
            bias = np.fromfile(a.bias, dtype=np.float16).reshape(a.M, a.N)
            golden = golden + torch.from_numpy(bias).float().half()
        golden = golden.numpy()
        print("golden: PyTorch fp16")
    else:
        golden = A.astype(np.float64) @ B.astype(np.float64)
        if a.bias:
            golden += np.fromfile(a.bias, dtype=np.float16).reshape(a.M, a.N).astype(np.float64)
        print("golden: numpy fp64 (torch 未安装)")

    err = float(np.abs(C.astype(np.float32) - golden).max() / np.abs(golden).max())
    print(f"max rel err = {err:.3e}  (tolerance {TOL})")
    ok = err < TOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
