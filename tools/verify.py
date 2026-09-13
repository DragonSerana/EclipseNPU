#!/usr/bin/env python3
"""把 Simulator 产出的 out.raw 与 PyTorch fp16 参考对拍，输出 cosine + max rel err。

参考 = A@B (+bias) (+relu)；加 --ewise 时参考 = A <op> B（A/B 同形状 [M,N]）。用法:
    verify.py <c.raw> <a.raw> <b.raw> [--M M --N N --K K]
              [--bias bias.raw] [--relu] [--ewise add|sub|mul|div]
              [--cos-tol F] [--err-tol F] [--quiet]
"""
import argparse

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

COS_TOL = 0.999
ERR_TOL = 1e-2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("c")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--bias", default=None)
    ap.add_argument("--relu", action="store_true")
    ap.add_argument(
        "--ewise",
        choices=["add", "sub", "mul", "div"],
        default=None,
        help="逐元素对拍：A/B 都是 [M,N]，参考 = A <op> B",
    )
    ap.add_argument("--cos-tol", type=float, default=COS_TOL)
    ap.add_argument("--err-tol", type=float, default=ERR_TOL)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    A = np.fromfile(a.a, dtype=np.float16)
    B = np.fromfile(a.b, dtype=np.float16)
    if a.ewise:
        A = A.reshape(a.M, a.N)
        B = B.reshape(a.M, a.N)
    else:
        A = A.reshape(a.M, a.K)
        B = B.reshape(a.K, a.N)
    C = np.fromfile(a.c, dtype=np.float16).reshape(a.M, a.N)

    if HAS_TORCH:
        Af = torch.from_numpy(A).float()
        Bf = torch.from_numpy(B).float()
        if a.ewise == "add":
            golden = (Af + Bf).half()
        elif a.ewise == "sub":
            golden = (Af - Bf).half()
        elif a.ewise == "mul":
            golden = (Af * Bf).half()
        elif a.ewise == "div":
            golden = (Af / Bf).half()
        else:
            golden = (Af @ Bf).half()
        if a.bias:
            golden = golden + torch.from_numpy(
                np.fromfile(a.bias, dtype=np.float16).reshape(a.M, a.N)
            ).float().half()
        if a.relu:
            golden = torch.relu(golden) if hasattr(torch, "relu") else golden
        golden = golden.numpy().astype(np.float32)
        ref = "PyTorch fp16"
    else:
        if a.ewise == "add":
            golden = A.astype(np.float64) + B.astype(np.float64)
        elif a.ewise == "sub":
            golden = A.astype(np.float64) - B.astype(np.float64)
        elif a.ewise == "mul":
            golden = A.astype(np.float64) * B.astype(np.float64)
        elif a.ewise == "div":
            golden = A.astype(np.float64) / B.astype(np.float64)
        else:
            golden = A.astype(np.float64) @ B.astype(np.float64)
        if a.bias:
            golden += np.fromfile(a.bias, dtype=np.float16).reshape(a.M, a.N).astype(np.float64)
        if a.relu:
            golden = np.maximum(golden, 0.0)
        ref = "numpy fp64 (torch 未安装)"

    Cf = C.astype(np.float32)
    Gf = golden.astype(np.float32)
    # cosine 相似度（形状/方向对齐）
    denom = np.linalg.norm(Cf) * np.linalg.norm(Gf)
    cos = float(np.dot(Cf.ravel(), Gf.ravel()) / denom) if denom > 0 else 0.0
    # 归一化 max rel err（幅度精度）
    err = float(np.abs(Cf - Gf).max() / np.abs(Gf).max()) if np.abs(Gf).max() > 0 else 0.0

    ok_cos = cos >= a.cos_tol
    ok_err = err < a.err_tol
    ok = ok_cos and ok_err
    if not a.quiet:
        print(f"golden: {ref}")
        print(f"cosine = {cos:.6f}  (tol {a.cos_tol})")
        print(f"max rel err = {err:.3e}  (tol {a.err_tol})")
        print("PASS" if ok else "FAIL")
    else:
        print(f"cosine={cos:.6f} err={err:.3e} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
