#!/usr/bin/env python3
"""生成随机 fp16 数据 -> 跑 driver -> 与 golden 对比。

默认用 build/bin/matmul_golden；--driver 可指定一条 shell 命令模板
（含 {a} {b} {c} {bias} 占位符），用于跑编译器产出的 .easm。
"""
import argparse
import os
import subprocess
import sys

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "tests", "data")
DEFAULT_DRIVER = os.path.join(ROOT, "build", "bin", "matmul_golden")
TOL = 1e-2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument(
        "--driver",
        default=None,
        help="shell command template with {a} {b} {c} {bias}; default=matmul_golden",
    )
    ap.add_argument(
        "--bias",
        default=None,
        help="write a bias tensor to this path and include it in the reference",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((args.M, args.K)).astype(np.float16)
    b = rng.standard_normal((args.K, args.N)).astype(np.float16)

    a_path = os.path.join(RAW_DIR, "a.raw")
    b_path = os.path.join(RAW_DIR, "b.raw")
    c_path = os.path.join(RAW_DIR, "c.raw")
    a.tofile(a_path)
    b.tofile(b_path)

    bias_path = None
    bias = None
    if args.bias:
        bias_path = args.bias
        bias = rng.standard_normal((args.M, args.N)).astype(np.float16)
        bias.tofile(bias_path)

    if args.driver:
        cmd = args.driver.format(a=a_path, b=b_path, c=c_path, bias=bias_path or "")
        subprocess.run(cmd, shell=True, check=True)
    else:
        subprocess.run([DEFAULT_DRIVER, a_path, b_path, c_path], check=True)

    c = np.fromfile(c_path, dtype=np.float16).reshape(args.M, args.N)

    if HAS_TORCH:
        golden = (torch.from_numpy(a).float() @ torch.from_numpy(b).float()).half()
        if bias is not None:
            golden = golden + torch.from_numpy(bias).float().half()
        golden = golden.numpy()
        print("golden: PyTorch fp16")
    else:
        golden = a.astype(np.float64) @ b.astype(np.float64)
        if bias is not None:
            golden += bias.astype(np.float64)
        print("golden: numpy fp64 (torch 未安装)")

    err = float(np.abs(c.astype(np.float32) - golden).max() / np.abs(golden).max())
    print(f"max rel err = {err:.3e}  (tolerance {TOL})")
    ok = err < TOL
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
