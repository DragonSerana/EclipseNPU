#!/usr/bin/env python3
"""matmul 对拍脚本：生成随机 fp16 数据 → 跑 driver → 与 golden 对比。

golden 优先用 PyTorch fp16（fp32 累加，与 cmodel 语义一致）；
没有 torch 时退化为 numpy fp64 真值（更严格）。
"""
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
DRIVER = os.path.join(ROOT, "build", "bin", "matmul_golden")
M = N = K = 128
TOL = 1e-2


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((M, K)).astype(np.float16)
    b = rng.standard_normal((K, N)).astype(np.float16)

    a_path = os.path.join(RAW_DIR, "a.raw")
    b_path = os.path.join(RAW_DIR, "b.raw")
    c_path = os.path.join(RAW_DIR, "c.raw")
    a.tofile(a_path)
    b.tofile(b_path)
    subprocess.run([DRIVER, a_path, b_path, c_path], check=True)

    c = np.fromfile(c_path, dtype=np.float16).reshape(M, N)

    if HAS_TORCH:
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)
        golden = (ta.float() @ tb.float()).half()  # fp32 累加后舍回 fp16
        err = float((c.astype(np.float32) - golden.numpy()).max() /
                    np.abs(golden.numpy()).max())
        print(f"golden: PyTorch fp16")
    else:
        golden = (a.astype(np.float64) @ b.astype(np.float64))
        err = float(np.abs(c.astype(np.float64) - golden).max() /
                    np.abs(golden).max())
        print(f"golden: numpy fp64 (torch 未安装)")

    print(f"max rel err = {err:.3e}  (tolerance {TOL})")
    ok = err < TOL
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
