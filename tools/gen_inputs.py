#!/usr/bin/env python3
"""生成随机的 fp16 测试输入 raw。

matmul：A[M,K], B[K,N]，可选 bias[M,N]。
ewise（--ewise）：A[M,N], B[M,N] 同形状。

seed 固定可复现（出问题能原样重跑）。用法:
    gen_inputs.py <out_dir> --M M --N N --K K [--bias] [--seed S]
    gen_inputs.py <out_dir> --M M --N N --K K --ewise [--nonzero-b]
"""
import argparse
import os

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="输出目录（a.raw/b.raw[/bias.raw] 写在这）")
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--bias", action="store_true", help="额外生成 bias[M,N]")
    ap.add_argument(
        "--ewise",
        action="store_true",
        help="逐元素模式：A/B 都是 [M,N]（K 忽略）",
    )
    ap.add_argument(
        "--nonzero-b",
        action="store_true",
        help="把 |B|<0.5 的值推到 ±0.5，避免逐元素除法除零",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    if a.ewise:
        A = rng.standard_normal((a.M, a.N)).astype(np.float16)
        B = rng.standard_normal((a.M, a.N)).astype(np.float16)
        if a.nonzero_b:
            B = np.where(np.abs(B) < np.float16(0.5), np.float16(0.5), B)
            B = B.astype(np.float16)
    else:
        A = rng.standard_normal((a.M, a.K)).astype(np.float16)
        B = rng.standard_normal((a.K, a.N)).astype(np.float16)

    A.tofile(os.path.join(a.out_dir, "a.raw"))
    B.tofile(os.path.join(a.out_dir, "b.raw"))
    print(f"wrote {os.path.join(a.out_dir, 'a.raw')} ({A.shape})")
    print(f"wrote {os.path.join(a.out_dir, 'b.raw')} ({B.shape})")

    if a.bias:
        bias = rng.standard_normal((a.M, a.N)).astype(np.float16)
        bias.tofile(os.path.join(a.out_dir, "bias.raw"))
        print(f"wrote {os.path.join(a.out_dir, 'bias.raw')} ({bias.shape})")


if __name__ == "__main__":
    main()
