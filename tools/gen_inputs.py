#!/usr/bin/env python3
"""生成随机的 fp16 测试输入 raw。

matmul：A[M,K], B[K,N]，可选 bias[M,N]。
ewise（--ewise）：A[M,N], B[M,N] 同形状。
act（--act）：A[M,N] 单输入（K 忽略）。

seed 固定可复现（出问题能原样重跑）。用法:
    gen_inputs.py <out_dir> --M M --N N --K K [--bias] [--seed S]
    gen_inputs.py <out_dir> --M M --N N --K K --ewise [--nonzero-b]
    gen_inputs.py <out_dir> --M M --N N --K K --act relu|exp|rsqrt
"""
import argparse
import os

import numpy as np

# ACT 各算子的输入值域。exp 故意铺满 ±12：负端 exp(-12)≈6.1e-6 落到 fp16
# 次正规，正端 exp(12)≈1.6e5 超出 fp16 上限变 inf —— 一次 e2e 就能同时压到
# 「渐进下溢不 FTZ」和「正溢出变 inf 不饱和」两条契约（见 docs/spec/accuracy.md）。
ACT_RANGES = {
    "relu": ("normal", 0.0, 0.0),  # 标准正态，正负都要有
    "exp": ("uniform", -12.0, 12.0),
    "rsqrt": ("loguniform", -3.0, 3.0),  # exp(U(-3,3)) -> x∈[0.05,20]，rsqrt∈[0.22,4.5]
}


def act_values(kind, count, rng):
    """按算子值域生成 ACT 输入（原样返回 fp64，由调用方转 fp16）。"""
    dist, lo, hi = ACT_RANGES[kind]
    if dist == "normal":
        return rng.standard_normal(count)
    if dist == "uniform":
        return rng.uniform(lo, hi, count)
    return np.exp(rng.uniform(lo, hi, count))


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
    ap.add_argument(
        "--act",
        choices=sorted(ACT_RANGES),
        default=None,
        help="ACT 模式：只生成单输入 a.raw[M,N]（K 忽略）",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    if a.act and a.ewise:
        raise SystemExit("--act 与 --ewise 不能同时用")
    if a.act and a.bias:
        raise SystemExit("--act 是一元算子，没有 bias")
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    if a.act:
        A = act_values(a.act, a.M * a.N, rng).astype(np.float16).reshape(a.M, a.N)
        A.tofile(os.path.join(a.out_dir, "a.raw"))
        print(f"wrote {os.path.join(a.out_dir, 'a.raw')} ({A.shape}) act={a.act}")
        return

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
