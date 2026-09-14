#!/usr/bin/env python3
"""EXP / RSQRT 的 LUT 精度实验（纯 numpy，不涉及 cmodel）。

目的：给 ISA v0.2 的超越函数精度类提供数据背书。合同只定精度类（误差上界），
实现算法（LUT 项数、插值、迭代次数）由本实验冻结。方法见 docs/plans/h3-plan.md §7。

做法：**穷举所有有限 fp16 取值**作为输入（不是随机采样），所以结论是确定的。
判据分两层：
  1. fp32 相对误差——算法本身的精度；
  2. fp16 输出——和"fp64 真值正确舍入到 fp16"比，看差几个 ulp。
     fp16 尾数只有 10 位（half-ulp = 2^-11 ≈ 4.9e-4），LUT 误差低于它就"免费"。

用法: python3 scripts/lut_experiment.py
输出 markdown 表，抄进 docs/spec/accuracy.md。
"""
import numpy as np

LOG2E = np.float32(1.4426950408889634)
FP16_HALF_ULP = 2.0**-11
FP16_MIN_NORMAL = np.float16(2.0**-14).astype(np.float32)


# --------------------------------------------------------------------------
# EXP: e^x = 2^(x*log2e) = 2^n * 2^f, f in [0,1)，2^f 查表 + 线性插值
# --------------------------------------------------------------------------
def exp_lut_table(N):
    return (2.0 ** (np.arange(N + 1) / N)).astype(np.float32)


def exp_lut(x, lut):
    """全程 fp32，模拟定点功能单元。"""
    N = len(lut) - 1
    t = (x.astype(np.float32) * LOG2E).astype(np.float32)
    n = np.floor(t).astype(np.float32)
    f = (t - n).astype(np.float32)
    idx = (f * np.float32(N)).astype(np.float32)
    i = np.clip(np.floor(idx).astype(np.int64), 0, N - 1)
    w = (idx - i.astype(np.float32)).astype(np.float32)
    y = (lut[i] + w * (lut[i + 1] - lut[i])).astype(np.float32)
    return np.ldexp(y, n.astype(np.int32))  # *2^n 是精确的


# --------------------------------------------------------------------------
# RSQRT: 1/sqrt(x)。x = m * 2^(2k), m in [1,4)，查表 + 线性插值，可选 1 次牛顿
# --------------------------------------------------------------------------
def rsqrt_lut_table(N):
    m = 1.0 + 3.0 * np.arange(N + 1) / N
    return (1.0 / np.sqrt(m)).astype(np.float32)


def rsqrt_lut(x, lut, newton=True):
    N = len(lut) - 1
    x = x.astype(np.float32)
    e = np.floor(np.log2(x)).astype(np.int64)
    e = e - (e & 1)  # 取偶数指数
    m = (x / np.ldexp(np.float32(1.0), e.astype(np.int32))).astype(np.float32)
    idx = ((m - np.float32(1.0)) / np.float32(3.0) * np.float32(N)).astype(np.float32)
    i = np.clip(np.floor(idx).astype(np.int64), 0, N - 1)
    w = (idx - i.astype(np.float32)).astype(np.float32)
    y = (lut[i] + w * (lut[i + 1] - lut[i])).astype(np.float32)
    y = (y * np.ldexp(np.float32(1.0), (-e // 2).astype(np.int32))).astype(np.float32)
    if newton:
        y = (y * (np.float32(1.5) - np.float32(0.5) * x * y * y)).astype(np.float32)
    return y


# --------------------------------------------------------------------------
def all_finite_fp16():
    bits = np.arange(0x0000, 0x10000, dtype=np.uint16)
    h = bits.view(np.float16)
    return np.unique(h[np.isfinite(h)]).astype(np.float64)


def stats(got32, ref64):
    """返回 (n, max_rel_fp32, mismatch, bit_exact_rate, max_ulp)。只在正规域统计。"""
    ref16 = ref64.astype(np.float16)
    got16 = got32.astype(np.float16)
    r32 = ref16.astype(np.float32)
    normal = np.isfinite(r32) & (r32 >= FP16_MIN_NORMAL)
    n = int(normal.sum())
    if n == 0:
        return 0, 0.0, 0, 0.0, 0.0
    rel = float(
        np.max(np.abs(got32[normal].astype(np.float64) - ref64[normal]) / ref64[normal])
    )
    mis = int(np.sum(got16[normal] != ref16[normal]))
    expo = np.floor(np.log2(np.abs(r32[normal])))
    ulp = np.ldexp(np.float64(1.0), (expo - 10).astype(np.int64))
    maxulp = float(
        np.max(
            np.abs(got16[normal].astype(np.float64) - r32[normal].astype(np.float64))
            / ulp
        )
    )
    return n, rel, mis, 1.0 - mis / n, maxulp


def row(name, got, ref):
    n, rel, mis, rate, maxulp = stats(got, ref)
    print(f"| {name} | {rel:.3e} | {rate*100:.4f}% ({mis}/{n}) | {maxulp:.2f} |")


def main():
    xs = all_finite_fp16()

    print(f"输入：穷举所有有限 fp16 取值（{len(xs)} 个）。")
    print(f"fp16 half-ulp = {FP16_HALF_ULP:.3e}；先验插值误差 ≈ 0.06/N²。")
    print()

    print("### EXP: e^x = 2^(x·log2e)，2^f 均匀 LUT + 线性插值")
    print()
    print("| N | fp32 max rel err | fp16 逐位匹配率（正规域） | max ulp 差 |")
    print("| --- | --- | --- | --- |")
    xe = xs[(xs >= -30.0) & (xs <= 11.0)]
    for N in [32, 64, 128, 256]:
        row(f"{N}", exp_lut(xe, exp_lut_table(N)), np.exp(xe))
    print()

    print("### RSQRT: 1/sqrt(x)，m∈[1,4) 均匀 LUT + 线性插值 + 1 次牛顿")
    print()
    print("| N | fp32 max rel err | fp16 逐位匹配率（正规域） | max ulp 差 |")
    print("| --- | --- | --- | --- |")
    xr = xs[xs > 0]
    for N in [32, 64]:
        row(f"{N}", rsqrt_lut(xr, rsqrt_lut_table(N), True), 1.0 / np.sqrt(xr))
    print()

    print("### RSQRT 对照：不做牛顿迭代（只查表）")
    print()
    print("| N | fp32 max rel err | fp16 逐位匹配率（正规域） | max ulp 差 |")
    print("| --- | --- | --- | --- |")
    for N in [32, 64, 128, 256]:
        row(f"{N}", rsqrt_lut(xr, rsqrt_lut_table(N), False), 1.0 / np.sqrt(xr))


if __name__ == "__main__":
    main()
