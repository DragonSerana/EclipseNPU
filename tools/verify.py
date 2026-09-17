#!/usr/bin/env python3
"""把 Simulator 产出的 out.raw 与参考实现对拍。

matmul/ewise：参考 = A@B (+bias) (+relu) 或 A <op> B（A/B 同形状 [M,N]），
输出 cosine + 归一化 max rel err。
act（一元）：参考 = fp64 算完按 RNE 舍到 fp16（docs/spec/accuracy.md 的契约参考），
输出 max ulp + 特殊值失配数。用法:
    verify.py <c.raw> <a.raw> [b.raw] [--M M --N N --K K]
              [--bias bias.raw] [--relu] [--ewise add|sub|mul|div]
              [--act relu|exp|rsqrt] [--ulp-tol N]
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
ULP_TOL = 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("c")
    ap.add_argument("a")
    ap.add_argument("b", nargs="?", default=None, help="ACT（一元）不需要")
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
    ap.add_argument(
        "--act",
        choices=["relu", "exp", "rsqrt"],
        default=None,
        help="一元 ACT 对拍：参考 = fp64 <op> A 后舍到 fp16，按 ulp 判",
    )
    ap.add_argument(
        "--broadcast",
        choices=["row", "col", "blk"],
        default=None,
        help="ewise 的 B 缩小：row=[1,N]、col=[M,1]、blk=[M,blk]（逐行按 j mod blk 读）",
    )
    ap.add_argument("--blk", type=int, default=32)
    ap.add_argument("--ulp-tol", type=int, default=ULP_TOL)
    ap.add_argument("--cos-tol", type=float, default=COS_TOL)
    ap.add_argument("--err-tol", type=float, default=ERR_TOL)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def expand_rhs(B, shape, bcast, blk):
    """把缩小的 B 按广播读模式展开成 [M,N]，后面直接套已有的参考公式。"""
    if bcast == "blk":
        return np.ascontiguousarray(B[:, np.arange(shape[1]) % blk])
    if bcast:
        return np.ascontiguousarray(np.broadcast_to(B, shape))
    return B


def ordered_key(bits):
    """fp16 位模式 -> 保序整数。两个 key 之差就是 ulp 距离（inf/NaN 也保序）。"""
    b = np.asarray(bits, dtype=np.uint16)
    neg = (b & 0x8000) != 0
    return np.where(neg, (~b) & 0xFFFF, b | 0x8000).astype(np.int64)


def act_reference(kind, A):
    """契约参考：fp64 计算 + RNE 舍到 fp16（docs/spec/accuracy.md）。

    刻意不夹紧：exp 正溢出就该是 IEEE inf（cmodel 也是），交给调用方比特殊值。
    """
    x = A.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        if kind == "relu":
            g = np.maximum(x, 0.0)
        elif kind == "exp":
            g = np.exp(x)
        else:  # rsqrt
            g = 1.0 / np.sqrt(x)
        return g.astype(np.float16)


def act_ulp(c_bits, g_bits):
    """返回 (max_ulp, 特殊值失配数)。

    inf/NaN 不参与 ulp 比较：契约要求它们逐位一致（正溢出 -> 同号 inf，不饱和，
    NaN 不变成有限值）。有限结果才比 ulp（正规域契约 <= 1 ulp）。
    """
    c_exp, g_exp = (c_bits >> 10) & 0x1F, (g_bits >> 10) & 0x1F
    c_man, g_man = c_bits & 0x3FF, g_bits & 0x3FF
    c_inf = (c_exp == 0x1F) & (c_man == 0)
    g_inf = (g_exp == 0x1F) & (g_man == 0)
    c_nan = (c_exp == 0x1F) & (c_man != 0)
    g_nan = (g_exp == 0x1F) & (g_man != 0)

    # 必须成对出现；inf 还要求同号（NaN 载荷不苛求）
    bad = (c_inf != g_inf) | (c_nan != g_nan) | (c_inf & (c_bits != g_bits))
    finite = ~(c_inf | c_nan | g_inf | g_nan)
    if not finite.any():
        return 0, int(np.count_nonzero(bad))
    ulp = np.abs(ordered_key(c_bits[finite]) - ordered_key(g_bits[finite]))
    return int(ulp.max()), int(np.count_nonzero(bad))


def main():
    a = parse_args()
    if a.act and a.ewise:
        raise SystemExit("--act 与 --ewise 不能同时用")

    A = np.fromfile(a.a, dtype=np.float16)
    C = np.fromfile(a.c, dtype=np.float16).reshape(a.M, a.N)

    if a.act:
        G = act_reference(a.act, A.reshape(a.M, a.N))
        max_ulp, bad = act_ulp(C.view(np.uint16), G.view(np.uint16))
        ok = (max_ulp <= a.ulp_tol) and (bad == 0)
        if a.quiet:
            print(f"ulp={max_ulp} special={bad} {'PASS' if ok else 'FAIL'}")
        else:
            print("golden: numpy fp64 -> fp16 RNE (契约参考)")
            print(f"max ulp = {max_ulp}  (tol {a.ulp_tol})")
            print(f"special value mismatch = {bad}")
            print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    if a.b is None:
        raise SystemExit("matmul/ewise 需要一个 b.raw")
    B = np.fromfile(a.b, dtype=np.float16)
    if a.ewise:
        A = A.reshape(a.M, a.N)
        bshape = {
            "row": (1, a.N),
            "col": (a.M, 1),
            "blk": (a.M, a.blk),
        }.get(a.broadcast, (a.M, a.N))
        B = expand_rhs(B.reshape(bshape), (a.M, a.N), a.broadcast, a.blk)
    else:
        A = A.reshape(a.M, a.K)
        B = B.reshape(a.K, a.N)

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
