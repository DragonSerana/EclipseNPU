#!/usr/bin/env python3
"""端到端精度测试：对每个 case，生成随机输入 -> 编译 .easm -> 跑 Simulator -> 对拍 + 查 hazard。

用法: accuracy_check.py [--seed S] [--layout bump|golden-mirror] [--filter SUBSTR]
每个 case 一行汇总: name | cosine | max rel err | cycle | hazard | PASS/FAIL。
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EOPT = os.path.join(ROOT, "build", "bin", "eclipse-opt")
ERUN = os.path.join(ROOT, "build", "bin", "eclipse-run")
GEN = os.path.join(ROOT, "tools", "gen_inputs.py")
VERIFY = os.path.join(ROOT, "tools", "verify.py")
HAZARD = os.path.join(ROOT, "tools", "hazard_check.py")
E2E = os.path.join(ROOT, "tests", "e2e")
WORK = os.path.join(ROOT, "build", "tests-data")

# case table：链路 x 尺寸。seed 固定 -> 可复现；bias/relu 决定参考公式。
CASES = [
    {"name": "matmul_128",       "M": 128, "N": 128, "K": 128, "bias": False, "relu": False},
    {"name": "matmul_add_128",   "M": 128, "N": 128, "K": 128, "bias": True,  "relu": False},
    {"name": "matmul_add_relu_128", "M": 128, "N": 128, "K": 128, "bias": True, "relu": True},
    {"name": "matmul_16",        "M": 16,  "N": 16,  "K": 16,  "bias": False, "relu": False},
    {"name": "matmul_add_16",    "M": 16,  "N": 16,  "K": 16,  "bias": True,  "relu": False},
    {"name": "matmul_add_relu_16", "M": 16, "N": 16,  "K": 16,  "bias": True,  "relu": True},
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gen_inputs(case, seed, out_dir):
    cmd = [sys.executable, GEN, out_dir, "--M", str(case["M"]), "--N", str(case["N"]),
           "--K", str(case["K"]), "--seed", str(seed)]
    if case["bias"]:
        cmd.append("--bias")
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"gen_inputs failed: {r.stderr}")


def compile_mlir(mlir, easm, layout):
    cmd = [EOPT, mlir,
           "--one-shot-bufferize=bufferize-function-boundaries",
           "--convert-linalg-to-eclipse",
           "--eclipse-elide-copies",
           "--canonicalize",
           f"--eclipse-allocate=layout={layout}",
           f"--eclipse-to-easm=output-easm={easm}"]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed: {r.stderr}")


def run_sim(easm, out_raw, inputs):
    cmd = [ERUN, easm, out_raw] + inputs
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"eclipse-run failed: {r.stderr}")
    # cycle 从 eclipse-run 的 stdout 里取 "total = N" 或 "total cycle = N"
    for line in r.stdout.splitlines():
        if line.startswith("total cycle =") or line.startswith("total ="):
            return int(line.split("=")[1].strip())
    return None


def check_verify(case, out_raw, data_dir):
    cmd = [sys.executable, VERIFY, out_raw,
           os.path.join(data_dir, "a.raw"), os.path.join(data_dir, "b.raw"),
           "--M", str(case["M"]), "--N", str(case["N"]), "--K", str(case["K"]),
           "--quiet"]
    if case["bias"]:
        cmd += ["--bias", os.path.join(data_dir, "bias.raw")]
    if case["relu"]:
        cmd.append("--relu")
    r = run(cmd)
    out = r.stdout.strip()
    # 解析 "cosine=... err=... PASS/FAIL"
    cos, err = None, None
    if "cosine=" in out:
        cos = float(out.split("cosine=")[1].split()[0])
    if "err=" in out:
        err = float(out.split("err=")[1].split()[0])
    return r.returncode == 0, cos, err


def check_hazard(easm):
    r = run([sys.executable, HAZARD, easm])
    if r.returncode == 0:
        return True, 0
    return False, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", default="bump")
    ap.add_argument("--filter", default=None, help="只跑名字含该子串的 case")
    a = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    print(f"{'case':<20} {'cosine':>9} {'max rel err':>12} {'cycle':>7} {'hazard':>6}  result")
    all_ok = True
    for case in CASES:
        name = case["name"]
        if a.filter and a.filter not in name:
            continue
        data_dir = os.path.join(WORK, name)
        easm = os.path.join(WORK, name + ".easm")
        out_raw = os.path.join(WORK, name + ".raw")
        try:
            gen_inputs(case, a.seed, data_dir)
            compile_mlir(os.path.join(E2E, name + ".mlir"), easm, a.layout)
            inputs = [os.path.join(data_dir, "a.raw"), os.path.join(data_dir, "b.raw")]
            if case["bias"]:
                inputs.append(os.path.join(data_dir, "bias.raw"))
            cycle = run_sim(easm, out_raw, inputs)
            ok_num, cos, err = check_verify(case, out_raw, data_dir)
            ok_haz, _ = check_hazard(easm)
            ok = ok_num and ok_haz
            all_ok = all_ok and ok
            print(f"{name:<20} {cos:>9.6f} {err:>12.3e} {str(cycle):>7} "
                  f"{'ok' if ok_haz else 'HZ!':>6}  {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            all_ok = False
            print(f"{name:<20} {'-':>9} {'-':>12} {'-':>7} {'-':>6}  ERROR: {e}")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
