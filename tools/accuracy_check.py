#!/usr/bin/env python3
"""端到端精度测试：对每个 case，生成随机输入 -> 编译 .easm -> 跑 Simulator -> 对拍 + 查 hazard。

用法: accuracy_check.py [--seed S] [--layout bump|golden-mirror] [--filter SUBSTR]
每个 case 一行汇总: name | cosine | metric | cycle | hazard | PASS/FAIL。
matmul/ewise 的 metric 是归一化 max rel err，ACT 的是 max ulp（cosine 不适用，打 -）。
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

# op 取值唯一决定链路，三组互不相交：
#   "matmul"              A[M,K] @ B[K,N] (+bias) (+relu)
#   add/sub/mul/div       A <op> B，A/B 同形状 [M,N]（K 忽略）
#   relu/exp/rsqrt        ACT 一元算子，只要 A[M,N]
ACT_OPS = ("relu", "exp", "rsqrt")
EWISE_OPS = ("add", "sub", "mul", "div")

# case table：链路 x 尺寸。seed 固定 -> 可复现；op/bias/relu 决定参考公式。
CASES = [
    {"name": "matmul_128",       "M": 128, "N": 128, "K": 128, "op": "matmul", "bias": False, "relu": False},
    {"name": "matmul_add_128",   "M": 128, "N": 128, "K": 128, "op": "matmul", "bias": True,  "relu": False},
    {"name": "matmul_add_relu_128", "M": 128, "N": 128, "K": 128, "op": "matmul", "bias": True, "relu": True},
    {"name": "matmul_16",        "M": 16,  "N": 16,  "K": 16,  "op": "matmul", "bias": False, "relu": False},
    {"name": "matmul_add_16",    "M": 16,  "N": 16,  "K": 16,  "op": "matmul", "bias": True,  "relu": False},
    {"name": "matmul_add_relu_16", "M": 16, "N": 16,  "K": 16,  "op": "matmul", "bias": True,  "relu": True},
    {"name": "ewise_add_16",     "M": 16,  "N": 16,  "K": 1,   "op": "add",    "bias": False, "relu": False},
    {"name": "ewise_sub_16",     "M": 16,  "N": 16,  "K": 1,   "op": "sub",    "bias": False, "relu": False},
    {"name": "ewise_mul_16",     "M": 16,  "N": 16,  "K": 1,   "op": "mul",    "bias": False, "relu": False},
    {"name": "ewise_div_16",     "M": 16,  "N": 16,  "K": 1,   "op": "div",    "bias": False, "relu": False},
    {"name": "ewise_add_128",    "M": 128, "N": 128, "K": 1,   "op": "add",    "bias": False, "relu": False},
    {"name": "ewise_sub_128",    "M": 128, "N": 128, "K": 1,   "op": "sub",    "bias": False, "relu": False},
    {"name": "ewise_mul_128",    "M": 128, "N": 128, "K": 1,   "op": "mul",    "bias": False, "relu": False},
    {"name": "ewise_div_128",    "M": 128, "N": 128, "K": 1,   "op": "div",    "bias": False, "relu": False},
    {"name": "act_relu_16",      "M": 16,  "N": 16,  "K": 1,   "op": "relu",   "bias": False, "relu": False},
    {"name": "act_exp_16",       "M": 16,  "N": 16,  "K": 1,   "op": "exp",    "bias": False, "relu": False},
    {"name": "act_rsqrt_16",     "M": 16,  "N": 16,  "K": 1,   "op": "rsqrt",  "bias": False, "relu": False},
    {"name": "act_relu_128",     "M": 128, "N": 128, "K": 1,   "op": "relu",   "bias": False, "relu": False},
    {"name": "act_exp_128",      "M": 128, "N": 128, "K": 1,   "op": "exp",    "bias": False, "relu": False},
    {"name": "act_rsqrt_128",    "M": 128, "N": 128, "K": 1,   "op": "rsqrt",  "bias": False, "relu": False},
    # 广播：B 缩小成 [1,N]（row）、[M,1]（col）、[M,blk]（blk，逐行按 j mod blk 读）。
    # 128x128 是方阵，正好验证"靠 shape 里的 1 维区分"、而不是靠行列不相等。
    {"name": "ewise_bcast_row_128",     "M": 128, "N": 128, "K": 1, "op": "mul", "bcast": "row", "bias": False, "relu": False},
    {"name": "ewise_bcast_col_128",     "M": 128, "N": 128, "K": 1, "op": "mul", "bcast": "col", "bias": False, "relu": False},
    {"name": "ewise_bcast_col_div_128", "M": 128, "N": 128, "K": 1, "op": "div", "bcast": "col", "bias": False, "relu": False},
    {"name": "ewise_bcast_blk_128",     "M": 128, "N": 128, "K": 1, "op": "mul", "bcast": "blk", "blk": 32, "bias": False, "relu": False},
    {"name": "ewise_bcast_blk_32x128",  "M": 32,  "N": 128, "K": 1, "op": "mul", "bcast": "blk", "blk": 32, "bias": False, "relu": False},
    # REDUCE：A[M,K] 沿 K 归约成 [M,1]（N=1 是输出行宽）。K 是归约长度。
    {"name": "reduce_sum_128",        "M": 128, "N": 1, "K": 128, "op": "reduce", "reduce": "sum",        "bias": False, "relu": False},
    {"name": "reduce_square_sum_128", "M": 128, "N": 1, "K": 128, "op": "reduce", "reduce": "square_sum", "bias": False, "relu": False},
    {"name": "reduce_max_128",        "M": 128, "N": 1, "K": 128, "op": "reduce", "reduce": "max",        "bias": False, "relu": False},
]


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def bcast_args(case):
    """广播 case 传给 gen_inputs / verify.py 的公共参数。"""
    if not case.get("bcast"):
        return []
    args = ["--broadcast", case["bcast"]]
    if case["bcast"] == "blk":
        args += ["--blk", str(case.get("blk", 32))]
    return args


def gen_inputs(case, seed, out_dir):
    cmd = [sys.executable, GEN, out_dir, "--M", str(case["M"]), "--N", str(case["N"]),
           "--K", str(case["K"]), "--seed", str(seed)]
    op = case["op"]
    if op in ACT_OPS:
        cmd += ["--act", op]
    elif op == "reduce":
        cmd += ["--reduce", case["reduce"]]
    elif op in EWISE_OPS:
        cmd.append("--ewise")
        if op == "div":
            # 逐元素除法要避开除零
            cmd.append("--nonzero-b")
        cmd += bcast_args(case)
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


def parse_metrics(out):
    """解析 verify.py --quiet 的 "k=v" 串（matmul/ewise 给 cosine+err，ACT 给 ulp+special）。"""
    metrics = {}
    for tok in out.split():
        if "=" in tok:
            key, value = tok.split("=", 1)
            metrics[key] = value
    return metrics


def check_verify(case, out_raw, data_dir):
    op = case["op"]
    cmd = [sys.executable, VERIFY, out_raw, os.path.join(data_dir, "a.raw")]
    if op not in ACT_OPS and op != "reduce":
        cmd.append(os.path.join(data_dir, "b.raw"))
    cmd += ["--M", str(case["M"]), "--N", str(case["N"]), "--K", str(case["K"]),
            "--quiet"]
    if op in ACT_OPS:
        cmd += ["--act", op]
    elif op == "reduce":
        cmd += ["--reduce", case["reduce"]]
    elif op in EWISE_OPS:
        cmd += ["--ewise", op]
        cmd += bcast_args(case)
    if case["bias"]:
        cmd += ["--bias", os.path.join(data_dir, "bias.raw")]
    if case["relu"]:
        cmd.append("--relu")
    r = run(cmd)
    return r.returncode == 0, parse_metrics(r.stdout.strip())


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
    print(f"{'case':<20} {'cosine':>9} {'metric':>12} {'cycle':>7} {'hazard':>6}  result")
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
            inputs = [os.path.join(data_dir, "a.raw")]
            if case["op"] not in ACT_OPS and case["op"] != "reduce":
                inputs.append(os.path.join(data_dir, "b.raw"))
            if case["bias"]:
                inputs.append(os.path.join(data_dir, "bias.raw"))
            cycle = run_sim(easm, out_raw, inputs)
            ok_num, metrics = check_verify(case, out_raw, data_dir)
            ok_haz, _ = check_hazard(easm)
            ok = ok_num and ok_haz
            all_ok = all_ok and ok
            if case["op"] in ACT_OPS:
                # ACT 是一元算子，cosine 不适用，看 max ulp
                cos_s, metric_s = "-", f"{metrics.get('ulp', '?')} ulp"
            else:
                cos_s = f"{float(metrics['cosine']):.6f}"
                metric_s = f"{float(metrics['err']):.3e}"
            print(f"{name:<20} {cos_s:>9} {metric_s:>12} {str(cycle):>7} "
                  f"{'ok' if ok_haz else 'HZ!':>6}  {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            all_ok = False
            print(f"{name:<20} {'-':>9} {'-':>12} {'-':>7} {'-':>6}  ERROR: {e}")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
