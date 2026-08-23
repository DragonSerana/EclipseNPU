import os

import lit.formats

config.name = "EclipseNPU"
config.test_format = lit.formats.ShTest(False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(os.path.abspath(__file__))
config.test_exec_root = os.path.join(
    os.path.dirname(os.path.dirname(config.test_source_root)),
    "build", "tests", "lit")
config.parallel = False

repo_root = os.path.dirname(os.path.dirname(config.test_source_root))

config.substitutions.append(
    ("%eclipse-opt",
     os.environ.get("ECLIPSE_OPT",
                    os.path.join(repo_root, "build", "bin", "eclipse-opt"))))
config.substitutions.append(
    ("%mlir-opt",
     os.environ.get("MLIR_OPT",
                    "/home/serana/mlir/llvm-project/install/bin/mlir-opt")))
config.substitutions.append(
    ("%FileCheck",
     os.environ.get("FILECHECK",
                    "/home/serana/mlir/llvm-project/build/bin/FileCheck")))
config.substitutions.append(
    ("%not",
     os.environ.get("NOT",
                    "/home/serana/mlir/llvm-project/build/bin/not")))
