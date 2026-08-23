# CI 版本 pin 决策

## 背景

本地 MLIR 是源码构建的，不是 apt 装的。CI 如果换一套 MLIR，TableGen、pass 名、生成代码都可能不一样，本地绿了 CI 不一定绿。所以先把版本钉死。

## 固定的版本

- LLVM/MLIR 源码：`/home/serana/mlir/llvm-project`
- commit：`a67efda258fa73c7b6b915fb31b8412b800a15e9`
- install 路径：`/home/serana/mlir/llvm-project/install`
- FileCheck：`/home/serana/mlir/llvm-project/build/bin/FileCheck`
- clang-format：`/home/serana/mlir/llvm-project/install/bin/clang-format`
- lit：`~/.local/bin/lit`

## 决策

这里先写结论，理由一句话带过。

- CI 构建方式：待定（Docker 镜像 / ccache / 自托管 runner 三选一）
- 镜像或缓存怎么保存：待定
- MLIR 是否每次重新构建：待定
- 升级策略：不追上游；只有显式改 pin 时才换版本

## 需要记录的理由

- 构建时间：本地全量构建大概多久，CI 能不能接受
- 可复现性：Docker 更容易复现，但镜像维护成本高
- 成本：自托管跑在现有机器上，但要自己管环境

## 待办

- [ ] 在 CI 里跑一次 pin 后的构建，确认和本地一致
- [ ] 把 install / FileCheck / clang-format 路径写进 CI 配置
- [ ] 把上面的“待定”补成实际选择
