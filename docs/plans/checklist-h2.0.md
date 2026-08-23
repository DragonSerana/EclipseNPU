# H2.0 出口闸门

做完下面这些才进 H2.1。每项后面尽量写证据或备注，别只打勾。

## golden 基线

- [x] K 分块循环每轮 MATMUL 后补 SYNC，消掉 WAR
- [x] matmul_golden 检查 argc，readFile 失败会报错退出
- [x] matmul_golden 能输出 `.easm`（默认 `golden.easm`）
- [x] `python3 tests/golden/matmul_check.py` PASS，total = 11536
- [x] `hazard_check.py` 对 `tests/golden.easm` 报 0 违例

## Eclipse dialect

- [x] `EclipseOps.td` 单一来源，在 `compiler/include/` 下
- [x] `eclipse-opt` 已注册 Eclipse dialect
- [x] 六个 leaf op 已定义：dma_load / dma_store / matmul / elementwise_add / act / sync
- [x] act 的 kind 用枚举，不是裸 i32
- [x] verifier 按 op 拆到 `Ops/` 目录
- [x] matmul / elementwise_add / act 的 shape 检查有报错信息

## lit

- [x] `check-eclipse` 目标能跑
- [x] `leaf_ops.mlir` 正例通过
- [x] 7 个负例通过
- [x] `Eclipse-test` 一键跑 lit

## CI / 文档

- [ ] CI pin 决策记录
- [ ] 实际 CI 配置（H2.3 再接，这里先留决策）
- [ ] 本文件在 H2.0 提交时一起入库

## 备注

当前 lit 共 8 个测试：1 个正例 + 7 个负例。
