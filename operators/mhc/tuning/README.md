# mHC 当前版本与运行入口

| 规格 | 入口 | 保留原因 |
|---|---|---|
| shifted 旧 pre，hidden=5120 | `run_split.sh` | 同轮比较两 kernel、三 kernel 预绑定与适配 AC |
| 本次新 pre，hidden=4096 | `run_original.sh` | 原版融合 AC、上游普通/预绑定 TL 与三 kernel TL |
| 本次新 pre，hidden=5120 | `run_big5120.sh` | 上游普通/预绑定 TL 与原版 PreV2 + Sinkhorn |

每个脚本支持 `smoke`；详细设置见 [SPLIT.md](SPLIT.md)、[ORIGINAL.md](ORIGINAL.md)、[BIG5120.md](BIG5120.md)。

shifted 小 N 三 kernel 更合适，大 N 两 kernel 更合适，不能删除其中一条。
新 pre 与旧 pre 语义、输出不同，不能互相替代。
普通调用与预绑定用于区分调用开销与数值 kernel 性能，仍有对照价值。

`apply_tail.py`、`coeff_tail.py`、`source.json` 及许可证为当前数值实现与来源记录。
`operators/spec_v1/mhc_fusion.py` 提供共享分配/候选构造，`mhc_bound.py` 提供当前预绑定实现；
两者仍是 `run_split.sh` 和新 pre 路径的依赖，不是可以直接删除的旧版本。

已删除被 `run_split.sh` 覆盖的 `run_fusion.sh`、`run_bound.sh` 独立入口、
对应 suite 注册和旧 BOUND 操作说明；历史结果仍保留在 reports/mhc_fusion 和 mhc_bound。
已验证 token 尾块、新旧 pre 语义、投影及分配合约的测试继续保留。

参见 [当前版本索引](../../../reports/ACTIVE_VERSIONS.md) 和 [结果总览](../../../reports/TEST_SUMMARY.md)。
