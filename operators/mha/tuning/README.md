# MHA 保留版本与运行入口

当前规格为 B1/H1/S4096/D128、BF16 输入输出。

- 当前候选：`stats_split_variant.py` 中的 `tilelang_s2buf`，同时保留 `tilelang_kv2` 对照。
- 原始基线与原生 AscendC：`operators/mha/run_bf16.sh`。
- 常规验证/计时：`run_stats_split.sh check`、`run_stats_split.sh bench`。
- 流水线采集：`run_timeline.sh pipeline`；数据导出：`export_timeline.py`。
- 恢复缺失环境：`../recover_reclone.sh`、`../env_recovered.sh`。

S2 相对 KV2 的微小收益尚不稳定；不要只保留 S2。

## 依赖与消融对照

`nostore_variant.py` → `finalize_variant.py` → `buffers_variant.py` → `stats_split_variant.py`
是当前候选的实现链；`stages_variant.py` 及对应入口用于已采用优化的消融对照。
这些模块、原始源码校验、CPU 合约测试及 profiling 工具继续保留。
alpha 双版本等负对照用于验证缓冲生命周期退化，保留在组合实验中，默认不作为最优候选。

## 已清理的实验

Q64、FMA、手工阶段调度、rowfused、rowbatch、Q 驻留 L0A、firstPV、P 双版本、L2 提示：
删除独立实现、suite、启动脚本、专属测试和过期操作说明。
删除只服务于 rowfused 对照的 `export_codegen.py`。
历史性能数据继续保存在 `reports/mha_*/`，具体代码可从清理前提交
`b53f66a` 查看；这些报告描述历史实验，不再提供当前分支可运行的入口。

完整保留/删除依据见 [版本索引](../../../reports/ACTIVE_VERSIONS.md)。
