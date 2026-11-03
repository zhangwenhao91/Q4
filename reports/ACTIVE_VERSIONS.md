# 当前版本索引与清理记录

2026-11-03：当前仅推进[MHA的AC/TL与MFU](MHA_FOCUS.md)，mHC后续工作挂起。

本次清理基于已有测试结果；没有更改保留候选的数值 kernel，也没有新增 NPU 测试。
清理前代码保存在 Git 提交 `b53f66a`；历史报告和原始结果摘要保留。

## 优先使用

| 场景 | 实现/入口 | 边界 |
|---|---|---|
| MHA BF16 基线 | `operators/mha/run_bf16.sh` | 原始 TL 与原生 AC，metadata 两种模式 |
| MHA 当前候选 | `operators/mha/tuning/run_stats_split.sh` | S2 与 KV2 同轮保留，S2 收益尚不稳定 |
| MHA 流水线诊断 | `operators/mha/tuning/run_timeline.sh` | 正在定位优化机会，结果导出后离线分析 |
| mHC shifted 5120 | `operators/mhc/tuning/run_split.sh` | 两/三 kernel 各有优势，保留两者 |
| mHC 新 pre 4096 | `operators/mhc/tuning/run_original.sh` | 上游预绑定与原版融合 AC，同主输出 |
| mHC 新 pre 5120 | `operators/mhc/tuning/run_big5120.sh` | AC 额外缓存输出需单独注明 |
| mHC Post | `operators/spec_v1/bench.py --suite mhc_post` | 同规格原生对照 |
| MHA / 三 kernel mHC 吞吐与 MFU | `operators/mfu/run.sh` | 有效矩阵 TFLOPS，28 AIC 峰值配置已提供，可离线补算 MFU |

其他保留项：原始基线、投影兼容补丁、固定源码/许可与哈希校验、已采用优化的消融对照、
组合缓冲实验的负对照、精度/调用合约测试、构建恢复与采集工具。
反向目录仍按此前请求挂起，未因未完成而删除。

## 删除的 MHA 独立实验

| 版本 | 删除依据 |
|---|---|
| Q64 | API 约增加 23.8%，设备侧同样明显退化 |
| FMA | 没有有效 API 收益 |
| manual schedule | API 约增加 4.1% |
| rowfused | API 约增加 17.7% |
| rowbatch | API 约增加 6.2% |
| Q resident | 无稳定收益 |
| firstPV | 比对应 KV2 对照慢约 1.9% |
| P buffer | 比对应 KV2 对照慢约 4.5% |
| Q/O L2 提示 | 已测候选均未改善 |

删除各版本的独立 transform、suite wrapper、启动脚本、专属测试、过期操作说明，
并移除 bench.py 的 suite 注册、分派和结果比值分支。
rowfused 专用三侧 codegen 导出器也删除；当前时间轴采集仍导出在用候选源码。

## mHC 清理边界

移除旧 `run_fusion.sh`、`run_bound.sh` 及两个独立 suite，统一由 `run_split.sh` 比较
当前两/三 kernel 预绑定候选。对应操作文档同步更新。
`mhc_fusion.py`、`mhc_bound.py` 是共享实现依赖，仍被当前候选使用，保留。
五 kernel 基线、原生 AC 适配对照、新 pre 三 kernel、普通/预绑定比较仍有独立对照意义。

## 历史结果

`reports/mha_*/` 和 `reports/mhc_*/` 的数据属于研究记录。清理代码不删除测量证据，
也不将未采用版本的历史命令当作当前可执行命令。需要复现已删实验时，可在独立 checkout
使用清理前提交 `b53f66a`，不要覆盖当前实验目录。
