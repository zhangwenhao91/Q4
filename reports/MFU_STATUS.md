# MFU 测试准备与历史吞吐换算

更新：本页是首次准备时的历史换算；2026-11-02 新实测及峰值核对后的百分比见
[本轮 MFU 报告](mfu_20261102/report.md)。

以下仅对已有常规 API 延迟做离线计算，**不是新一轮 NPU 测量，也不是 MFU 百分比**。
当前没有已确认的 Ascend950PR_9579 可见 28 AIC 稠密 BF16/HF32 峰值，百分比暂缺。
计数为有用矩阵 FLOPs，完整算子调用延迟包含其余向量操作、调用与分配开销。

| 算子 | N | 实现 | 历史 event us | 有效矩阵 TFLOPS |
|---|---:|---|---:|---:|
| MHA BF16 | 4096 | tilelang_q128 | 52.743690 | 162.861844 |
| MHA BF16 | 4096 | tilelang_kv2 | 51.145120 | 167.952185 |
| MHA BF16 | 4096 | tilelang_s2buf | 51.020999 | 168.360768 |
| MHA BF16 | 4096 | ascendc | 40.092840 | 214.251087 |
| mHC shifted H5120 HF32 | 64 | tilelang_coeff_bound | 25.725715 | 2.445590 |
| mHC shifted H5120 HF32 | 128 | tilelang_coeff_bound | 24.718615 | 5.090460 |
| mHC shifted H5120 HF32 | 256 | tilelang_coeff_bound | 25.904526 | 9.714837 |
| mHC shifted H5120 HF32 | 512 | tilelang_coeff_bound | 28.638594 | 17.574762 |
| mHC shifted H5120 HF32 | 8192 | tilelang_coeff_bound | 586.431046 | 13.732328 |

依据：[MHA 同轮摘要](mha_stats_split/summary.json)、[mHC 三 kernel 摘要](mhc_split/summary.json)。
MHA 分子 `4*S*S*D`；mHC 分子 `2*N*20480*24`。不计 padding 或内部重算。
MHA 不含 QKV/输出投影；mHC 为旧 pre shifted 三 kernel 预绑定，不是新 pre 上游融合版。

## 新测试入口

已增加 [测量与离线分析说明](../operators/mfu/README.md)。服务器执行：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
bash operators/mfu/run.sh all
```

每个测点三进程；完整精度检查和逐轮数据均保留。MHA 含原始 TL、KV2、S2、AC 对照，
mHC 测 N=64/128/256/512/8192 的 tilelang_coeff_bound。当前无峰值时输出有效 TFLOPS、MFU 留空。
不需要重编译已安装原生算子，所需 vendor 缺失时脚本停止并要求恢复。
将最终终端表格或失败日志更新 error.txt，并保留 MFU OUTPUT 目录。
确认各模式峰值与来源后，可用 analyze.py --peak-json 离线补算，不必重跑。
