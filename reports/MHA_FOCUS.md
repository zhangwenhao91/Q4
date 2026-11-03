# 当前工作：仅比较 MHA 的 AC/TL 与 MFU

最新准备：[S3584全部保留实现路径对照](MHA_ALL_SCENARIOS.md)，同轮测7条TL路径与原生AC，
序列固定为原来的7/8；执行命令见该页。

用户于2026-11-03明确将mHC后续工作挂起。以下只分析MHA；历史mHC数据保留。

## 当前同规格结果

证据：a55bb09:error.txt，`results/mfu/20261103-115353-28394`。
B1/H1/D128，BF16输入/输出，dense非因果，AscendC复用metadata，
包含完整API调用和输出/临时分配；不含QKV与输出投影。
AC/TL为同序列长度下的API event延迟之比，**大于1表示TL更快**。
MFU使用有用QK+PV矩阵FLOPs与28 AIC Cube BF16峰值378 TFLOPS。

| S | TL版本 | TL us | AC us | AC/TL | TL MFU | AC MFU |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | 原始 | 52.664 | 40.088 | 0.761 | 43.15% | 56.69% |
| 4096 | KV2 | 51.097 | 40.088 | 0.785 | 44.47% | 56.69% |
| 4096 | S2 | 50.957 | 40.088 | **0.787** | **44.60%** | 56.69% |
| 3584 | 原始 | 24.652 | 26.778 | 1.086 | 70.58% | 64.97% |
| 3584 | KV2 | 24.022 | 26.778 | **1.115** | **72.43%** | 64.97% |
| 3584 | S2 | 24.251 | 26.778 | 1.104 | 71.74% | 64.97% |

当前汇总S4096以S2最快，S3584以KV2最快，保留两者以免过早固定配置。
S3584的KV2领先AC约1.115倍，但这不能替代原规格S4096的结论。
跨shape的延迟比不作为同工作量AC/TL加速比。

## 下一步：先核对已有MHA六个进程，不重测

导出工具现在支持 `--operator mha`，只校验并复制原测试中两种S、各三进程的证据。
不需要mHC结果文件，也不执行任何NPU操作。导出的plan只含MHA，
保留原plan哈希；导出后可在本机独立重算所有指标。

```bash
cd /home/z30086261/Q4
git pull --no-rebase
python operators/mfu/export.py \
  --input results/mfu/20261103-115353-28394 \
  --output reports/mfu_20261103/mha_capture \
  --operator mha
git add reports/mfu_20261103/mha_capture
git commit -m "Add MHA-only timing evidence"
git push origin main
```

取得原始timings、metadata和correctness后，优先核对：

1. 每个进程的同规格AC/TL是否均大于1，以及各轮/进程分布。
2. S3584的KV2相对S2排序、source/native库哈希和运行版本。
3. event/wall结论是否一致，避免只看聚合中位数隐藏波动。

分析工具已增加 `mha_comparisons`，包含同规格聚合AC/TL、wall比、三进程配对比及胜场数。
聚合比值使用双方各自三进程中位数之比，配对进程单列；两者不混为一个统计量。
当前只有粘贴汇总，尚不能给出三进程胜场。

后续只有必要时复测：`bash operators/mfu/run.sh mha both`。
现在不传参数的默认入口也是 `mha both`，仅检查q4fa vendor，不测mHC。

## 后续实验判读

若稳定复核通过，可围绕S3584的KV2继续优化，并保留S4096基线。
若研究28任务对齐的机制，下一步应固定KV长度、只改Q长度，隔离KV循环/计算量的变化；
该定向实验当前尚未实现或执行。
