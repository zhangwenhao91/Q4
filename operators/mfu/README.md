# MHA 与三 kernel 预绑定 mHC 的 MFU 测量

## 当前范围：仅 MHA（2026-11-03）

mHC后续测试挂起。当前结果、同规格AC/TL、MFU及仅导出MHA的命令见
[MHA当前工作页](../../reports/MHA_FOCUS.md)。
默认运行入口已改为 `bash operators/mfu/run.sh mha both`（不传参数效果相同）。
分析新增同规格AC/TL及三进程配对比；导出新增 `--operator mha`，只依赖MHA六个进程文件。
下方联合实验命令保留为历史说明，当前不要求重跑联合实验。

## 28 AIC：7/8 规模复测（2026-11-03）

本轮已经完成，见 [结果报告](../../reports/mfu_20261103/report.md)。
后续优先用 `export.py` 导出现有逐进程证据，命令见报告，无需立即重测。

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
bash operators/mfu/run.sh all both
```

`both` 同一轮测原规模和 7/8 规模；每个规格仍是三个独立进程，第二轮反转形状执行顺序。
共 36 个计时子进程，逐例先验精度，失败即停。已有 AscendC vendor 不需重建；
3584 的 TL kernel 首次需要 JIT。新规格的设备精度和性能尚待服务器验证。

| 算子 | 原规模 | 7/8 规模 |
|---|---|---|
| MHA 序列 S，B1/H1/D128 | 4096 | 3584 |
| mHC token N，旧 pre shifted H5120 | 64/128/256/512/8192 | 56/112/224/448/7168 |

MHA 同时缩短 Q/K/V，Q tile 和 KV 循环数各从32变28，FLOPs比例49/64。
mHC 只改N，投影FLOPs比例7/8；hidden、24输出通道、数值公式与预绑定三kernel保持原设置。
尾块与投影 padding 不计有用 FLOPs；56 含非16整块，需本轮精度检查通过后才算新规格验证。
28 AIC 的峰值分母仍是 BF16 378 / HF32按TF32档189 TFLOPS，不再次乘7/8。
缩小输入本身不保证提高 MFU，尤其 mHC 仍受 tile 补齐、调用和向量操作影响。

报告除各规格 MFU 外，新增 scaled/full 的 FLOPs、延迟和有效TFLOPS比值。
不同输入的延迟比不是同工作量加速比；只有有效TFLOPS/MFU改善才表明该规模下效率改善。
完整分析边界见 [7/8测试计划](../../reports/MFU_SEVEN_EIGHTHS.md)。

仅测新规模用 `bash operators/mfu/run.sh all seven-eighths`；
只测一种算子用 `bash operators/mfu/run.sh mha both` 或 `mhc both`。
当前默认改为 `mha both`，自动读取已提供的28 AIC峰值配置。

2026-11-02：已对照新增 `MFU计算.md` 和官方白皮书表3-1，添加
`peaks/ascend950pr_28aic.json`：BF16 Cube 378 TFLOPS、TF32 Cube 189 TFLOPS。
本项目 HF32 投影按 TF32 档归一化，配置中记录该映射条件。
这不是 Cube+Vector 总算力，也不按实时频率缩放额定峰值。
本轮结果见 [MFU 报告](../../reports/mfu_20261102/report.md)。

先测有用矩阵 FLOPs / 完整 API event 延迟，输出有效 TFLOPS。
理论峰值尚未确认时，MFU 百分比留空，不使用其他芯片、稀疏峰值或整机峰值替代。
这是算子级矩阵计算口径的 MFU proxy，不是完整模型训练 MFU 或硬件指令利用率。

## 执行

沿用既有 zwh/CANN 环境和已安装 q4fa/q4mhcmatch vendor，无需重编译算子：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
bash operators/mfu/run.sh all
```

也可分别用 `mha` 或 `mhc`。开始前检查所需 vendor；缺失时停止，保留报错并反馈。
运行输出打印 `MFU OUTPUT` 目录和最终表格。将终端输出更新至 error.txt。
每个子进程完整日志、源码/设备信息、精度检查、逐轮时间和摘要保留在输出目录。
子进程失败时停止并打印日志末尾，不把部分样本当作完整结果。

MHA：B1/H1/S4096/D128，BF16 输入/输出，dense/noncausal，cached metadata，
原始 TL、KV2、S2、原生 AC 同进程顺序平衡；三进程，每进程预热 100 次、8 轮各 100 次。
mHC：旧 pre shifted H5120，标签 tilelang_coeff_bound，完整三 kernel 调用；
N=64/128/256/512/8192，每形状三进程、预热 100 次、9 轮各 100 次。
每次重新计算并分配输出/工作区，投影的公开 compile 在计时外完成。
两类 vendor 使用不同进程和库路径。计时前执行既有 CPU 参考精度校验，mHC 另验投影/平方和。

## FLOPs 和峰值口径

乘加按 2 FLOPs：

- MHA：`4*B*H*S*S*D = 8,589,934,592`，包括 QK 和 PV，匹配稠密 BF16 峰值。
- mHC：`2*N*(4*5120)*24 = 983,040*N`，为有用投影矩阵 FLOPs，匹配 HF32 峰值。
  BF16 特征经过当前投影路径参与 HF32 运算，不能因为输入张量是 BF16 就用 BF16 峰值。

`effective_TFLOPS = useful_matrix_FLOPs / event_us / 1e6`。
`matrix_MFU_percent = 100 * effective_TFLOPS / dense_peak_TFLOPS`。

未把 softmax、平方和、RMS、apply、sigmoid、Sinkhorn 等标量/向量操作计入矩阵分子；
它们的执行时间均在分母内。不给 padding、重算或内部循环额外 FLOP credit。
同时保留 wall 延迟及其有效 TFLOPS。设备 Duration/单 kernel 时间不替代完整 API 延迟。
若需要更完整的标量 FLOP 估计，须单独定义 exp/div/sqrt 的计数规则，不混入本矩阵口径。

最终延迟是三个独立进程各自轮次中位数的中位数；记录进程间吞吐 min/max，
不挑最短时间，不用单次 PipeTimeline 的重放时间代替常规计时。

## 确认峰值后离线补算

准备 peak.json，按实际设备信息填写数值；以下字段说明不是可直接使用的 JSON：

| 字段 | 要求 |
|---|---|
| 顶层 bf16 / hf32 | 各自一条峰值记录，可只填已确认的一种 |
| tflops | 正数；当前可见设备/分区的稠密对应模式峰值 |
| source | 规格文档链接、页码或有出处的设备计算说明 |
| scope | 固定字符串 visible_device_dense |
| name | 与本轮 metadata.device_properties.name 完全一致 |
| cube_core_num | 与本轮可见 AIC 数量一致，目前查询为 28 |

```bash
python operators/mfu/analyze.py --input results/mfu/实际目录 --peak-json /path/to/peak.json
```

当前已确认 28 AIC 设备可直接使用已提交配置，无需重新计时：

```bash
python operators/mfu/analyze.py --input results/mfu/实际目录 \
  --peak-json operators/mfu/peaks/ascend950pr_28aic.json
```

保留实际目录名；本轮粘贴的 error.txt 未包含 MFU OUTPUT 路径，不能从表格恢复该路径。

若没有 peak-json，仍可离线重复生成有效 TFLOPS 报告。JSON 中峰值为 null，
不把缺失峰值解释成 0% MFU。MFU 大于 100% 时保留结果用于排查峰值/时钟/计时口径，
不裁剪到 100%。本机无 NPU，真实计时与编译效果需服务器确认。
