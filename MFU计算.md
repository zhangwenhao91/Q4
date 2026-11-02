# 昇腾 NPU 上的 MFU 计算方法

> 从 2MNK 到掩码场景的有效计算量

适用平台：昇腾 NPU（以 Ascend 950PR 为例） · 2026 年 10 月

MFU（Model FLOPs Utilization，模型算力利用率）衡量模型在给定硬件上把峰值算力用出了多少：分子是模型真正需要的浮点运算量，分母是运行时间与硬件峰值的乘积。它只反映矩阵乘这类浮点运算的效率，并不等于硬件利用率——访存、通信、调度占用的时间都计在分母里，却不会增加分子。

下面先给出无掩码 GEMM 的标准算法，再说明掩码场景为什么必须按有效计算量统计。

> **核心公式**
> 
> **MFU = 有效浮点运算量 ÷ 运行时间 ÷ 硬件峰值算力**
> 
> 无掩码 GEMM 的运算量是 2MNK；有掩码时只统计掩码保留的有效位置。

## 1  MFU 的定义

**MFU = FLOPs<sub>model</sub> / (T × P<sub>peak</sub>)**

- FLOPs<sub>model</sub>：一次迭代（或一个算子）的浮点运算次数，单位 FLOP。
- T：与之对应的运行时间，单位 s；取预热后稳定区间的中位数，剔除编译与数据加载。
- P<sub>peak</sub>：硬件峰值算力，单位 FLOP/s。手头若是 TFLOPS，先换算：P<sub>peak</sub> = TFLOPS × 10<sup>12</sup>。

换成百分比形式：

**MFU (%) = FLOPs<sub>model</sub> / (T × TFLOPS × 10<sup>12</sup>) × 100%**

分子与分母必须同口径：同一数据类型（FP16 / BF16 / FP32 / INT8）、同一稀疏模式（dense / sparse）、同一设备范围（单卡 / 整机）。昇腾不同型号、不同 dtype 的峰值差别很大，口径混用会得到没有意义的数字。

T 与 FLOPs 还必须对应同一范围：单算子对单算子，整步对整步。

## 2  硬件峰值算力

P<sub>peak</sub> 取自官方规格或下表。Ascend 950PR 的峰值如下，按启用的 AIC 数量分为 32 / 28 两档。

**表 1  Ascend 950PR 技术规格**

**核心数量**

| 项目 | 32 AIC | 28 AIC |
| --- | ---: | ---: |
| Cube Core 数量 | 32 | 28 |
| Vector Core 数量 | 64 | 56 |

**Cube + Vector 总算力**

| 数据类型 | 32 AIC | 28 AIC |
| --- | ---: | ---: |
| MXFP4（TFLOPS） | 1784 | 1561 |
| HiF8/MXFP8/FP8（TFLOPS） | 919 | 804 |
| INT8（TOPS） | 919 | 804 |
| BF16/FP16（TFLOPS） | 486 | 425 |
| TF32（TFLOPS） | 243 | 212 |

**Cube 算力**

| 数据类型 | 32 AIC | 28 AIC |
| --- | ---: | ---: |
| MXFP4（TFLOPS） | 1730 | 1513 |
| HiF8/MXFP8/FP8（TFLOPS） | 865 | 756 |
| INT8（TOPS） | 865 | 756 |
| BF16/FP16（TFLOPS） | 432 | 378 |
| TF32（TFLOPS） | 216 | 189 |

**Vector 算力**

| 数据类型 | 32 AIC | 28 AIC |
| --- | ---: | ---: |
| FP16/BF16（TFLOPS） | 54 | 47 |
| FP32（TFLOPS） | 27 | 23 |
| INT8（TOPS） | 54 | 47 |
| INT16（TOPS） | 27 | 23 |
| INT32（TOPS） | 13 | 11 |
| INT64（TOPS） | 6 | 5 |

数据来源：[昇腾官网](https://www.hiascend.com/hardware/processor)（2026 年 10 月查询）。浮点算力单位 TFLOPS，整数算力单位 TOPS。

> 注：32 AIC / 28 AIC 指该配置启用的 AI Core（Cube Core）数量；官网仅以 32/28 并列，未标注列名。PR 与 DT 是另一组产品线：PR 最高 32 AIC / 128GB / 1.6TB/s，DT 最高 36 AIC / 144GB / 4TB/s，两者同属 950 代、灵衢架构一致。

用于 MFU 时，分母只取 Cube 算力中与实际执行 dtype 对应的一行：FP16 输入配 FP32 累加仍按 FP16 计；Cube + Vector 总算力把不参与矩阵乘的 Vector 也加了进来，会使分母偏高约 12.5%，MFU 相应偏低；28 AIC 配置的峰值约为 32 AIC 的 87.5%，应以实际机器读到的规格为准。整机 MFU 时，P<sub>peak</sub> 为各卡峰值之和。

## 3  无掩码矩阵乘：FLOPs = 2MNK

最基本的算子是 GEMM：C[M×N] = A[M×K] × B[K×N]。

**FLOPs = 2 × M × N × K**

输出 C 共有 M×N 个元素，每个元素由 K 次乘加得到；一次乘加包含 1 次乘法与 1 次加法，合计 2 FLOP。因此总乘加次数为 MNK，总运算量为 2MNK。

系数 2 来自“一乘一加各计一次”。若某个工具把一次 FMA 记作 1 次运算，它给出的是 MNK，此时峰值也要按同一口径折算，否则 MFU 会相差 2 倍。昇腾规格与业界惯例通常按 1 FMA = 2 FLOP 计。

**MFU = 2MNK / (T × P<sub>peak</sub>)**

示例：M = N = K = 4096 的 FP16 GEMM，若 T = 1.0 ms、P<sub>peak</sub> = 432 TFLOPS（950PR 32 AIC 的 FP16 Cube 峰值），则 FLOPs ≈ 1.37 × 10<sup>11</sup>，MFU ≈ 32%。

多卡时，若报整机 MFU，分子取各卡 FLOPs 之和、分母取各卡峰值之和；若报单卡，则只取该卡承担的 FLOPs 与其峰值。PP 的流水空泡计入 T，但不增加 FLOPs。

统计时只计矩阵乘：LayerNorm、Softmax、通信不计入分子，但其耗时留在 T 中；受 cube 利用率、tile 形状与访存限制，实测 MFU 通常明显低于 100%。

## 4  有掩码：按有效计算量统计

掩码会把部分输出位置置为无效：causal attention 上三角、变长序列 padding、块稀疏掩码都属此类。硬件仍按完整 tile 计算，但语义上只有保留位置参与结果，因此分子应改用有效计算量：

**FLOPs<sub>eff</sub> = 2 × K × N<sub>valid</sub>**

N<sub>valid</sub> 为有效输出元素个数。逐行统计时 N<sub>valid</sub> = Σ n<sub>i</sub>，n<sub>i</sub> 是第 i 行的有效元素数；按块统计时 FLOPs<sub>eff</sub> = Σ 2 M<sub>b</sub>N<sub>b</sub>K<sub>b</sub>。代回定义：

**MFU<sub>eff</sub> = FLOPs<sub>eff</sub> / (T × P<sub>peak</sub>)**

以 causal self-attention 为例：Q、K 均为 S×d，score 矩阵为 S×S，仅下三角（含对角线）有效；有效元素数为 S(S+1)/2，占全量的 (S+1)/(2S)，S 较大时约为 1/2。于是 QK<sup>T</sup> 的有效运算量为

**FLOPs<sub>QK</sub> = 2 S<sup>2</sup> d × (S+1)/(2S) = d·S(S+1)**

P·V 同理，也约为全量的一半；直接套用 2MNK 会把 attention 高估近 2 倍。

- 有效 MFU：分子为 FLOPs<sub>eff</sub>，衡量掩码之后真正有用的计算效率，也是本文推荐的口径。
- 执行 MFU（硬件利用率）：分子为全量 2MNK，衡量芯片实际执行了多少运算。
- 两者之比即掩码带来的算力放大倍数，causal attention 约为 2∶1；报告时注明口径即可。

## 5  速查表

**表 2  常见场景的运算量**

| 场景 | 运算量（FLOPs） | 说明 |
| --- | --- | --- |
| 无掩码 GEMM，M×N×K | 2MNK | 一次乘加计 2 FLOP |
| Causal QK<sup>T</sup>，S×d | d·S(S+1) ≈ S<sup>2</sup>d | 全量 2S<sup>2</sup>d 的一半 |
| Causal PV，S×d | d·S(S+1) ≈ S<sup>2</sup>d | 同上 |
| 任意掩码 | 2K·N<sub>valid</sub> | N<sub>valid</sub> = Σ n<sub>i</sub> |
| 整机（N 卡） | 单卡 FLOPs × N | 峰值同步求和 |

