# MHA 与三 kernel 预绑定 mHC：MFU 结果

证据：`9fe3f55:error.txt`。本次服务器输出了完整的九行汇总，本机只做峰值核对与离线归一化，未新增设备测量。
用户粘贴的是最终表格，没有 plan、metadata、各进程计时和 MFU OUTPUT 路径；
当前脚本生成最终表格前会校验精度、三进程及逐轮完整性，但本机尚未独立重验这些原始文件。

## 分母与公式

已核对新增 WIKI `MFU计算.md`，并从昇腾产品页进入官方《昇腾950 NPU架构白皮书》表3-1，
确认 950PR 的 32/28 AIC、64/56 AIV 规格，以及对应 Cube BF16 432/378、TF32 216/189 TFLOPS。
本设备此前查询为 Ascend950PR_9579、28 AIC/56 AIV，采用 BF16 **378**、TF32 **189 TFLOPS**。
不使用 Cube+Vector 总峰值 425/212，也不使用 32 AIC 峰值。
官方表没有名为 HF32 的一行；本项目显式开启 HF32 的 tf32_hc_prenorm_gemm 投影按 TF32 Cube 档归一化，
以下 mHC 百分比以该软件/硬件模式对应关系为前提。它不是 IEEE FP32 标量峰值，也不是 BF16 峰值。

- MHA：`4*S*S*D = 8,589,934,592 FLOPs`，包含 dense QK 和 PV，非因果，无需按三角减半。
- mHC：`2*N*20480*24 = 983,040*N FLOPs`，旧 pre shifted H5120 投影。
- `MFU(%) = effective_TFLOPS / Cube_peak_TFLOPS * 100`。
- 分母使用完整 API event 延迟；softmax/apply/RMS/平方和/Sinkhorn 等耗时在内，矩阵分子不计这些操作。
- 这是算子级有效矩阵 MFU，不等于完整模型 MFU、单 projection kernel 利用率或硬件忙碌比例。

## 实测结果与离线 MFU

| 算子 | N | 实现 | API event us | 有效 TFLOPS | Cube 峰值 | MFU |
|---|---:|---|---:|---:|---:|---:|
| MHA BF16 | 4096 | tilelang_q128 | 52.679100 | 163.061529 | 378 | 43.138% |
| MHA BF16 | 4096 | tilelang_kv2 | 51.089995 | 168.133401 | 378 | 44.480% |
| MHA BF16 | 4096 | tilelang_s2buf | 50.964119 | 168.548671 | 378 | 44.590% |
| MHA BF16 | 4096 | ascendc | 40.085506 | 214.290285 | 378 | 56.691% |
| mHC 三 kernel HF32/TF32 | 64 | tilelang_coeff_bound | 26.715851 | 2.354953 | 189 | 1.246% |
| mHC 三 kernel HF32/TF32 | 128 | tilelang_coeff_bound | 25.961220 | 4.846811 | 189 | 2.564% |
| mHC 三 kernel HF32/TF32 | 256 | tilelang_coeff_bound | 25.657110 | 9.808518 | 189 | 5.190% |
| mHC 三 kernel HF32/TF32 | 512 | tilelang_coeff_bound | 39.343319 | 12.792934 | 189 | 6.769% |
| mHC 三 kernel HF32/TF32 | 8192 | tilelang_coeff_bound | 585.551186 | 13.752963 | 189 | 7.277% |

## 解读与 WIKI 使用边界

MHA S2、KV2、原始 TL 的 MFU 依次约 44.59%、44.48%、43.14%，原生 AC 约 56.69%。
S2 相对 KV2 的百分点差仍很小，不能仅凭汇总值声称稳定优化。
mHC 的分子是窄投影 K=20480、输出通道24 的有用矩阵工作，时间却含三 kernel、向量尾部及分配；
低 MFU 不表示 92%～99% 时间空闲，不能直接据此推导可实现加速比。
N=512 本轮39.343319us，较历史三kernel汇总28.638594us增加约37.4%；其他样例变化较小。
未取得本轮各进程/轮次，不能确定主机开销、设备竞争或频率等具体原因，保留原结果，不挑历史最短值替代。

WIKI 的有效 FLOPs、只用 Cube 峰值、不同 dtype/设备范围不混用，与当前脚本口径一致。
有两处需要限定：MFU 通用定义不必只含矩阵乘，当前是双方约定的矩阵口径；
掩码后的实际执行 FLOPs 也不一定是全量矩形2MNK，FlashAttention可能跳过整块。
仅在算法确实执行全部无效位置时，执行/有效比才接近2，不能把全量2MNK自动当硬件实际执行量。
本次 MHA 无mask，这些掩码区别不影响当前结果。

## 后续操作

无需重新计时。峰值配置已提交 `operators/mfu/peaks/ascend950pr_28aic.json`。
服务器若要为原结果生成带百分比的 mfu.json/mfu.md，可离线执行：

```bash
git pull --no-rebase
python operators/mfu/analyze.py --input results/mfu/实际目录 \
  --peak-json operators/mfu/peaks/ascend950pr_28aic.json
```

`实际目录` 使用该次运行打印的 MFU OUTPUT 路径，不要替换成旧 timeline 目录。
若继续分析 N=512 波动，优先传回此目录中的 plan.json、metadata.json、correctness.json、timings.csv、summary.json 和 mfu.json，
再决定是否需要复测。

来源：[用户 WIKI](../../MFU计算.md)、[官方产品页](https://www.hiascend.com/hardware/processor)、
[官方白皮书，表3-1](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)。
