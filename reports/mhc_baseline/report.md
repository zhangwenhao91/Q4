# mHC Post / shifted Pre 基线结果

采集目录：`results/mhc-baseline/20261030-170916-all-3754372`。
证据：`91136fb:error.txt`，为服务器离线汇总输出，不是本机测量。
固定 hidden=5120、hc_mult=4；其他语义与 operators/spec_v1/cases.py 一致。

## 完整性和精度

26份 correctness.json、12份计时摘要、8个 profile 目录，符合 all 模式预期。
195项输出检查全部通过。N=512有三个种子，N=8192仅本轮默认种子，
不把本轮 smoke 精度检查解释为模型级精度认证。
每组 profile 恰好一份 kernel_details.csv，所有 Duration 可解析。
Post 每侧30条；TL shifted 每侧150条（5 kernel/API）；
AC组合每侧180条（6 kernel/API，包含2次cast）。
ACL→NPU flow 解析曾报错，但设备CSV计数符合预期；本报告不依赖CPU流关联。

## API计时

单位微秒；三个进程各自轮次中位数的中位数。加速比为两侧汇总耗时之比 AC/TL。

| 范围 | N | TL event | AC event | TL wall | AC wall | AC/TL event |
|---|---:|---:|---:|---:|---:|---:|
| Post | 512 | 19.511 | 20.669 | 20.002 | 21.100 | 1.059 |
| Post | 8192 | 500.750 | 524.888 | 501.311 | 525.356 | 1.048 |
| shifted Pre | 512 | 83.986 | 94.127 | 84.712 | 94.580 | 1.121 |
| shifted Pre | 8192 | 647.023 | 3120.484 | 647.890 | 3121.467 | 4.823 |

三个进程的每项比较均为TL更快。Post为原生双侧比较；shifted Pre的AC侧为
PreV2+Sinkhorn+PyTorch旧pre apply组合，含额外未使用h_in和缓存工作，
不能将4.823倍解释为TL对等价纯AscendC融合实现的优势。

## 设备kernel

单位微秒，单次profile各30次调用的平均Duration，不含Wait Time。

| Post N | TL | AC | AC/TL |
|---|---:|---:|---:|
| 512 | 12.666 | 20.632 | 1.629 |
| 8192 | 499.855 | 525.037 | 1.050 |

小N时TL API约19.5us、设备kernel约12.7us，提示调用开销或发射间隙值得检查；
数据来自不同计时模式，不能直接相减后当作精确的主机开销测量。

| TL shifted阶段 | N=512 | N=8192 | 大N设备工作量占比 |
|---|---:|---:|---:|
| apply | 6.979 | 280.808 | 43.37% |
| 投影+平方和 | 16.336 | 270.140 | 41.73% |
| reduce/RMS | 2.788 | 8.214 | 1.27% |
| split | 3.010 | 4.554 | 0.70% |
| Sinkhorn | 8.660 | 83.681 | 12.93% |
| Duration合计/API | 37.773 | 647.397 | 100% |

合计是设备工作量；没有时间轴时不将其一律解释为关键路径延迟。
小N的API约84us，kernel合计约38us，值得优先尝试减少入口数和分配的融合尾部；
不能将差值全部归因于Python。大N的apply+投影占设备工作量约85.1%，
仅移除reduce/split不能带来大的设备加速。

AC组合大N中两次cast+mul+sum合计约2411.219us/API，占其设备工作量约77.2%；
原生PreV2约667.044us、Sinkhorn约46.386us。
TL Sinkhorn大N约83.681us，较AC独立Sinkhorn慢约1.80倍；
这是待分析的性能差距，不据此直接判定布局、指令或流水线的具体原因。

## 下一步

1. 保留本轮基线，实现独立shifted系数尾部：RMS+split+20轮Sinkhorn，直接输出
   next_pre/post/comb，旧prev_pre对应apply仍独立。由5个kernel/API降为3个，
   预期收益必须通过同进程新旧候选和AC组合实测。
2. 融合时重点核查4x4 Sinkhorn的token批量、SIMD布局、循环及中间UB读写；
   不仅减少启动次数，也需检查是否改善大N Sinkhorn吞吐。
3. 若融合后大N仍由apply/投影主导，再分开调apply的hidden tile/缓冲和
   投影内部split-K/分块。更深的apply+投影融合涉及跨流归约和同步，放在后续。
4. Post已超过当前AC基线，后续调优需继续保留原始TL、候选、AC三侧对照。

此报告没有新增设备实验，也没有宣称融合候选已经实现或获得收益。
