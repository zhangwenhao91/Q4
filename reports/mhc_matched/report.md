# 对齐 shifted 语义的 mHC 设备结果

证据：`46dc587:error.txt`，服务器profile运行目录
`results/mhc-matched/20261030-175736-profile-3862789`。
这是既有设备结果的离线分析，没有新增设备实验。

## 已确认的范围

N=512/8192、hidden=5120，Post与shifted Pre双方及互比共60项输出检查全部通过。
每组一份设备CSV，Duration均有效；Post各30条，TL Pre各150条（5 kernel/API），
AscendC Pre各60条（2 kernel/API）。
新AscendC路径实际出现MhcShiftedPreV2和MhcSinkhorn，没有旧PyTorch apply的cast/mul/sum。
ACL→NPU flow解析仍有报错；本报告只依赖计数完整的设备Duration，不使用CPU流关联。

## 设备耗时

单位微秒，Post为单kernel平均Duration，Pre为每次API对应kernel平均Duration合计。
合计表示设备工作量，不等于存在重叠或发射间隙时的API延迟。每组为一次profile的30次调用。

| 算子 | N | TL | 适配AscendC | AC/TL |
|---|---:|---:|---:|---:|
| Post | 512 | 12.798 | 20.669 | 1.615 |
| Post | 8192 | 499.617 | 525.132 | 1.051 |
| shifted Pre | 512 | 37.579 | 40.865 | 1.087 |
| shifted Pre | 8192 | 648.984 | 703.056 | 1.083 |

不再使用旧PyTorch组合路径的4.823倍作为新对手的结论。

| TL阶段 | N=512 | N=8192 |
|---|---:|---:|
| apply | 6.947 | 280.686 |
| 投影/平方和 | 16.329 | 272.042 |
| reduce/RMS | 2.777 | 8.214 |
| split | 2.914 | 4.533 |
| Sinkhorn | 8.612 | 83.509 |

AC原生shifted Pre约32.176/656.316us，独立Sinkhorn约8.690/46.740us。
大N时TL apply+投影约552.728us，优于AC融合Pre的656.316us，但TL Sinkhorn较慢。
不能将这些阶段差异全部归因于语言或单一管线。
下一候选仍为RMS+split+Sinkhorn融合，以降低5个入口的调用开销并检查Sinkhorn吞吐。

## 仍缺的证据

本次日志只有profile模式，API summary数量为0，不能据此给出新对手的API加速比。
也未包含check模式的动态N、三种子和特殊旧pre结果，不能宣称完整动态N已验收。
可用 `python operators/mhc/matched/collect_results.py` 离线汇总服务器已有check/bench/profile目录，
不需要重新执行算子；脚本分别报告各次运行和源码/库哈希，不把不同运行混成一次测量。
真实模型权重尚未由本次日志证明已加载。
