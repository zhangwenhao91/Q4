# 对齐 shifted 语义的 mHC 精度与性能结果

完整汇总证据：`5f9a12e:error.txt`；最初profile证据：`46dc587:error.txt`。
服务器目录：

- check：`results/mhc-matched/20261030-175049-check-3849101`
- bench：`results/mhc-matched/20261030-175456-bench-3857828`
- profile：`results/mhc-matched/20261030-175736-profile-3862789`

这是既有设备结果的离线分析，没有新增设备实验。

## 已确认的范围

hidden=5120。check/bench/profile分别171/90/60项输出检查，合计321项全部通过；
另有24/12/8项原始投影与平方和检查，合计44项全部通过。
N=512三个种子；动态N=1/17/127/128/129/513；特殊输入包括全零残差、全零旧pre和
只选择第3路残差的旧pre，全部通过。N=8192在bench/profile中用默认种子检查通过。
这是所列动态N样例和smoke门限的通过结果，不代表全部输入/任意N的模型级认证。

三组运行的Pre源码版本、原生库SHA256、HC投影源码SHA256一致：

- Q4：`b8083b14a7721bbf561109be1c2cffd5fef95a33`
- native library：`4307e7ad6acadc9c2bc2746a50fd2dd4eb469235ec428a53e4cb0700018e53af`
- HC source：`984334759325a7f4da7ff4c2a43fc984d2f18bc43706df70139210a0bef59830`

每组一份设备CSV，Duration均有效；Post各30条，TL Pre各150条（5 kernel/API），
AscendC Pre各60条（2 kernel/API）。
新AscendC路径实际出现MhcShiftedPreV2和MhcSinkhorn，没有旧PyTorch apply的cast/mul/sum。
ACL→NPU flow解析仍有报错；本报告只依赖计数完整的设备Duration，不使用CPU流关联。

## 完整API耗时

单位微秒，三个独立进程各自轮次中位数的中位数；每进程预热100次、9轮各100次，
双方在轮次间交替执行先后。加速比为汇总AC event/TL event。

| 算子 | N | TL event | AC event | TL wall | AC wall | AC/TL event |
|---|---:|---:|---:|---:|---:|---:|
| Post | 512 | 19.650 | 20.734 | 20.114 | 21.067 | 1.055 |
| Post | 8192 | 500.394 | 524.603 | 500.964 | 525.146 | 1.048 |
| shifted Pre | 512 | 79.536 | 40.870 | 80.187 | 41.346 | 0.514 |
| shifted Pre | 8192 | 647.634 | 705.278 | 648.522 | 705.905 | 1.089 |

Post两个形状以及大N Pre均为TL更快，小N Pre则是AscendC约快1.946倍。
这四项比较的三个独立进程方向全部一致，wall与event方向也一致。
因此不能依据设备Duration合计较短就宣称TL Pre在全部形状上完成性能目标。

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

## 下一步和结论边界

小N TL的API约79.536us，设备Duration合计约37.579us；两者来自独立计时模式，
差值只能提示主机调用、启动间隙和分配开销值得排查，不能作为精确开销分解。
当前TL Pre为5个kernel入口，AC为2个；优先实现系数尾部融合，预期降至3个入口，
保持完整投影及旧pre apply工作，每次调用重新执行，候选收益需同进程对照验证。
若融合后小N仍慢，再评估投影后处理进一步融合或等价图重放，后者单列结果。
大N仍重点关注apply/投影搬运和Sinkhorn吞吐。

AscendC为本仓库根据固定上游适配的原生实现，不是未经修改的官方融合算子。
目前权重文件字段均为None，仍使用合成有效权重；没有宣称实际模型权重已验证。
HF32内部投影模式和既有精度门限保持不变；不声称逐bit FP32算术等价。
