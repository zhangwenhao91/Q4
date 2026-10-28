# Q128 softmax减少UB写回：初步API收益约1%

单head、4096x128、BF16输入/输出，AscendC复用metadata；相同输入及精度门限。
3个种子检查及3次计时前检查共36项通过，候选与原始TL在这些输入上逐元素相同。

| 实现 | API事件耗时：3进程中位数 μs |
|---|---:|
| 原始Q128 | 52.645249 |
| softmax减少写回 | 52.110252 |
| 原生AscendC | 39.924481 |

3个进程各自的基线/候选加速比分别为1.01030、1.01107、1.01018。
汇总中位耗时降低约1.02%，有一致的小幅改善；尚未超过AscendC，AscendC仍快约1.305倍。
后续设备profile已确认同方向的小幅改善，详见下方；保留原始实现作为固定对照。

原始证据：https://github.com/zhangwenhao91/Q4/blob/dab8ec6/error.txt

## 设备profile复核

每侧30次kernel记录，均纳入统计，无剔除；以下为Duration(us)，不含Wait Time。

| 实现 | 均值 μs | 中位数 μs |
|---|---:|---:|
| tilelang_q128 | 52.645667 | 52.592000 |
| tilelang_nostore | 52.159133 | 52.169000 |
| ascendc | 39.790167 | 39.863500 |

候选设备平均耗时降低0.92%，支持前述小幅API收益；仍未超过AscendC。
本次只有默认设备计时，没有管线计数器，不能据此确认具体瓶颈。ACL流关联告警不影响这里完整的30条设备耗时，但不用于CPU/NPU关联分析。
证据：https://github.com/zhangwenhao91/Q4/blob/5b37592/error.txt
