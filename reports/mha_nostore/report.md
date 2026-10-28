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
尚未采集此候选设备profile，暂不认定为已经确认的设备kernel收益，不替换默认基线。
下一步用现有run_nostore.sh profile对比设备耗时，再判断是否沿此方向继续。

原始证据：https://github.com/zhangwenhao91/Q4/blob/dab8ec6/error.txt
