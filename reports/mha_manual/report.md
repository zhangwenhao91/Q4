# Q128手工调度结果

规格：单head、4096x128、BF16输入/输出，AscendC复用metadata。36项精度检查通过。

| 实现 | API事件耗时：3进程中位数 μs |
|---|---:|
| 原始Q128 | 52.673960 |
| 手工阶段调度 | 54.837489 |
| 原生AscendC | 39.936891 |

候选比原始TL慢4.11%，不采用。未采集此候选设备profile，不能定位具体阶段。
当前最佳仍为自动调度Q128。本结果不叠加FMA/Q64。
原始证据：https://github.com/zhangwenhao91/Q4/blob/b1f808b/error.txt
