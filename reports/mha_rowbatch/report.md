# 批量统计修正版结果：不采用

单head、4096x128、BF16输入/输出，原生AscendC复用metadata。
36项精度检查全部通过；已测输入上候选与原始TL逐元素相同。

| 实现 | API事件耗时：三进程中位数 μs |
|---|---:|
| tilelang_q128 | 52.675891 |
| tilelang_rowbatch | 55.926719 |
| ascendc | 39.998391 |

候选比同轮原始TL慢6.17%，不采用。与上一实验逐行融合约61.97us相比有所恢复，
但跨轮差异不能精确归因于某条指令；未采集rowbatch设备profile。
当前最佳已验证候选仍是nostore：API约1.02%、设备约0.92%的耗时降低，未超过AscendC。
本轮不再追加profile；两个单遍逐行方案均退化，暂停此具体改法。
后续应优先检查QK/PV的tile形状、L0搬运和流水线重叠，保留nostore为候选对照。
此前Level1中TL的AIC MAC活跃时间约43.5us、AC约31.3us，这说明差距并未局限在softmax；
这些聚合值有重叠，也不是关键路径分解，不能相加或直接归因为某个布局。

证据：https://github.com/zhangwenhao91/Q4/blob/556ee1e/error.txt
