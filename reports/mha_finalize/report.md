# 最终归一化/BF16转换融合：API结果

B1/H1/S4096/D128，BF16输入输出，AscendC复用metadata。
三种子精度检查及三个计时进程的检查共60项通过；各进程检查CPU参考与全部两两组合。

| 实现 | API事件耗时：三进程中位数 us |
|---|---:|
| tilelang_q128 | 52.679255 |
| tilelang_nostore_s2 | 51.846936 |
| tilelang_finalize | 51.609390 |
| ascendc | 40.006185 |

融合相对同轮nostore stages2耗时降低0.46%，相对原始TL降低2.03%。
收益较小，下一步通过设备profile复核，不将不同轮收益直接相加。尚未超过AscendC。
证据：https://github.com/zhangwenhao91/Q4/blob/23155ee/error.txt

## 设备复核

每侧30次，统计Duration，不包含Wait Time，未剔除记录。

| 实现 | 均值 us | 中位数 us |
|---|---:|---:|
| tilelang_q128 | 52.582567 | 52.571500 |
| tilelang_nostore_s2 | 51.850967 | 51.930000 |
| tilelang_finalize | 51.596333 | 51.581000 |
| ascendc | 39.709367 | 39.643500 |

输出融合比同轮nostore stages2降低0.49%，组合比原始TL降低1.88%。
AscendC仍快1.299倍。当前最佳已测组合为nostore + stages2 + finalize融合。
本轮未采集Level1，不将总时间差归因为单一管线；收益限于当前固定规格和环境。
证据：https://github.com/zhangwenhao91/Q4/blob/2d99bac/error.txt
