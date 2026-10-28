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
