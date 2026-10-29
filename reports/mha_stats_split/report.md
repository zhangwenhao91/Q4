# S与alpha版本数拆分结果

B1/H1/S4096/D128，BF16输入输出，原生AscendC复用metadata。
90项精度检查全部通过；3计时进程、每进程10轮，五侧执行位置平衡。

| 实现 | API事件耗时：三进程中位数 us |
|---|---:|
| tilelang_q128 | 52.743690 |
| tilelang_kv2 | 51.145120 |
| tilelang_s2buf | 51.020999 |
| tilelang_alpha2buf | 63.124545 |
| ascendc | 40.092840 |

S从3版本改2，alpha保留3版本：耗时降低0.24%。
alpha从3版本改2，S保留3版本：耗时增加23.42%。
本轮明确alpha双版本单独就会导致明显退化，不能将此前S/alpha同时改2的退化归因于S。
但调度具有交互效应，也不能把组合差值完全线性归因于alpha或断言其物理关键路径。
S双版本的额外API收益很小，下一步通过设备profile复核后再决定是否推进当前最佳。
alpha继续保留3版本，不为明显退化候选追加profile。
证据：https://github.com/zhangwenhao91/Q4/blob/8d5878e/error.txt
