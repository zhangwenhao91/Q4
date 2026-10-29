# P_shared双版本实验：不采用

B1/H1/S4096/D128，BF16输入输出，原生AscendC复用metadata。
60项精度全部通过；候选与原始及此前最佳在已测输入上逐元素相同。

| 实现 | API事件耗时：三进程中位数 us |
|---|---:|
| tilelang_q128 | 52.732406 |
| tilelang_kv2 | 51.149712 |
| tilelang_pbuffer | 53.465040 |
| ascendc | 40.072856 |

P双版本相对当前KV2最佳耗时增加4.53%，不采用。
CODEGEN_IDENTICAL_PBUFFER=False；候选P的L1读写地址包含随KV迭代交替的槽位索引，
而当前最佳P的L1地址固定。说明注解确实改变了该路径，不能认定编译器原本已经双缓冲。
同时AIC循环边界40变为37，实际调度和资源安排发生变化；不能据边界变短推断更快。
不为明显退化候选追加profile。保持原有P配置，按优先级转入Q/O的L2策略独立实验。
证据：https://github.com/zhangwenhao91/Q4/blob/09e0996/error.txt
