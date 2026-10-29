# Q/O L2策略结果：保留默认

B1/H1/S4096/D128，BF16输入输出，原生AscendC复用metadata。
126项精度检查全部通过，3个计时进程、各12轮，六侧执行位置平衡。

| 实现 | API事件耗时：三进程中位数 us |
|---|---:|
| tilelang_q128 | 52.720766 |
| tilelang_kv2 | 51.146965 |
| tilelang_q_l2 | 51.474030 |
| tilelang_o_l2 | 51.197040 |
| tilelang_qo_l2 | 51.568894 |
| ascendc | 40.036325 |

各缓存候选相对同轮KV2最佳的耗时变化：
- tilelang_q_l2：+0.639%。
- tilelang_o_l2：+0.098%。
- tilelang_qo_l2：+0.825%。

没有超过当前最佳，不采用。仅O改动差异很小，也不能当作有效收益。
结果限于相同输入地址重复预热/计时，不表示所有业务输入复用条件下都一样。
模式4为非分配策略，不能将结果解释成实际HBM或L2字节数的测量。
不为这批未提速候选追加profile。下一优先级拆分S和alpha版本数。
证据：https://github.com/zhangwenhao91/Q4/blob/d3009d7/error.txt
