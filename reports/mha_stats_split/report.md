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

## 设备复核

每侧30次，统计全部Duration，不含Wait Time。

| 实现 | 平均 us | 中位数 us |
|---|---:|---:|
| tilelang_q128 | 52.632667 | 52.565000 |
| tilelang_kv2 | 50.975167 | 50.830500 |
| tilelang_s2buf | 50.921633 | 50.784500 |
| ascendc | 39.845967 | 39.798000 |

S双版本相对KV2组合的设备均值耗时降低0.105%，相对原始TL降低3.251%。
AscendC仍快约1.278倍。
证据：https://github.com/zhangwenhao91/Q4/blob/c9348c0/error.txt

API和本次设备统计方向一致，但设备差值仅约0.054us，落在单次记录的波动范围内。
这不是统计显著性证明，不将0.105%当成已经稳定确立的收益；S双版本保留为观测最快候选。
配置为nostore+stages2+输出融合+K/V双版本+S双版本，alpha保留3版本，P和缓存保持默认。
后续若比较更大改动，应同轮保留已验证KV2对照，避免将此微小差异视作固定收益。
