# 显式buffer版本数结果

B1/H1/S4096/D128，BF16输入输出，原生AscendC复用metadata。
三种子检查及三次计时前检查，共126项精度全部通过。
计时3进程、每进程12轮、每轮100次，六侧轮换执行位置。

| 实现 | API事件耗时：三进程中位数 us |
|---|---:|
| 原始TL | 52.701621 |
| 此前最佳finalize组合 | 51.677735 |
| 仅K/V版本2 | 51.056559 |
| 仅S/alpha版本2 | 64.743211 |
| K/V/S/alpha均版本2 | 64.144139 |
| 原生AscendC | 40.105214 |

K/V版本2相对此前最佳耗时降低1.20%，相对同轮原始TL降低3.12%。
三个进程的此前最佳/候选加速比分别1.012474、1.011853、1.012257，方向一致。
API最快组合为nostore + stages2 + finalize融合 + K/V版本2；设备计时待复核。

S/alpha版本2导致明显退化，不能因为num_stages=2就将所有缓冲统一双版本。
目前同时改变了S和alpha，不能定位退化究竟由哪一个引起。
CODEGEN显示这些配置的流水线循环边界也发生变化，不能只按缓冲总字节数推断性能。
下一步只profile原始、此前最佳、K/V版本2和AscendC；不额外profile两个明显较慢候选。

原始证据：https://github.com/zhangwenhao91/Q4/blob/67119d6/error.txt

## 设备复核

每侧30次，全部纳入；Duration不含Wait Time。

| 实现 | 平均 us | 中位数 us |
|---|---:|---:|
| tilelang_q128 | 52.653333 | 52.753000 |
| tilelang_finalize | 51.744433 | 51.662000 |
| tilelang_kv2 | 51.071200 | 51.074000 |
| ascendc | 39.727567 | 39.708000 |

K/V双版本相对此前最佳设备平均耗时降低1.30%，相对原始TL降低3.00%。
当前最佳已测组合为nostore+stages2+finalize融合+K/V版本2，S/alpha仍3版本。
证据：https://github.com/zhangwenhao91/Q4/blob/ca81464/error.txt
