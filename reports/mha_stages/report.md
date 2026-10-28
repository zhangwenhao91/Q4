# 两阶段/三阶段流水线结果

B1/H1/S4096/D128、BF16输入输出，原生AscendC复用metadata。
3种子精度及3次计时前检查，共90项全部通过。

| 实现 | API事件耗时：三进程中位数 us |
|---|---:|
| tilelang_q128 | 52.667751 |
| tilelang_nostore | 52.122426 |
| tilelang_s2 | 52.999761 |
| tilelang_nostore_s2 | 51.902931 |
| ascendc | 40.032744 |

nostore stages2是本轮API最快候选。原始softmax单独降至stages2没有收益，说明不能将阶段数收益与softmax改动视作可直接相加。
相对同轮nostore stages3的额外改善较小，暂不替换默认实现；下一步对两者及AscendC采集设备Duration。
源码及API测量不能证明buffer版本、实际重叠变化或关键路径原因。
完成设备确认后再独立实施输出归一化/BF16转换融合。

证据：https://github.com/zhangwenhao91/Q4/blob/03ee3e8/error.txt

## 设备计时复核

每侧30次，全部纳入；以下Duration均值，不含Wait Time。

| 实现 | 平均 us | 中位数 us |
|---|---:|---:|
| tilelang_nostore | 52.078533 | 52.030500 |
| tilelang_nostore_s2 | 51.765333 | 51.638500 |
| ascendc | 39.879067 | 39.893000 |

nostore stages2相对stages3平均耗时降低0.60%。
本轮证据：https://github.com/zhangwenhao91/Q4/blob/761d84b/error.txt
