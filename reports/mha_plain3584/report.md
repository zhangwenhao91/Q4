# 原版TL MHA，序列长度7/8：3584

2026-11-03，证据1437c47:error.txt。
results/mha-plain3584/20261103-151933-all-216035，日志记录运行提交
661c44965a68d07e78aab30a7aa07b10ce9df6bd，上游TL源码版本
80364b3d9758854a9692cc0f14089025f54d3f3f。

## 规格与验证

Q/K/V及O=[3584,128]，输入输出BF16，batch=heads=1，scale=1/sqrt(128)，
causal=False、dropout=0，无bias、不返回LSE。只缩序列4096*7/8=3584，head_dim不变。
O从原规格FP32改为双方直接写BF16，为用户明确授权的差异，无额外输出转换。
TL调用固定上游core.py默认构建配置，未加入之前分块、缓冲、融合或流水优化候选；
编译fast-math沿用上游example。源码哈希由入口验证，未升级服务器TileLang。
48项输出检查全部通过，包含CPU参考及两侧互验；实际模型精度仍不在本次结论范围。

## 完整API

单位us，三个进程各自10轮中位数的中位数，预热100次，每轮100次，双侧交替顺序。
全部attention与输出/临时分配计时，编译排除；两种metadata口径分别计量。

| AC metadata口径 | TL event | AC event | AC/TL |
|---|---:|---:|---:|
| cached，复用固定shape metadata | 24.681 | 26.558 | **1.076×** |
| per_call，每次重新生成metadata | 24.625 | 64.796 | **2.631×** |

cached逐进程AC/TL为1.0813/1.0749/1.0742，三个进程均TL更快，wall方向一致。
per_call为2.6184/2.6314/2.6219，三个进程也均领先，但包含AC metadata生成成本。
对固定shape反复调用的attention主体比较，应优先引用cached的1.076×。
不能将per_call的2.631×描述为attention计算kernel本身的优势。
完整进程摘要保存在summary.json。

## 设备Duration

每组一次profile，kernel均30条，单位us，均值，不与API直接相减作为精确host开销。

| 口径 | TL main_kernel | AC FlashAttn | AC metadata |
|---|---:|---:|---:|
| cached | 24.526 | 26.646 | — |
| per_call | 24.600 | 26.550 | 38.269 |

cached AC/TL设备Duration约1.086×。per_call的额外延迟主要对应metadata kernel，
与完整API差异方向一致。日志的profile-*.log CSV0项是离线脚本误将日志当目录，
已修正过滤；实际四个profile目录均各一份CSV，未影响上述kernel统计，无需重跑。

## 结论

在此次3584、head_dim128、双方BF16输出的同规格比较中，未经调优的上游TL默认MHA
相对原生AC固定metadata路径获得约1.076倍完整API加速。
主要矩阵乘计算量相对4096为49/64，而不是7/8；不同序列长度间不混算同工作加速比。
本报告不采用其他优化MHA候选或mHC结果。
