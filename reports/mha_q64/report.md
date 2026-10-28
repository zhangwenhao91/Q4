# Q64 分块实验结果：不采用

范围：B=1、H=1、S=4096、D=128，BF16输入/输出，复用AscendC metadata，三侧同输入、同进程对照。
Q64保留KV=128、stages=3；为64宽SIMD读写将行统计数组存储补齐至64，原始Q128源码不改。

36项精度记录通过（三种种子检查与三个计时进程）；Q64与Q128输出一致。

| 实现 | API事件耗时（3进程中位数）μs | kernel均值 μs | profile调用数 |
|---|---:|---:|---:|
| tilelang_q128 | 52.615 | 52.654 | 30 |
| tilelang_q64 | 65.133 | 64.970 | 30 |
| ascendc | 39.958 | 39.824 | 30 |

Q64 API耗时较Q128增加约23.8%，设备kernel耗时增加约23.4%。没有超过原始TL或AscendC。
保留Q128为当前最佳TL基线。数据否定了该配置下“缩小Q分块即可加速”的假设，
但仅由Duration不能断言具体是重复搬运、Cube效率或Vector/同步瓶颈。
下一步采集Level1 PipeUtilization指标，再决定流水线调整；不继续盲目缩小Q。

Profile曾提示ACL/NPU flow关联失败；设备CSV每侧30条完整。统计仅使用设备Duration，不推断CPU调用关联。

原始证据：
- https://github.com/zhangwenhao91/Q4/blob/c42c48b/error.txt （API）
- https://github.com/zhangwenhao91/Q4/blob/5d06bd5/error.txt （profile）

summary.json保存各进程摘要和kernel统计。此为负向优化实验，未替换已验证的原始实现。

## Level1 管线指标

每侧30次。以下为相应字段均值，采样可能影响耗时，不替换Level0的结论。

| 实现 | AIC MAC ratio | AIC MTE2 ratio | AIV Vector时间 μs | cube_utilization (%) |
|---|---:|---:|---:|---:|
| tilelang_q128 | 82.57% | 73.92% | 42.390 | 55.55 |
| tilelang_q64 | 54.52% | 82.90% | 38.186 | 75.09 |
| ascendc | 77.91% | 69.08% | 21.975 | 74.56 |

Q64的搬运活跃占比更高、MAC活跃占比更低，支持停止继续缩小Q；但不能据聚合指标证明精确临界路径。
Q128相较AC的Vector活跃时间明显较长，下一轮独立尝试Q128输出累积FMA，先精度再计时。
各管线重叠，比例不能相加；cube_utilization与MAC ratio定义不同，Q64某个百分比较高不等于有用吞吐更高。
原始证据：https://github.com/zhangwenhao91/Q4/blob/3494649/error.txt
