# Q128 Vector FMA 实验结果：未观察到有效API收益

规格：单head、4096x128、BF16输入/输出，AC复用metadata。未叠加Q64。
36项精度检查通过。FMA相较原始TL存在预期的FP32舍入差异，但均在相同门限内。

| 实现 | API事件耗时：3进程中位数 μs |
|---|---:|
| tilelang_q128 | 52.692962 |
| tilelang_fma | 52.692990 |
| ascendc | 39.895830 |

基线和FMA的汇总值几乎相同，进程间差异大于候选收益；不作为有效优化采用。
没有单独采集FMA的设备profile，本结论限于完整API调用；不能证明指令数未变化或编译器已经融合。
下一轮测试原始Q128的上游manual_schedule阶段方案，不叠加FMA。
原始证据：https://github.com/zhangwenhao91/Q4/blob/4947173/error.txt
