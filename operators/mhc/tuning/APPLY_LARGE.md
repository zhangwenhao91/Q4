# 大N计算/搬运流水收益验证

```bash
bash operators/mhc/tuning/run_apply_large.sh smoke
bash operators/mhc/tuning/run_apply_large.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-apply-large
```

复用已通过小N精度的设备候选，不改数值源码。
性能N=1024/2048/4096/8192；内部块1024/1280、阶段1/3。
保留原H5120三kernel，不使用之前任务级H1280分块。
每进程五侧：原两kernel、原三kernel、嵌套流水、连续流水、适配AC，
同输入轮换位置和正反顺序，20轮、每轮100次、预热100次，三个进程。
check含2049尾数、五个N及三个种子。全部四输出及所有两侧互验先通过才计时。

判断收益有三个边界：
1. 相同内部块stages3是否比stages1更快。
2. 完整流水候选是否超过原H5120三kernel。
3. 实际是否应选流水候选，还是原两kernel仍然更快。

profile增加原三kernel（每N一次）与嵌套/连续候选（每N每配置各一次），
因此可以比较同轮apply Duration，但profile不是同进程交替的无噪声收益分解。
每类kernel30条；all预期144份正确性文件、48份API摘要、36个profile。
不因重叠比例增加就宣称性能更好；若候选有净收益，再采大N单核仿真定位机制。
无新增kernel实现，沿用旧pre四输出契约、合成权重与既有精度门限。
原默认实现保持不变，N8192是已有安全容量范围，不加入更大的未确认内存形状。
