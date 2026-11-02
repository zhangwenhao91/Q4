# 原三kernel基线：apply内部搬运/计算流水实验

```bash
bash operators/mhc/tuning/run_apply_pipeline.sh smoke
bash operators/mhc/tuning/run_apply_pipeline.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-apply-pipeline
```

保留原H5120三kernel、原两kernel、适配AC，不采用前轮H1280任务分块。
候选外层Persistent只按token分任务，每个token内部hidden用T.Pipelined迭代，
内部块256/1024/1280分别测stages1与3，共六档。相同内部块的1/3阶段成对对照，
用于区分细分搬运的收益与流水版本的收益。候选仍为完整三kernel。
apply数值VF由固定TileKernels c1cf914 pre_apply_mix_asc.py复制，保留MIT版权来源，
见LICENSE_TILEKERNELS；只改变任务循环和缓冲版本，旧pre四输出语义不变。
内部tile均整除5120，无hidden尾块。token尾数129及多种子继续检查。

这只是编译器软件流水候选，尚未证明MTE2/Vector/MTE3产生重叠。
嵌套Persistent/Pipelined可能受当前TileLang版本限制，先smoke验证编译和精度。
每个结果目录保存apply_pipeline_kernel.cpp及源码哈希，供检查事件等待和buffer版本。
成功运行后须检查流水图，不能以num_stages=3或Duration变短单独证明真实重叠。
缓存、搬运粒度及核利用率也可能导致性能变化。

N=64/128/256/512，各候选三进程，16轮四侧顺序平衡、每轮100次、预热100次。
all预期186份正确性文件、72份API摘要、24个profile；每类kernel30条。
profile保存于结果目录，采集方式沿用现有profiler。全部输出和工作区每次重分配，
全部投影和计算完整计时，无数值缓存或图重放。
当前默认实现不变；必须同时通过精度、apply设备Duration和完整API验证。
