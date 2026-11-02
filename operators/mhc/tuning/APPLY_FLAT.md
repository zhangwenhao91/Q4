# 跨token连续apply流水

```bash
bash operators/mhc/tuning/run_apply_flat.sh smoke
bash operators/mhc/tuning/run_apply_flat.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-apply-flat
```

参考reports/MHC_PIPELINE_STUDY.md中的Ascend连续job流水思路，使用已在服务器
验证可编译的T.Pipelined与多版本buffer，不依赖较新T.Stage API，不升级TileLang。
原H5120三kernel基线保留；token仍按core_id+job*num_vec_cores归属原核。
step=job*hidden_tiles+hidden_tile，一条循环跨token推进，减少逐token排空的可能性。
精确per-core jobs=ceil(max(N-core_id,0)/cores)，避免无效token，不加循环内token条件。
核间迭代次数可以不同；当前kernel没有显式跨核同步，仍需设备编译与运行验证。

内部块1024/1280，每档stages1/3；每个进程五侧同输入：原两kernel、原三kernel、
嵌套流水、连续流水、适配AC。20轮覆盖两个完整五侧位置/正反周期，三进程。
嵌套和连续候选使用相同内部块/stages，VF算术AST相同；全部重算与分配边界不变。
N=64/128/256/512，check另含129尾数与三个种子。
all预期124份正确性文件、48份API摘要、16个profile；每类kernel30条。

生成代码分别保存apply_pipeline_kernel.cpp（嵌套）和apply_flat_pipeline_kernel.cpp（连续），
profile针对连续候选；原基线、嵌套和AC的完整API每进程保留。
实际MTE2/Vector/MTE3重叠必须由生成代码及时间轴确认，stage参数本身不是证据。
只做实验候选，未改变默认实现。先smoke确认符号迭代及buffer版本编译和精度。
