# shifted5120 小N投影扫描

```bash
bash operators/mhc/tuning/run_projection_scan.sh smoke
bash operators/mhc/tuning/run_projection_scan.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-projection-scan
```

固定旧pre语义和四输出。保留原两kernel、原三kernel、适配原生AC，
每次加入一个三kernel投影候选，四侧16轮位置/顺序平衡，三个进程。
N=64/128/256/512，K splits=1/2/4。K4可能与默认配置相同，作为重复配置对照保留。
不改变token块128、hidden块256或设备源码。按K分片重新推导活动核数、
每分片核数、每分片hidden块数与chunk长度，活动核预算不超过默认配置。
设置超过预算时明确报错，不静默超配。

分片改变浮点归约顺序，必须通过未放宽的原始投影、平方和及四输出CPU/互验门限。
每次仍分配并重算全部输出，无预计算，无图重放。配置及生成设备源码哈希记录在
metadata.json的split_bound_launch.projection_scan中。
profile测每个候选的三阶段Duration，原两/三kernel和AC的API在所有候选进程中保留。
all预期93份正确性文件、36份API摘要、12个profile；check包含129尾数与三个种子。
本脚本使用合成权重。尚无NPU性能结论，先运行smoke确认编译，再运行all。
