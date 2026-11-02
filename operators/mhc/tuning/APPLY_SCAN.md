# shifted5120 独立apply分块扫描

```bash
bash operators/mhc/tuning/run_apply_scan.sh smoke
bash operators/mhc/tuning/run_apply_scan.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-apply-scan
```

只改变三kernel路径的独立旧pre apply hidden tile：256/512/1024/1280/2560/5120。
投影、系数尾部、旧pre语义、四输出和精度门限保持不变。5120是重复配置对照。
每档与当前两kernel、默认三kernel、适配原生AC四侧同进程比较，
16轮覆盖两个完整位置/正反顺序周期，三个进程，预热100次、每轮100次。
性能N=64/128/256/512/8192；check包括129尾数和三个种子。
每次完整分配并重新计算，预绑定不缓存数值或复用输出。

block均为64倍数且整除5120，不改上游数值源码，任务数量从N变为N*(5120/block)。
增加任务可能提高并行度，也会增加调度和系数重复读取，收益须实测。
metadata的split_bound_launch.apply_scan保存候选源码哈希和分块。
all预期210份正确性文件、90份API摘要、30个profile；smoke先验证六档N128。
profile每个候选三类kernel各30条。脚本中的K标签沿用扫描目录命名，
此处K数字表示apply hidden tile，不表示投影K分片；实际候选标签tilelang_apply_h*。
当前无NPU性能结论，保持原默认实现，等待设备测试后选择。
