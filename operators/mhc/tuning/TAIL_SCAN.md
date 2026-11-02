# shifted5120 系数尾部核数/缓冲阶段扫描

```bash
bash operators/mhc/tuning/run_tail_scan.sh smoke
bash operators/mhc/tuning/run_tail_scan.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-tail-scan
```

token_block仍16：现有SIMD布局依赖完整16-token组，缩小需要重写，当前不改。
扫描尾部num_vec_cores=14/28/56与num_stages=1/2共六组，保持数值源码不变。
固定投影、旧pre四输出、精度门限及计时边界。候选和默认尾部基线均使用apply H1280，
apply核数仍为设备默认值，不随tail核数改变。两kernel及AC额外保留同轮对照。
注意本轮tilelang_coeff_bound标签表示H1280、默认尾部（非早期H5120基线），
metadata split_bound_launch.tail_scan记录两侧配置；不得跨轮错误混算。

N=64/128/256/512，三个进程，每进程16轮、每轮100次、预热100次。
check有129尾数及三个种子。N64上H1280不是推荐默认，此处只是控制变量。
默认56:2配置作为重复对照（服务器应有56向量核；超出设备核预算明确报错）。
all预期186份正确性文件、72份API摘要、24份profile。
各候选每调用三个kernel，profile每类30条；脚本目录K14:1等表示cores:stages，
并非投影K分片。所有输出与工作区每次重新分配，无预计算/图重放。
本地仅验证调用接口及配置隔离，NPU编译、精度和收益必须通过服务器smoke/all实测。
