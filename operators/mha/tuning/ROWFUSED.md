# Q128逐行寄存器softmax实验

前一候选减少UB写回的设备收益约0.92%。Level1显示其Vector活跃时间并未减少：
原始42.395us，候选42.974us；总时长仍下降约0.94%。这些重叠管线时间不能相加，
也不能据此证明Vector是唯一瓶颈。结果见reports/mha_nostore/pipe_metrics.json。

本轮从原始Q128独立生成，只替换softmax宏：每对行读取一次原始分数，
在寄存器完成缩放、max、exp、sum和在线m/l/alpha更新，保留原有BF16概率打包。
通过S.vdupv广播归约max，省去两遍扫描和中间统计量的UB往返。
保留原始buffer声明、latency提示735、Q128/KV128/stages3及自动调度；
编译器可能调整实际阶段安排。没有叠加FMA或手工调度。

代价是每行独立更新统计量，指令数及寄存器压力可能增加；不承诺提速。
本机NumPy检查在线状态数学关系和源码修改范围，不能代替NPU验证；
SIMD归约、指数实现及编译后的同步必须以NPU精度检查为准。
当前仍保留全部原始对照，不更改默认实现。

## 服务器

无需重建AscendC，沿用zwh/CANN/q4fa环境。先执行：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
set -o pipefail
python operators/spec_v1/bench.py --suite mha_rowfused --ac-metadata cached --check-only \
  --output "results/mha-tuning/rowfused-first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee mha-rowfused-first.log
```

首次检查通过后：

```bash
bash operators/mha/tuning/run_rowfused.sh check
bash operators/mha/tuning/run_rowfused.sh bench
```

若编译或精度失败则停止，保留日志，不回退隐藏错误。
确认API收益后才执行profile或pipe模式。
