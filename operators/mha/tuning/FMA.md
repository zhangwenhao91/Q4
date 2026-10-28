# Q128 Vector FMA 实验

前一轮Q64退化，未采用。Level1指标显示：Q128的Vector活跃时间约42.39us，原生AC约21.98us；
Q64的AIC MTE2比例约82.9%、MAC比例约54.5%，Q128约73.9%/82.6%。
这些字段存在重叠，不相加；busy比例和cube_utilization不是同一定义，不能按某个百分比直接判断吞吐。
指标支持优先检查Vector路径，但不能仅由这组聚合指标证明具体临界路径。

本实验回到原始Q128/KV128/stages3，只修改 accumulate_output：

```
旧：updated = (alpha * previous) + current，FP32乘加分别舍入
新：accumulated = current; vmula(accumulated, alpha, previous)，FP32融合乘加一次舍入
```

S.vmula的目的寄存器使用S.alloc_var(T.float32)，不向不可变SSA值写入。
其余shape、buffer、softmax、流水线和同步均保持；BF16内核直接输出；不叠加Q64存储补齐。
保留原始源码哈希检查，独立生成 third_party/mha-fma-candidate/core_fma.py。
此改动有FP32舍入差异，不能假设输出逐bit相同，仍使用相同CPU参考和精度门限。
尚未有NPU编译、精度或性能结果，不能宣称已有加速；编译器也可能已对旧路径做类似优化。

## 服务器先只做单种子精度

当前zwh、CANN和q4fa环境保留，不重建AscendC包：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
set -o pipefail
python operators/spec_v1/bench.py --suite mha_fma --ac-metadata cached --check-only \
  --output "results/mha-tuning/fma-first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee mha-fma-first.log
```

失败即更新error.txt；成功后再依次执行：

```bash
bash operators/mha/tuning/run_fma.sh check
bash operators/mha/tuning/run_fma.sh bench
bash operators/mha/tuning/run_fma.sh profile
```

同进程同输入三侧：tilelang_q128、tilelang_fma、ascendc（复用metadata）。
校验逐侧对参考及全部两两比较；门限不放宽。计时3进程、9轮、100次，执行位置平衡；
profile每侧30次调用。q128_over_fma>1表示改善TL，ascendc_over_fma>1表示超过AC。
不采用候选，除非同轮API和设备kernel时间、精度共同支持。

原始管线指标证据：Q4提交3494649的error.txt。
SIMD指令语义来自固定TileLang源码 tilelang/ascend/language/simd.py 的 vmula 定义，
以及已使用的TileKernels mhc/pre_apply_mix_asc.py 中alloc_var/vmula示例。
