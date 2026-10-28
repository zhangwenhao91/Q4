# Q128 上游手工阶段调度实验

Q64退化，FMA未观察到有效API收益，均不采用。当前最佳仍为原始Q128。

本轮仅启用固定core.py已有的 manual_schedule=True：kv_manual用T.Stage为K搬运、QK GEMM、
softmax/概率打包、V搬运、PV GEMM、输出累积指定阶段。数值宏、shape、buffer版本和精度保留，
没有修改上游源码，也不叠加Q64/FMA。Q=128、KV=128、num_stages=3、BF16输入/输出，
对照原始自动调度与复用metadata的原生AscendC。候选是否更快尚未验证。

此实验测试上游已有的调度方案，不能将性能变化归因到某一条具体同步指令。
若编译器拒绝阶段约束或精度失败，停止并保留日志，不自动降级到自动调度。

## 服务器先单种子检查

保持当前zwh/CANN/q4fa环境，不重建AscendC：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
set -o pipefail
python operators/spec_v1/bench.py --suite mha_manual --ac-metadata cached --check-only \
  --output "results/mha-tuning/manual-first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee mha-manual-first.log
```

成功后依次执行：

```bash
bash operators/mha/tuning/run_manual.sh check
bash operators/mha/tuning/run_manual.sh bench
bash operators/mha/tuning/run_manual.sh profile
```

精度门限及三侧全部两两核对保持不变。3进程、9轮，每轮100次，轮换执行位置。
q128_over_manual>1表示改善原始TL，ascendc_over_manual>1表示超过同轮AC。
根据设备时间和API一起判断，不将微小波动视作加速。
