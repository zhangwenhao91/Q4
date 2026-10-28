# Q128 softmax减少UB写回实验

手工阶段调度结果见reports/mha_manual/report.md：精度通过，但比基线慢4.11%，不采用。

本轮使用原始Q128/KV128/stages3、自动调度、BF16输入/输出。仅在softmax中：

- 第一遍读取原始分数，先取最大值再乘正数scale，不将缩放分数写回UB。
- 第二遍读取同一原始分数，在寄存器缩放后执行原有指数、求和及概率打包。

每行省去128个FP32分数的UB写入，同时增加部分缩放指令；API测量观察到约1%的改善，设备收益待profile确认，详见reports/mha_nostore/report.md。
对有限分数和本规格正scale，最大值选择与缩放可交换；概率打包和在线归一化保留。
所有barrier、buffer、流水线、其他宏不变，不叠加Q64/FMA/manual候选。
本机检查源码修改范围和NumPy数值关系，不能替代NPU编译/精度验证。

## 服务器

沿用zwh、CANN和q4fa vendor环境，无需重建AscendC。先执行：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
set -o pipefail
python operators/spec_v1/bench.py --suite mha_nostore --ac-metadata cached --check-only \
  --output "results/mha-tuning/nostore-first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee mha-nostore-first.log
```

成功后执行：

```bash
bash operators/mha/tuning/run_nostore.sh check
bash operators/mha/tuning/run_nostore.sh bench
```

三侧：原始TL、候选TL、原生AscendC复用metadata；门限不变。
bench进行3进程、各9轮、每轮100次，轮换顺序。若精度失败就停止。
只有观察到收益或需定位新问题时才追加profile：

```bash
bash operators/mha/tuning/run_nostore.sh profile
```
