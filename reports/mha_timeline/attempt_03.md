# Kernel 名称过滤采集成功

证据：71d4bd3:error.txt；目录 timeline-load-20261029-172926-2563195。
S2、KV2 均精度通过，relative L2 0.00242606697，max abs 0.00048828125。
两侧实际采集 main_kernel，工具确认 50 次预热，成功导出 8 类 CSV 和二进制数据。

| 实现 | 本轮 Task Duration (us) | block 0 cube 时间 (us) |
|---|---:|---:|
| S2 | 53.632 | 26.112728 |
| KV2 | 53.798 | 26.317575 |

S2 本轮总耗时低 0.166 us，约 0.309%。这是单次诊断采集，不构成稳定收益证明，
不替代常规基准，也不与之前 AscendC 常规时间直接计算加速比。
Block Dim=32，Mix Block Dim=64，频率 1650 MHz。

Occupancy 报告存在耗时较长的标签组，S2 与 KV2 的列表不完全一致。
CSV 使用 block_id/sub_block_id，不能直接视为物理核编号；当前预览仅包含 block 0。
既不能把约 26 us 的 block 时间直接乘二解释整个任务，也不能据此确定 32/28 调度尾部。
MTE 活跃带宽低于 80% 的工具提示同样不足以单独证明带宽瓶颈。

下一步先导出现有 CSV，无需重新采集：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
python operators/mha/tuning/timeline_details.py --input results/mha-tuning/timeline-load-20261029-172926-2563195
```

将输出更新至 error.txt。导出包含完整 block 流水线耗时分布，以及各 sub-block 组
缓存命中率、等待和冲突指标的 min/median/max；不合并不同采集的时间轴。
拿到分布后再决定 PipeTimeline 采集或具体 kernel 修改。
