# PipeTimeline 采集成功，待传回时间轴

证据：893fad4:error.txt，目录 timeline-pipeline-20261030-091704-3140527。
两侧精度均通过，工具成功结束；除聚合 CSV 外，新增：

| 实现 | trace.json 字节数 | visualize_data.bin 字节数 | Task Duration (us) |
|---|---:|---:|---:|
| S2 | 356378 | 689908 | 52.965000 |
| KV2 | 369379 | 702892 | 54.148998 |

两侧 dump/timeline.bin.0 均为 37748880 字节。当前文本仅列出了这些文件，
没有事件内容，故尚不能声称已定位 QK/PV 的等待、具体同步依赖或物理核调度。
本轮 S2 诊断时间比 KV2 低约 2.19%，不同于上轮约 0.31%；不将单次采集差距
替代常规重复基准或作为新优化收益。当前没有更改 kernel。

下一步导出已有文件供本机分析，避免继续通过摘要反复采集。
export_timeline.py 复制 trace、可视化数据、CSV、源码及采集配置，记录 SHA256；
不复制大型原始 dump，不覆盖已有导出，不执行 NPU 工作。

```bash
cd /home/z30086261/Q4
git pull --no-rebase
python operators/mha/tuning/export_timeline.py \
  --input results/mha-tuning/timeline-pipeline-20261030-091704-3140527 \
  --output reports/mha_timeline/captures/20261030_pipeline
git add reports/mha_timeline/captures/20261030_pipeline
git commit -m "Add MHA pipeline traces for offline analysis"
git push origin main
```

推送后本机检查 trace 格式、时间单位、轨道标签及事件覆盖范围，再决定如何计算
重叠与空隙。不能预设 trace 一定包含指令级事件或完整 block 到物理核映射。
官方工具说明区分管线图和指令图：
[msopprof 用户指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)。
