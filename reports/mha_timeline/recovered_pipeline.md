# 重建后首次流水线采集

证据：08a82bf:error.txt。实际输出目录：
`results/mha-tuning/timeline-pipeline-20261030-143525-3487443`。

两侧 TL 精度检查与采集均完成；物理设备 4、逻辑设备 0。
relative L2 均为 0.0024260669693699313，max abs 均为 0.00048828125。
固定源码和 q4fa 包的 make_case 前置检查通过，原生 metadata 初始化可用；
这不等于已重新验证 AscendC attention 本体的精度和性能。
日志开头扩展编译后继续成功运行；稀疏算子 converter 导入警告未阻断本次流程。

| 实现 | 本轮诊断 Task Duration (us) | trace.json 字节数 |
|---|---:|---:|
| S2 | 53.164001 | 304729 |
| KV2 | 52.669998 | 359944 |

本轮 S2 比 KV2 慢约 0.94%，先前诊断采集曾快约 0.31% 和 2.19%。
诊断差距方向并不稳定，不能将这些单次采集当作稳定优化收益，也不能仅据此改选基线。
需要原始时间轴来分析流水线；重建后环境/库哈希的可比性也要通过导出 metadata 核对。

## 导出并推送本次采集

无需再次编译或采集。用新的独立目录，避免与已丢失的旧采集混淆。
以下每步成功后再执行下一步；推送失败时保留仓库和结果目录。

```bash
cd /home/z30086261/Q4
git pull --no-rebase
python operators/mha/tuning/export_timeline.py \
  --input results/mha-tuning/timeline-pipeline-20261030-143525-3487443 \
  --output reports/mha_timeline/captures/20261030_pipeline_recovered
git add reports/mha_timeline/captures/20261030_pipeline_recovered
git commit -m "Add recovered MHA pipeline capture"
git push origin main
```

本机获取文件后校验 manifest 哈希，再检查时间轴格式、轨道和事件覆盖范围。
