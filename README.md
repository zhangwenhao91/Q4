# Q4：算子性能对比

比较 TileLang 与 AscendC 实现，记录可复现的环境、正确性和性能数据。

| 算子 | 说明 | 状态 |
|---|---|---|
| [mHC](operators/mhc/README.md) | Post 前向已实测；Pre / Sinkhorn 扩展验证脚本 | Post 已验证，Pre / Sinkhorn 待服务器验证 |

每个算子放在 `operators/<算子名>/`，包含独立说明和脚本。后续算子添加同级目录。

## 在 NPU 服务器开始

```bash
git clone https://github.com/zhangwenhao91/Q4.git
cd Q4
cat operators/mhc/README.md
```

按说明准备 Ascend 950、CANN 和 Python 环境后，在仓库根目录执行：

```bash
# 下载并锁定被比较的源码
bash operators/mhc/prepare_sources.sh
# 按 operators/mhc/README.md 构建安装两个实现，再执行：
bash operators/mhc/run.sh smoke
bash operators/mhc/run.sh full
```

`prepare_sources.sh` 不安装依赖或算子包；构建安装步骤见 mHC 文档。结果写入 `results/mhc/<时间>/`，源码下载到 `third_party/`，均默认不提交。

Post 前向已在服务器完成精度冒烟检查、API 复测和设备 profiler 采集；Pre / Sinkhorn 扩展脚本尚待服务器验证。Event 与 wall time 用于 API 性能比较，设备 kernel Duration 单独统计。

Pre、Sinkhorn 和 Pre+Sinkhorn 的安装、精度检查、计时及 profiler 指令见 [扩展测试说明](operators/mhc/extended/README.md)。

## mHC 反向覆盖

Post、Sinkhorn、Pre 与 Pre+Sinkhorn 的反向校验、独立反向及前向+反向计时，见 [反向运行说明](operators/mhc/backward/README.md)。Pre 梯度暂为 TileKernels + PyTorch 投影梯度的混合路径；NPU 验证结果待补充。

## 按同事规格表测试

严格规格的 MHA 单侧前向、mHC Post 和 shifted Pre 前向，参见 [规格 v1 运行说明](operators/spec_v1/README.md)。MHA 的 AscendC FP32 输出尚未匹配；shifted Pre 的 AscendC 侧为注明额外工作的混合组合路径。

## MHA BF16 双侧原生比较

用户确认将输出统一为 BF16，新增 [MHA BF16 原生比较说明](operators/mha/README.md)，分别统计 AscendC metadata 复用和每次生成的计时边界。历史 FP32 输出结果保留。

## MHA BF16 测试结果

[双侧原生前向结果报告](reports/mha_bf16/report.md)：固定形状且复用 metadata 时 AscendC API 快约1.318倍；每次生成 metadata 时 TL 完整调用快约1.554倍。报告分列API与设备kernel耗时，并保留原始日志提交号。
