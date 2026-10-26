# Q4：算子性能对比

比较 TileLang 与 AscendC 实现，记录可复现的环境、正确性和性能数据。

| 算子 | 说明 | 状态 |
|---|---|---|
| [mHC](operators/mhc/README.md) | 实现核查、服务器安装指令、Post 前向对比 | 脚本已准备，等待 NPU 验证 |

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

当前只做了本机源码核查、Python 语法与 CLI 检查；没有 NPU 实测结果。Event 与 wall time 用于 API 性能比较，纯 kernel 结论需另用 profiler 验证。
