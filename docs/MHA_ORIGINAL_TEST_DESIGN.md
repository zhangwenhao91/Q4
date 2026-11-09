# 【TileLang】【A5】基于MHA算子验证TileLang版本能力

## SR 需求描述

验证原版TileLang MHA前向算子在32核Ascend 950DT上的功能与性能，规格以仓库根目录的[MHA和mHC规格表](../MHA和mHC规格表)为准。

| 维度 | 说明 |
|---|---|
| 需求来源 | TileLang版本能力校验 |
| 需求场景 | 使用原版MHA算子验证Attention计算的正确性与执行性能，检查版本功能与编译Pass性能问题 |
| 主成功场景 | 按下方测试算子规格输入Q、K、V，在NPU执行并输出符合精度要求的O，获得可复现的性能数据 |
| 技术需求点 | 使用上游原版TileLang DSL实现；验证QK矩阵乘、Softmax与PV矩阵乘的计算结果；保持原版默认分块与流水配置；记录精度、耗时及测试环境 |
| 硬件产品形态 | A5，Ascend 950DT，32个AIC、64个AIV |
| 验收准则 | MHA算子精度正确；具备同规格AscendC实现后，性能目标为AC/TL大于1.0 |
| 方案约束 | ops-transformer分支与CANN版本匹配；双方输入、输出类型及计算语义一致；本次仅测试原版，不引入优化变体 |

本需求覆盖Attention核心计算 `O = softmax(QKᵀ / sqrt(128))V`，不包含QKV线性投影和输出线性投影。SR及AR的正式需求编号由需求管理系统分配。

## AR 需求描述

| AR 编号 | AR 名称 | 输入 | 处理 | 输出 |
|---|---|---|---|---|
| AR-01 | Kernel执行验证 | Q、K、V均为[4096,128]，BF16 | NPU执行非因果MHA前向计算 | O为[4096,128]，FP32 |
| AR-02 | 精度验证 | 同一组Q、K、V | 与CPU FP32参考结果比较，检查形状、类型、有限值及数值误差 | 精度检查结果 |
| AR-03 | 性能验证 | 通过精度检查的固定规格输入 | 预热后重复测量；若具备同规格AscendC实现，使用相同输入和计时口径比较 | TL耗时；满足对比条件时输出AC耗时及AC/TL |

## 代码仓位置

TileLang仓库：[tile-ai/tilelang](https://github.com/tile-ai/tilelang)。测试固定源码提交为 `80364b3d9758854a9692cc0f14089025f54d3f3f`。

| 文件 | 用途 |
|---|---|
| [example_mha.py](https://github.com/tile-ai/tilelang/blob/80364b3d9758854a9692cc0f14089025f54d3f3f/examples/ascend/flash_attention/example_mha.py) | 原版mha调用入口，显式指定FP32输出 |
| [core.py](https://github.com/tile-ai/tilelang/blob/80364b3d9758854a9692cc0f14089025f54d3f3f/examples/ascend/flash_attention/core.py) | flash_attention_fwd内核定义及默认分块配置 |
| [test_mha.py](https://github.com/tile-ai/tilelang/blob/80364b3d9758854a9692cc0f14089025f54d3f3f/examples/ascend/flash_attention/test_mha.py) | 上游测试调用 |
| [Q4测试入口](../operators/spec_v1/bench.py) | 使用 `--suite mha` 执行原版规格精度与性能测试 |
| [Q4用例定义](../operators/spec_v1/cases.py) | mha_case定义及CPU参考计算 |

AscendC候选仓库：[cann/ops-transformer](https://gitcode.com/cann/ops-transformer)。此前使用的源码提交为 `ba5162411c919e15fbabb28298e8b420c11ad905`，算子为 `flash_attn` 和 `flash_attn_metadata`。该路径当前为BF16输入、BF16输出，与本次FP32输出规格不一致；找到并验证同规格实现前，仅记录TL单侧结果，不输出AC/TL。

## 环境配置

1. NPU服务器：Ascend 950DT，32个AIC、64个AIV。
2. CANN：9.3.0。
3. Python：3.12。
4. torch_npu：2.13.0。
5. torch：2.13.0+cpu。
6. Ascend TileLang：0.1.15。

上述软件版本沿用Q4已有测试环境记录，在目标950DT上需先确认安装兼容性，并保存实际版本、设备型号和可用核数。历史28核950PR测试数据不能作为本设备的验收结果。

## 测试算子规格

以下参数用于32核950DT，序列长度固定为4096，不采用7/8缩放。运行时动态shape为无，Q、K、V的数据内容为运行时输入。

**构建MHA Kernel**

| 变量 | 含义 | 数值 | 变化规则 |
|---|---|---|---|
| batch | 批大小 | 1 | 本次固定 |
| heads | 注意力头数 | 1 | 本次固定 |
| SEQ_LEN | Q、K、V序列长度 | 4096 | 按规格表固定 |
| head_dim | 注意力头维度 | 128 | 按规格表固定 |
| scaling | 注意力分数缩放系数 | 1/sqrt(128)，约0.08838835 | 按规格表固定 |
| out_dtype | 输出数据类型 | FP32 | 按规格表固定 |
| causal | 是否启用因果屏蔽 | False | 按规格表固定 |
| dropout | 随机丢弃概率 | 0 | 按规格表固定 |
| block_q | Q分块大小 | 128 | 上游默认值 |
| block_kv | KV分块大小 | 128 | 上游默认值 |
| num_stages | 流水级数 | 3 | 上游默认值 |
| num_blocks | 逻辑Q任务数 | 32 | 原版默认4096/128；对应32个Q分块 |

不添加attention bias，不返回LSE。本规格均为整块，不新增尾块场景。

**使用MHA Kernel**

| 变量 | 输入输出 | 形状 | 类型 |
|---|---|---|---|
| Q | 输入 | [4096,128] | BF16 |
| K | 输入 | [4096,128] | BF16 |
| V | 输入 | [4096,128] | BF16 |
| O | 输出 | [4096,128] | FP32 |

## 验收标准

| 维度 | 要求 |
|---|---|
| 精度 | 与CPU FP32参考结果比较，relative L2不大于0.01，最大绝对误差不大于0.02；输出形状及类型正确，且无NaN或Inf。该口径沿用Q4的MHA用例，不照搬GQA阈值 |
| 性能 | 同规格下目标AC/TL大于1.0，即AscendC耗时除以TileLang耗时；满足该值时表示TL更快。当前同规格AC路径待落实，性能验收待完成 |
| 测量口径 | 先通过精度检查，再进行3次独立进程测量；每次预热100次，测量10轮、每轮100次，以NPU Event单次耗时为主，保留各进程结果与汇总 |
| 对比公平性 | 相同设备、输入、输出类型、计算语义及计时边界；编译和初始化不计入执行耗时；若AC使用metadata，应明确复用与否并分别记录 |
| 结果留档 | 保存源码提交、实际软硬件环境、精度输出、原始计时及汇总；不得将28核设备或S3584的历史结果代入本次结论 |

当前文档定义测试方案，不代表已完成32核950DT实测。将BF16输出转换为FP32不能替代原生FP32输出的同规格实现。
