# 【A5】【TileLang】原版MHA算子实践

## 测试方法

### 1. 配套环境要求

1. Ascend950PR_9579服务器，28个AIC、56个AIV；已有测试使用物理设备4、进程内逻辑设备0。
2. CANN：9.3.0.weekly.20260930。
3. Python：3.12，使用已有Conda环境`zwh`。
4. torch_npu：2.13.0。
5. torch：2.13.0+cpu。
6. Ascend TileLang：0.1.15+cpu.gita7cb753d。
7. 编译工具：已有Conda GCC 13；原生FlashAttn与FlashAttnMetadata已安装到Q4独立`q4fa` vendor。

以上为本项目已验证环境。复测使用现有环境，记录实际设备、软件版本和代码提交。

### 2. 测试算子规格

MHA规格：构建kernel时计算属性如下。

| 计算属性 | 静态或动态 | 取值 |
|---|---|---|
| batch | 静态 | 1 |
| heads | 静态 | 1 |
| SEQ_LEN，Q/K/V序列长度 | 静态 | 3584，即原4096的7/8 |
| head_dim，注意力头维度 | 静态 | 128 |
| scaling，注意力分数缩放系数 | 静态 | 1/sqrt(128)，约0.08838835 |
| out_dtype，输出数据类型 | 静态 | bfloat16 |
| block_q / block_kv | 静态 | 128 / 128，原版默认值 |
| num_stages | 静态 | 3，原版默认值 |
| num_blocks，逻辑Q任务数 | 静态推导 | 3584/128=28，使用原版默认推导 |
| causal，因果屏蔽 | 静态 | False |
| dropout，随机丢弃概率 | 静态 | 0 |
| attention bias / LSE输出 | 静态 | 无 / 不返回 |

运行时动态shape：无；Q/K/V数据内容为运行时输入。

使用kernel的输入输出：

```text
input:
Q  shape:[SEQ_LEN,head_dim] = [3584,128]  dtype:BF16
K  shape:[SEQ_LEN,head_dim] = [3584,128]  dtype:BF16
V  shape:[SEQ_LEN,head_dim] = [3584,128]  dtype:BF16

output:
O  shape:[SEQ_LEN,head_dim] = [3584,128]  dtype:BF16
```

计算为`O = softmax(Q @ K.T / sqrt(128)) @ V`，覆盖注意力核心；QKV投影及输出投影不在本次范围内。
AC通过view使用同一组输入的BSND布局`[1,3584,1,128]`，不增加转置或数值补算。
双方内核直接写BF16。原规格表中的FP32输出已按任务确认统一为BF16，原序列长度4096已缩为3584。

TL调用固定上游`core.py::flash_attention_fwd`，使用原版默认tiling，编译fast-math沿用上游示例。
AC使用固定上游原生FlashAttn。PyTorch负责张量分配、view、调用与计时。

### 3. 算子性能

性能目标：同规格下`AC/TL > 1.0`，表示TileLang完整调用快于AscendC。
其中`AC/TL = AscendC API event延迟 / TileLang API event延迟`。

**测试口径**

- 精度：CPU FP32分块参考最终舍入BF16；双方各对参考校验，并相互比较。
- 验收：shape、dtype一致，输出finite，relative L2≤1%，max abs≤0.02；失败停止计时。
- 计时：每种metadata口径3个独立进程，每进程预热100次、10轮、每轮100次，双侧交替顺序。
- 汇总：先取各进程轮次中位数，再取3进程中位数；计入attention及输出/临时分配，排除首次编译和输入准备。
- cached：metadata计时外生成并复用，为固定shape的主比较口径。
- per_call：每次AC调用重新生成metadata并计时，作为补充口径单独报告。

**已验证结果**

2026-11-03原版独立复测，48项输出检查通过。时间单位为微秒。

| AC metadata口径 | TL event | AC event | AC/TL | TL矩阵MFU | AC矩阵MFU |
|---|---:|---:|---:|---:|---:|
| cached，主口径 | 24.681 | 26.558 | **1.076×** | **70.49%** | **65.51%** |
| per_call，补充口径 | 24.625 | 64.796 | 2.631× | 70.66% | 26.85% |

cached的三个进程AC/TL为1.0813、1.0749、1.0742，三个进程均TL更快，wall计时方向一致。
主结论引用cached的1.076×。per_call的2.631×包含metadata生成成本，不能作为attention计算kernel本身的加速比。

MFU按有效矩阵计算量与完整API延迟计算：

```text
FLOPs = 4 * batch * heads * SEQ_LEN² * head_dim
      = 6,576,668,672

有效TFLOPS = FLOPs / event_us / 1e6
矩阵MFU(%) = 有效TFLOPS / 378 * 100
```

分母使用28 AIC的Cube BF16峰值378 TFLOPS。softmax等操作不计矩阵FLOPs，但耗时计入完整调用。
这是算子级有效矩阵MFU，不等于硬件忙碌比例。序列缩为7/8后，矩阵FLOPs为原来的49/64。
上表MFU由本轮未舍入延迟离线计算；不同metadata口径或不同序列长度的时间不混算加速比。

### 4. 代码仓

项目仓库：[zhangwenhao91/Q4](https://github.com/zhangwenhao91/Q4)。

| 内容 | 位置 |
|---|---|
| 原版TL数值源码 | [operators/spec_v1/upstream_mha/core.py](../operators/spec_v1/upstream_mha/core.py) |
| 原版S3584对比入口 | [operators/spec_v1/mha_plain3584.py](../operators/spec_v1/mha_plain3584.py) |
| 公共精度与计时框架 | [operators/spec_v1/bench.py](../operators/spec_v1/bench.py) |
| 服务器执行脚本 | [operators/mha/run_plain3584.sh](../operators/mha/run_plain3584.sh) |
| 离线结果汇总 | [operators/mha/collect_plain3584.py](../operators/mha/collect_plain3584.py) |
| 测试结果与原始进程摘要 | [报告](../reports/mha_plain3584/report.md)、[summary.json](../reports/mha_plain3584/summary.json) |

源码版本固定：

- TileLang：`80364b3d9758854a9692cc0f14089025f54d3f3f`，来源为tile-ai/tilelang。
- AscendC：`ba5162411c919e15fbabb28298e8b420c11ad905`，来源为cann/ops-transformer，仅编译flash_attn、flash_attn_metadata。
- 上述复测运行版本：`661c44965a68d07e78aab30a7aa07b10ce9df6bd`；原始日志证据为`1437c47:error.txt`。

**执行方法**

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
bash operators/mha/run_plain3584.sh smoke
```

冒烟通过后执行完整精度、计时和设备profile：

```bash
bash operators/mha/run_plain3584.sh all
```

已有原生AC算子包无需重建。all结果写入`results/mha-plain3584/<时间>-all-<进程号>`。
正常应有16份correctness文件、6份API摘要、4个profile目录、48项输出检查；设备目标kernel各30条。
独立smoke目录不计入all目录数量。

汇总指定结果目录时执行：

```bash
python operators/mha/collect_plain3584.py \
  --input results/mha-plain3584/20261103-151933-all-216035
```

复测时将示例路径替换为实际新目录。交付时保留metadata、correctness、timings.csv、summary.json和profile，
将summary.txt内容更新至error.txt。results/和third_party/被Git忽略，删除仓库前须另行保存未上传结果。
