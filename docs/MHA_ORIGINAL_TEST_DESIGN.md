# 【A5】【TileLang】原版MHA算子实践

## 测试方法

### 1. 配套环境要求

1. 目标设备：**32核Ascend 950DT服务器**，32个AIC、64个AIV；测试前读取实际设备型号和核数确认。
2. CANN：9.3.0，具体安装版本及路径以目标950DT服务器部署为准。
3. Python：3.12，参考已有Conda环境`zwh`。
4. torch_npu：2.13.0。
5. torch：2.13.0+cpu。
6. Ascend TileLang：0.1.15+cpu.gita7cb753d。
7. 编译工具：Conda GCC 13，供需要编译的扩展使用。

以上软件版本为Q4现有参考配置，在目标950DT上的安装、导入及兼容性需核验，尚无本配置的设备测试结果。
记录实际设备、CANN、Python包、编译器及运行代码提交；设备编号按目标机器选择。

### 2. 测试算子规格

MHA规格严格使用仓库[规格表](../MHA和mHC规格表)，构建kernel时计算属性如下。

| 计算属性 | 静态或动态 | 取值 |
|---|---|---|
| batch | 静态 | 1 |
| heads | 静态 | 1 |
| SEQ_LEN，Q/K/V序列长度 | 静态 | **4096** |
| head_dim，注意力头维度 | 静态 | 128 |
| scaling，注意力分数缩放系数 | 静态 | 1/sqrt(128)，约0.08838835 |
| out_dtype，输出数据类型 | 静态 | **float32** |
| block_q / block_kv | 静态 | 128 / 128，固定上游默认值 |
| num_stages | 静态 | 3，固定上游默认值 |
| num_blocks，逻辑Q任务数 | 静态推导 | 4096/128=32，使用原版默认推导 |
| causal，因果屏蔽 | 静态 | False |
| dropout，随机丢弃概率 | 静态 | 0 |
| attention bias / LSE输出 | 静态 | 无 / 不返回 |

运行时动态shape：无；Q/K/V数据内容为运行时输入。

使用kernel的输入输出：

```text
input:
Q  shape:[SEQ_LEN,head_dim] = [4096,128]  dtype:BF16
K  shape:[SEQ_LEN,head_dim] = [4096,128]  dtype:BF16
V  shape:[SEQ_LEN,head_dim] = [4096,128]  dtype:BF16

output:
O  shape:[SEQ_LEN,head_dim] = [4096,128]  dtype:FP32
```

计算为`O = softmax(Q @ K.T / sqrt(128)) @ V`，覆盖注意力核心，QKV投影与输出投影不在本次范围。
使用固定上游`example_mha.py::mha`调用`core.py::flash_attention_fwd`，内核直接输出FP32。
使用原版默认tiling，编译fast-math沿用上游示例；本设计不缩短序列、不改为BF16输出。
32个逻辑Q任务与32个AIC数量相同，但实际物理调度及利用率仍以设备采集为准。

### 3. 算子性能

性能目标：**具备同规格AscendC实现后，AC/TL大于1.0**，表示TileLang完整调用更快。
`AC/TL = AscendC API event延迟 / TileLang API event延迟`。

**实现与结果状态**

| 项目 | 当前状态 |
|---|---|
| 原版TL，BF16输入、FP32输出、S4096 | 已有严格规格测试入口，目标32核950DT上待验证 |
| 同规格原生AC | 当前固定FlashAttn路径的BF16输入对应BF16输出，尚未匹配本规格FP32输出 |
| AC/TL加速比 | 同规格AC实现就绪后测量，当前不填数值 |
| 原版TL延迟、MFU | 目标32核950DT上待测 |

已有950PR 28核、S3584、BF16输出实验属于另一套设备与规格，不能作为本设计的实测结果。
不能将AC的BF16输出额外cast成FP32后标为原生同规格结果；参考值也保持FP32，不先舍入BF16。

**测试口径**

- 精度：CPU FP32分块参考；检查shape、dtype一致，输出finite，relative L2≤1%、max abs≤0.02。
- 执行：先冒烟，再以seed=42/43/44检查；精度失败则停止性能结论。
- 计时：3个独立进程，每进程预热100次、10轮、每轮100次；以完整API event延迟为主，wall为辅。
- 汇总：先取各进程轮次中位数，再取3进程中位数；attention与每次输出/临时分配计时，首次编译与输入准备排除。
- 对比：同规格AC就绪后，双方共享输入并交替执行，固定shape主口径复用metadata；每次生成metadata的口径另列。
- 设备profile：独立记录目标kernel的Duration与次数，不与API延迟直接相减作为精确主机开销。

MFU采用有效矩阵FLOPs与完整API event延迟：

```text
FLOPs = 4 * batch * heads * SEQ_LEN² * head_dim
      = 4 * 1 * 1 * 4096² * 128
      = 8,589,934,592

有效TFLOPS = FLOPs / event_us / 1e6
矩阵MFU(%) = 有效TFLOPS / 432 * 100
```

峰值分母对应**950DT 32 AIC的Cube BF16 432 TFLOPS**，见[官方白皮书表3-1](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)。
输出或累加为FP32不改变矩阵乘输入为BF16的口径；使用前核对目标设备和额定规格。
softmax等操作不计矩阵分子，完整耗时仍在分母；本指标不等于硬件忙碌比例。

### 4. 代码仓

项目仓库：[zhangwenhao91/Q4](https://github.com/zhangwenhao91/Q4)。

| 内容 | 位置 |
|---|---|
| MHA规格 | [MHA和mHC规格表](../MHA和mHC规格表) |
| 原版TL wrapper，S4096、FP32输出 | [operators/spec_v1/upstream_mha/example_mha.py](../operators/spec_v1/upstream_mha/example_mha.py) |
| 原版TL数值源码 | [operators/spec_v1/upstream_mha/core.py](../operators/spec_v1/upstream_mha/core.py) |
| 严格规格case | [operators/spec_v1/cases.py](../operators/spec_v1/cases.py)，mha_case |
| 公共精度与计时框架 | [operators/spec_v1/bench.py](../operators/spec_v1/bench.py)，--suite mha |
| 固定源码准备脚本 | [operators/spec_v1/prepare_mha.py](../operators/spec_v1/prepare_mha.py) |

源码固定：TileLang为`80364b3d9758854a9692cc0f14089025f54d3f3f`，来源tile-ai/tilelang。
当前AC源码参考为`ba5162411c919e15fbabb28298e8b420c11ad905`，来源cann/ops-transformer；
它的现有FlashAttn dtype边界不满足本次FP32输出规格，因此当前脚本只生成TL单侧数据。

**目标服务器执行方法**

先激活兼容环境并加载目标机器的CANN。`CANN_ENV_FILE`须预先设置为目标机实际`set_env.sh`路径：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
: "${CANN_ENV_FILE:?请设置为目标950DT上的CANN set_env.sh完整路径}"
source "$CANN_ENV_FILE"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export ASCEND_NPU_ARCH=dav-3510
python -c 'import torch_npu; p=torch_npu.npu.get_device_properties(0); print(p); assert "950DT" in p.name and p.cube_core_num==32 and p.vector_core_num==64'
python operators/spec_v1/prepare_mha.py
```

创建本轮独立结果目录，先做单种子冒烟：

```bash
MHA_RUN="results/mha-original-950dt32/$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$MHA_RUN"
python operators/spec_v1/bench.py --suite mha --check-only --output "$MHA_RUN/smoke"
```

冒烟通过后执行精度、计时和profile；在子shell中启用失败即停：

```bash
(
  set -e
  for seed in 42 43 44; do
    python operators/spec_v1/bench.py --suite mha --seed "$seed" \
      --check-only --output "$MHA_RUN/check-seed$seed"
  done
  for rep in 1 2 3; do
    python operators/spec_v1/bench.py --suite mha --warmup 100 --iters 100 --rounds 10 \
      --output "$MHA_RUN/bench-rep$rep"
  done
  python operators/spec_v1/bench.py --suite mha --profile tilelang --output "$MHA_RUN/profile"
)
```

当前入口会写出`comparison_status.json`，标记single_side_only；这是同规格AC尚未就绪的状态，不能据此生成双侧加速比。
以上是目标设备待执行指令，不代表已在32核950DT跑通。已有`run_plain3584.sh`及28核MFU峰值配置不适用于本设计。
交付时保留metadata、correctness、timings.csv、summary.json、comparison_status和profile，并记录环境与源码版本。
results/、third_party/被Git忽略，未上传产物需要另行保存。
