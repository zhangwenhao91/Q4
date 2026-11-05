# 【A5】【TileLang】原版MHA算子实践

## 测试方法

### 1. 配套环境要求

1. 目标设备：**32核Ascend 950DT服务器**。
2. CANN：9.3.0。
3. Python：3.12。
4. torch_npu：2.13.0。
5. torch：2.13.0+cpu。
6. Ascend TileLang：0.1.15。

### 2. 测试算子规格

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

### 3. 算子性能

性能目标：**具备同规格AscendC实现后，AC/TL大于1.0**。



### 4. 代码仓

项目仓库：https://github.com/tile-ai/tilelang/tree/main/examples/ascend/flash_attention。
