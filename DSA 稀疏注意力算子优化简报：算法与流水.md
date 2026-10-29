# DSA 稀疏注意力算子优化简报：算法与流水

> 对应 PR：[tile-ai/tilelang#3322](https://github.com/tile-ai/tilelang/pull/3322)
> 代码：`examples/ascend/sparse_attention/example_dsa.py`（行号以 PR head `4c0cb5f` 为准）
> 本文只覆盖算法与流水两部分。nd2nz copy lowering、模板内部实现等细节由负责模板的同事展开。

## 1. 算子与分块

### 1.1 算子本质

这个算子就是一层稀疏选择 + 标准 FlashAttention：

- Q：`[B, S_q, 64, 512]`，BF16；KV：`[B, kv_len, 512]`，BF16，64 个 query head 共享同一份 KV（shared KV / MQA）。
- 每个 query 预先拿到 `top_k` 个 KV 行下标。对每个 query，只对这 `top_k` 行做 `QK^T → online softmax → PV`，累加保持 FP32，最终输出 FP16。
- 因此它和普通 FA 的唯一本质区别是 **KV 的遍历方式**：从“连续 KV block”变成“按 index gather 出来的稀疏 block”。softmax 的 running max `m`、running sum `l`、以及 `alpha = exp(m_old - m_new)` 的 rescale 逻辑完全保留，只是外层循环从 `kv_len / block_kv` 变成了 `top_k / 128`。

索引采用 valid-prefix 约定：每个 query 的前 `count = min(q + 1, kv_len, top_k)` 个下标有效，其余位置填 `-1`；kernel 不检查 `-1`，也不做因果判断，`count` 直接由 query 位置推出。`top_k` 是编译期参数。

### 1.2 写死的分块方案

| 维度 | 分块 | 说明 |
| --- | --- | --- |
| head | 64 head / tile | 本算子一次处理全部 64 个 head；两个 AIV 各分 32 个 |
| head_dim | 512，不切 | Q 的 64 个 head 共享同一个 512 维的 KV head |
| topk | 128 / tile | `top_k` 必须是 128 的倍数；V1 的 VF 位宽决定（见 2.2） |
| batch × S_q | 展平后按物理核数分配 | `num_blocks = cube_core_num`（28 或 32）；`query = task * num_blocks + core_id`，按核轮转；要求 `seq_len % num_blocks == 0` |
| 流水 | 单层 `T.Pipelined`，`num_stages=2` | query 外层与 topk 内层合并成一层（见 3.2） |

这些分块不是调参出来的：head=64 由算子形状（64 个 query head 共享一份 KV）固定，topk=128 由 V1 的 VF 位宽固定，核数在运行时从设备查询。

片上 tile 大小：`Q_l1` 64×512、`KV_l1` 128×512、`P_l1` 64×128、`accum_l0c` 64×512（FP32）；KV 的 UB 暂存按 16 行一个 chunk（`KV_ub` 16×512，`KV_nz_ub` 17×512）。

### 1.3 V0–C1–V1–C2–V2 五个阶段

| 阶段 | 引擎 | 工作 | 关键 buffer |
| --- | --- | --- | --- |
| **V0** sparse load | AIV（MTE2 / VF） | 读 index；按 index gather 128 行 KV（每个 AIV 64 行，16 行一 chunk，两行一条 DMA）；ND→NZ；UB→L1 | `indices_ub`、`KV_ub`、`KV_nz_ub`、`KV_l1` |
| **C1** QK | Cube | `Q_l1(64×512) × KV_l1^T(512×128)` → L0C 64×128 | `Q_l1`、`KV_l1`、`accum_l0c` |
| **V1** online softmax | AIV（VF） | L0C→`S_ub`；scale、非法位置 mask、row max、exp、求和；cast 成 BF16 并直接按 NZ 写入 `P_nz_ub`；`P_nz_ub → P_l1` | `S_ub`、`P_nz_ub`、`P_l1`、`m/l/alpha` |
| **C2** PV | Cube | `P_l1(64×128) × KV_l1(128×512)` → L0C 64×512 | `P_l1`、`KV_l1`、`accum_l0c` |
| **V2** update O | AIV（VF / MTE3） | L0C→`O_tmp`（首块直接进 `O_ub`）；`O = O·alpha + O_tmp`；最后一个 block 做 `/l`、cast FP16、写 GM | `O_ub`、`O_tmp_ub`、`l_ub` |

Q 的加载只在每个 query 的 `block_id == 0` 发生一次，放在 V0 里顺带完成。

---

## 2. 算法优化

### 2.1 Sparse load 双发：两条 KV 行合成一条 DMA

对应 `example_dsa.py:122-139`：

```python
for pair in T.Serial(valid_rows // 2):
    index0 = T.int32(indices_ub[row_start + 2 * pair])
    index1 = T.int32(indices_ub[row_start + 2 * pair + 1])
    ...
    if index0 == index1:
        T.copy(KV[batch_id, index0, :], KV_ub[2 * pair, :])
        T.copy(KV[batch_id, index1, :], KV_ub[2 * pair + 1, :])
    else:
        first_index = T.min(index0, index1)
        row_stride = T.max(index0, index1) - first_index
        _copy_kv_pair(
            T.access_ptr(KV_ub[2 * pair, 0], "w"),
            T.access_ptr(KV[batch_id, first_index, 0], "r"),
            row_stride,
        )
```

gather 的朴素写法是“一个 index 一条 GM→UB 的 DMA”。这里把下标两两配对，当两行不同址时，用一条 2D DMA 同时搬运两行：

- `n_rows = 2`，`row_bytes = HEAD_DIM * 2 = 1024 B`；
- 源地址取两行中较小的下标，源 stride 为 `(max - min) * 1024 B`，目的 stride 为 1024 B（两行在 UB 中连续）。

于是每个 AIV 的 64 行通常从 64 条单行 DMA 降到 32 条双行 DMA。下标仍然是任意的，不要求有序或相邻；减少的是 MTE2 的指令条数和每次拷贝的启动开销，搬运的总字节数不变。`index0 == index1` 时两行同址，代码直接退回两条单行拷贝，不依赖 stride=0 的 2D 描述；`valid_rows` 为奇数时最后一行也单独处理。

### 2.2 V1 的 softmax VF：位宽变换的 interleave/deinterleave，以及融合的 nd2nz

对应 `example_dsa.py:186-206`：

```python
even_scores, odd_scores = S.vld2(S_ub[head, 0], "DINTLV_B32")
even_prob = S.vexpdif(even_scores, row_max, fp32_mask)
odd_prob = S.vexpdif(odd_scores, row_max, fp32_mask)
sum_acc = S.vadd(even_prob, odd_prob, fp32_mask)
even_bf16 = S.vcvt(even_prob, dtype, fp32_mask, sat=False, part=0)
odd_bf16 = S.vcvt(odd_prob, dtype, fp32_mask, sat=False, part=1)
packed = S.vor(
    T.reinterpret(even_bf16, "uint16x128"),
    T.reinterpret(odd_bf16, "uint16x128"),
    bf16_mask,
)
p_nz_ptr = S.vsstb(
    T.reinterpret(packed, "bfloat16x128"),
    p_nz_ptr,
    nz_stride,
    bf16_mask,
    update=True,
)
```

**为什么必须 deinterleave 再 interleave。** FP32 向量是 64 lane，BF16 向量是 128 lane；一个 head 的 128 个 topk score 恰好是“两条 FP32 向量 → 一条 BF16 向量”。但硬件 narrowing cast 的 `part=0/1`（`PART_EVEN` / `PART_ODD`）不是把 64 个结果压到连续的低半区，而是分别落到 128 lane 的偶数/奇数 16-bit 位上。所以：

1. 先用 `vld2(..., "DINTLV_B32")` 把 128 个 FP32 按偶/奇拆成两组（deinterleave）；
2. 偶数组用 `vcvt(..., part=0)`、奇数组用 `vcvt(..., part=1)`，结果正好落回偶数/奇数位；
3. 再用 `vor` 合并成一条连续的 128-lane BF16（interleave）。

这个顺序不是可选的技巧，而是由 cast 的 lane 映射决定的；如果直接按连续 64 个元素做 cast，概率会整体错位。

**这锁定了 topk 分块必须是 128。** 一轮 VF 处理覆盖“一个 head × 128 个 topk 位置”，正好填满 128 个 BF16 lane 和两条 64-lane FP32 向量。分块小于 128 会空 lane，不是 128 的倍数则要额外拼接，都会浪费 VF 的吞吐。所以代码里直接断言 `BLOCK_TOP_K == 128`、`top_k % BLOCK_TOP_K == 0`。

**集成的 nd2nz。** 最后一步 `vsstb` 用 NZ 的 stride 直接写到 `P_nz_ub`（compact NZ layout），相当于把 ND→NZ 的 scatter 融进了 softmax VF：

- 省掉一块与 `P_nz_ub` 同尺寸的 ND 布局 UB 暂存；
- 省掉一次独立的 VF/scatter 启动和它与 softmax 之间的同步；
- VF 结束后的 `P_nz_ub` 已经是 NZ，可以直接 `UB→L1` 给 C2 使用。

**多申请的一行 NZ UB。** `P_nz_ub` 分配成 `(HEADS_PER_AIV + 1, BLOCK_TOP_K)`，即 33 行而不是 32 行；`KV_nz_ub` 同理是 17 行而不是 16 行。有效数据仍然只写前 32/16 行，多出的一行只改变 C0 group 的 pitch（33/17 是奇数），让相邻 group 落到不同 bank，避免 NZ scatter/读取时的 UB bank 冲突。这是很常见的 padding 手法，nd2nz 模板 lowering 里新建的 scratch 也是 `scratch_pitch = rows + 1`（见 `src/ascend/transform/insert_nd2nz.cc:433-435`），幻方的 nd2nz 模板中同样包含这一功能；模板细节不在这里展开。

### 2.3 因果掩码的块级消除（未包含在本 PR）

当因果掩码按**右下角对齐**时（`q_len ≤ kv_len` 的 chunked prefill / decode 的常见约定），第 `i` 个 query 能看到的 key 上界是 `kv_len - S_q + i`。把 score 矩阵按 query block × KV block 分块后，严格位于这条对角线右上方的 block 是整块无效的：

- 这些 block 不需要算 QK，也不需要算 PV，可以直接从 KV block 循环里去掉（循环上界随 query block 递减）；
- 对 online softmax 来说，整块 mask 的贡献本来就是 0，跳过它不会改变 `m/l`，反而避免了 `-inf` 参与 `alpha = exp(m_old - m_new)` 的数值问题；
- 方阵时这块无效区域正好是 `0.5 · S_q · S_kv`，即最多省掉一半的 score/PV 计算；非方阵时是 `0.5 · S_q²`，同样可以整块裁掉。

这是**块级循环裁剪**，不是逐元素 mask 的加速；边界上跨对角线的 block 仍然要逐元素 mask。它没有出现在本 PR 中：本 PR 的稀疏路径用 TopK 索引约定把候选限制在有效前缀内，本身就没有显式因果 mask。这条优化针对的是仍按 `S_q × S_kv` 分块计算 score 的因果 mask 路径。

---

## 3. 流水优化

V0–C1–V1–C2–V2 的具体 stage offset、各 pipe 的起止时刻由 AutoSchedule 求解，人不需要手写流水。人能决定的是“它能不能排得起来”，具体是三件事：给流水链路足够的 buffer 版本、给调度器一层足够长且连续的循环、把依赖关系按真实计算块写清楚。下面三点分别对应这三件事。

### 3.1 要做流水，链路上的 buffer 都要多版本

对应 `example_dsa.py:283-284`：

```python
KV_ub: 2,
KV_nz_ub: 2,
```

V0 的搬运链是 `KV_ub --nd2nz--> KV_nz_ub --UB→L1--> KV_l1`，随后 `KV_l1` 同时被 C1（QK）和 C2（PV）消费。要让流水真正叠起来，这条链上的每一级都必须有独立 slot：

- `KV_ub` 只有 1 份：下一轮 gather 写 `KV_ub` 前，必须等上一轮 nd2nz 读完，V0 内部先串行；
- `KV_nz_ub` 只有 1 份：下一轮 scatter 写 NZ 前，必须等上一轮 `UB→L1` 读完，同样串行；
- `KV_l1` 只有 1 份：下一轮 V0 写 L1 前，必须等本轮 C1/C2 读完，下一轮 V0 和本轮 Cube 完全无法重叠。

任何一级是单版本，AutoSchedule 都只能在该级插入同步，整条 V0 链退化成串行，后面的 C1/V1/C2/V2 也就无从重叠。所以这里的做法不是“让调度器自己猜”，而是显式把整条链钉死：`KV_ub: 2`、`KV_nz_ub: 2`、`KV_l1: 3`（它从 V0 一直活到 C2，被两个 Cube stage 消费，因此给 3 份），`indices_ub: 2`、`S_ub: 2`、`P_nz_ub: 2`、`P_l1: 2`、`accum_l0c: 2`。

反过来，算法上就是循环携带的量保持单版本：`O_ub` 是跨 topk block 原地累加的输出，`m_ub/l_ub` 是跨 block 的 running state，强行多版本反而不符合语义。

需要强调 nd2nz copy lowering 的影响：`UB→L1` 的 NZ copy 在 lowering 后不是一条普通 DMA，而是 scatter（必要时还有 post-copy）等 task，这些 task 同样参与 AutoSchedule 的依赖分析。只把最终 `KV_l1` 或只把某一块 scratch 做成多版本，并不能让 lowering 出来的搬运跨迭代重叠；必须让 source staging、NZ staging、L1 tile 都具备独立 slot，流水才排得起来。具体 lowering 细节由模板同事说明。

### 3.2 循环合并：让 query 外层也进入流水

对应 `example_dsa.py:295-299`：

```python
for task_block in T.Pipelined(
    num_tasks * NUM_TOPK_BLOCKS,
    num_stages=PIPELINE_STAGES,
    annotations={"enable_offset": True},
):
    task = task_block // NUM_TOPK_BLOCKS
    block_id = task_block % NUM_TOPK_BLOCKS
    query = task * NUM_BLOCKS + core_id
```

如果把自然的两层循环（query 外层、topk 内层）直接交给 AutoSchedule：

```python
for task in ...:                 # query 外层
    for block_id in ...:         # topk 内层
        ...
```

这是受 AutoSchedule 能力制约的：它只为当前这一层循环体排流水，嵌套的外层循环不会和内层一起被拉开。如果按通常写法把 `T.Pipelined` 放在 topk 内层、query 放在外层，那么参与流水的只有内层循环体；外层 query 循环仍然是“跑完一个 query 的整条流水，再启动下一个 query”。每个 query 都要付一次 prologue/drain，query 越多、`NUM_TOPK_BLOCKS` 越小，这笔开销占比越大；当 `top_k == 128` 时内层只有一次迭代，这种写法下等于完全没有流水。

合并成一层 `num_tasks * NUM_TOPK_BLOCKS` 的循环后，query 边界退化成普通的迭代边界：query `i` 的最后一个 topk block 的 V2，可以和 query `i+1` 第一个 block 的 V0 重叠；`enable_offset` 的 stage offset 也能跨 query 连续生效，而不是每个 query 重置。代价是循环体里多了 `//` 和 `%`，但两者除数都是编译期常量，`query = task * NUM_BLOCKS + core_id` 对 `task_block` 仍然是仿射的，索引分析可以处理。

### 3.3 计算块划分：初始化要和它的消费者放在一起

对应 `example_dsa.py:311-314`（循环头现在只剩一项初始化）：

```python
if block_id == 0:
    T.copy(Q[batch_id, query_id, :, :], Q_l1, l2_cache_ctrl=4)
    with T.SimdVF():
        T.fill(KV_ub, T.float32(0.0))
```

以及 `example_dsa.py:324-333`（m/l 的初始化被挪到了 softmax 之前）：

```python
if NUM_TOPK_BLOCKS > 1:
    if block_id == 0:
        with T.SimdVF(latency=445):
            ...
            S.vsts(m_ub[0], initial_max, fp32_mask, "NORM_B32")
            S.vsts(l_ub[0], initial_sum, fp32_mask, "NORM_B32")
softmax_pack(...)
```

“计算块”指的是 AutoSchedule 看到的一段连续 task/语句。它按源码顺序建立 task 之间的依赖，并保持同一 pipe 上的发射顺序；它只调整各 task 的起止时刻，不会重新划分计算块，也不会把一段提前写下的初始化重新组织到使用点旁边。因此**初始化写在哪里，就会在哪里形成依赖边和同步点**。

早期版本把所有 buffer 初始化集中放在循环头（大致就是现在 313-314 的位置）。这样做的问题：

- 初始化本身是 producer。把它放在循环头，等于把被初始化 buffer 的 live range 拉长到整个循环体；它和上一轮结尾对该 buffer 的写构成 WAW/WAR，同步只能插在循环头；
- 于是本轮 V0 的 sparse load / C1 要等上一轮 V2 收尾，形成**本不该存在的等待关系**；
- 初始化离消费者很远，中间的 V0/C1 也无法和它重叠，流水被这段假依赖截断。

现在的划分方式是：

- 循环头只保留 V0 真正需要的一项：`T.fill(KV_ub, 0)`，它在 gather 之前，属于同一条 V0 链，放在这里是合理的；
- `m/l` 的初始化只在 `block_id == 0` 做，并且紧贴在 `softmax_pack` 之前，依赖只落在 V1 内部；
- `O_ub` 干脆不初始化：第一个 block 直接把 PV 结果 `dual_copy` 进 `O_ub`，后续 block 才做 `O = O·alpha + O_tmp`。这样就不存在“清零 O”与上一轮 V2 之间的 WAW。

原则就是：**有数据依赖的初始化、以及它的消费者，划分在同一个计算块内；不要在还不需要的时候就提前初始化。** 块内是一条真实的依赖链，块与块之间尽量独立，AutoSchedule 才能把 V0 的搬运、V1 的 VF、Cube 的 C1/C2 叠到不同 pipe 上。

---

## 4. 小结

- 算法上，这个算子 = attention + 按 index gather 的稀疏 KV block；三个关键点是：两行合一条 DMA 的 sparse load、V1 中为位宽变换服务的 interleave/deinterleave（并锁定 topk=128）、以及融合进 VF 的 nd2nz 与它的 anti-bank-conflict padding。因果 mask 的块级裁剪是后续可用的算法优化，不在本 PR 中。
- 流水上，V0–C1–V1–C2–V2 的排布大部分交给 AutoSchedule；人能决定的是它能不能排：链路 buffer 全部多版本（3.1）、把 query/topk 合并成一层可流水循环（3.2）、按真实依赖划分计算块并把初始化放到使用点旁边（3.3）。
- 分块是这些优化的共同前提：head 64、topk 128、`batch × S_q` 按 28/32 个物理核轮转；query 维度不是单独的串行外层，而是合并进同一条流水循环。

> 效果参考（PR 数据，B=1、64 head、head_dim=512、28 核 Ascend 950PR）：TopK=128 时相对 ops-transformer AscendC 约 1.95–2.13×，MFU 46–56%。

