# 从DSA和Ascend示例学习mHC流水

研究日期2026-11-02，参考tile-ai/tilelang固定提交
b5cb0263361c0ab21e39b2fc3bc8568e1c1971e3；服务器运行的是较早的a7cb753d，
不能默认最新API全部可用。本次为源码研究，未新增NPU性能结论。

## DSA手动流水

[DeepSeek V3.2说明](https://github.com/tile-ai/tilelang/blob/b5cb0263361c0ab21e39b2fc3bc8568e1c1971e3/examples/deepseek_v32/README.md)
包含稀疏MLA与indexer等DSA相关实现。
[手工流水版](https://github.com/tile-ai/tilelang/blob/b5cb0263361c0ab21e39b2fc3bc8568e1c1971e3/examples/deepseek_v32/sparse_mla_fwd_pipelined.py#L100)
分配bar_k_ready/bar_k_free，每个缓冲槽各有就绪和释放状态。
124行附近消费者等待ready，完成读取后通知free；198行附近生产者先等free，
209行用ptx_cp_async搬运，222行通知ready。索引stage和phase控制环形缓冲复用。
可学习的是生产/消费依赖与槽位生命周期，不能直接复制CUDA的PTX、TMA或warp机制到Ascend。

## Ascend自动及显式阶段

[向量加示例](https://github.com/tile-ai/tilelang/blob/b5cb0263361c0ab21e39b2fc3bc8568e1c1971e3/examples/ascend/example_vecadd.py#L25)
使用多版本UB与一个长T.Pipelined循环，循环内搬入、计算、写回。
该模式比仅在小内循环上增加stage数更适合保持稳态流水。

[Ascend FlashMLA](https://github.com/tile-ai/tilelang/blob/b5cb0263361c0ab21e39b2fc3bc8568e1c1971e3/examples/ascend/flashmla/kernel.py#L454)
把job和其内部块展平为step：job=step//nb，k=step%nb，
通过enable_offset=True与T.Stage组织不同逻辑阶段，允许跨job推进。
496/523行附近显示不同阶段的转换和矩阵计算；不能把源码顺序直接视为硬件完成顺序。

[软件流水文档](https://github.com/tile-ai/tilelang/blob/b5cb0263361c0ab21e39b2fc3bc8568e1c1971e3/docs/programming_guides/software_pipeline.md)
还说明stage/order描述可执行语句，而不是简单给每条Python标量赋值编号。
当前明确可用的循环名字是T.Pipelined，不能未经验证将其改写成T.Pipeline。

## 对上一轮mHC候选的启示

上一轮结构为Persistent(token)里面再Pipelined(hidden_tile)。
每token只有4或5个大块，内部流水可能频繁启动/排空；三阶段比单阶段更快，
但仍未稳定胜过原三kernel，现有error没有时间轴证明具体重叠。
这是结构分析线索，不是已证实的性能瓶颈。

下一候选优先尝试一个连续循环跨token推进：

```text
job = step // hidden_tiles
hidden_tile = step % hidden_tiles
token = core_id + job * num_cores

搬入阶段：x[token, :, hidden_tile]、旧pre -> 输入槽
计算阶段：读已就绪输入槽 -> 输出槽
写回阶段：写已完成输出槽 -> GM
```

每token仍由同一核拥有，基线仍为原H5120三kernel；不采用前轮任务级H1280分派。
先保留自动流水版本，再在服务器确认接口支持后研究显式阶段或手工事件。
token尾数需要边界保护，不能让无效job读写；跨token复用旧pre必须与输入槽版本一致。
输入槽重用前须等待向量读取结束，输出槽重用前须等待MTE3写回结束。
不能为了减少等待删除真实依赖，避免覆盖未消费的数据。

先比较连续循环stages1/3和上一轮嵌套循环，区分展平与stage收益。
同时检查生成代码中的事件wait位置、缓冲槽选择、提前load、最后一次写回完成，
再看流水图中MTE2/Vector/MTE3是否重叠。只有apply Duration与完整API都改善，
并通过旧pre四输出精度校验，才采用候选。暂不升级服务器TileLang或改默认算子。
