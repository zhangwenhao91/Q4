# 原版MHA：序列长度缩至7/8

用户确认：Q/K/V=[3584,128]，BF16；O=[3584,128]，BF16。
batch=1、heads=1、scale=1/sqrt(128)、causal=False、dropout=0。
只有序列长度从4096缩至3584；head_dim不变。输出BF16是相对原规格FP32的明确差异。

使用仓库已固定的tile-ai/tilelang提交80364b3d9758854a9692cc0f14089025f54d3f3f，
examples/ascend/flash_attention/core.py::flash_attention_fwd的默认tiling，源码逐文件哈希校验。
只传q_len/kv_len=3584、head_dim=128、out_dtype=bfloat16和既定scale。
编译fast-math开关沿用上游example_mha，未增加手工pipeline、分块、融合或候选调优。
不是追踪会变化的main HEAD，也不是原example_mha强制FP32输出的字节级相同调用。

AC使用固定ops-transformer FlashAttn，无PyTorch数值fallback或输出cast。
cached口径复用固定shape metadata；per_call口径把每次metadata生成计入AC耗时。
两口径分别报告，不混算。每次完整attention及输出/临时分配均计时，编译排除。
CPU分块FP32参考最后舍入BF16；同输入、逐侧校验和互验先通过再计时。
各口径三进程，每进程10轮双侧交替、100次预热、每轮100次，profile沿用30次active调用。

```bash
bash operators/mha/run_plain3584.sh smoke
bash operators/mha/run_plain3584.sh all
python operators/mha/collect_plain3584.py
```

运行前激活zwh并source CANN；脚本自动选择既有q4fa vendor。
本次不重建AC库或升级TileLang。mha_plain3584.py通过共同bench的mha_bf16 case入口注入
独立3584测试构建器，原4096实现与此前优化候选不变；metadata标明实际3584契约。
all预期16份正确性记录、6份API摘要、4个profile、48项输出检查。
结果位于results/mha-plain3584，最后汇总复制至error.txt即可。
两次矩阵乘主要计算量随序列长度平方变化，3584对应原4096的49/64，
不能把本轮耗时直接称为“原来7/8的计算量”。最终AC/TL以实测为准。
