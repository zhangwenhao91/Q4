# S3584：全部保留的 MHA 实现路径对照

当前按“全部保留且有对照价值的实现/优化阶段”解释用户的“所有场景”。
规格表及固定上游测试只定义B1/H1/D128、非因果这一组输入；本轮不另行推定batch/head列表。
Q/K/V序列均固定3584（原4096的7/8），BF16输入输出、scale=1/sqrt(128)、无dropout，
原生AC复用metadata。mHC继续挂起。

## 一轮覆盖

| 标签 | 路径 |
|---|---|
| tilelang_q128 | 原始TL，stages3 |
| tilelang_s2 | 原始TL，stages2 |
| tilelang_nostore | 减少softmax回写，stages3 |
| tilelang_nostore_s2 | 减少softmax回写，stages2 |
| tilelang_finalize | nostore + stages2 + 归一化/输出转换融合 |
| tilelang_kv2 | finalize + K/V双版本 |
| tilelang_s2buf | KV2 + S双版本，alpha三版本 |
| ascendc | 原生FlashAttn，cached metadata |

此前已删除的退化实验不恢复；缓冲生命周期的退化负对照保留在原诊断实验中，
本轮有效版本扫描不额外运行。具体数值kernel沿用已保留模块，新增统一构造/计时入口，
不为每个场景混用不同dtype或输入内容。

## 执行

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
python operators/mfu/run.py --operator mha --scale seven-eighths --all-mha-variants
```

只需3个独立计时进程，每进程同时加载8条路径，共享输入、CPU参考和AC基线。
每侧先对CPU参考验精度，再两两互验；每进程36项输出检查，3进程共108项。
每进程预热100次、16轮各100次，轮换位置并反转顺序，各实现各位置出现两次。
候选首次TL JIT在计时外完成；已安装AscendC包无需重建。

输出同规格AC/TL event与wall加速比、三进程配对AC/TL及胜场数、各实现有效TFLOPS和MFU。
AC/TL大于1表示TL更快；MFU仍用28 AIC Cube BF16峰值378TFLOPS。
分子4*3584²*128，分母完整API延迟；精度门限和metadata复用条件不变。

将终端表格或失败日志更新error.txt，并保留MFU OUTPUT目录。
本机无NPU，新增场景在S3584的实际编译、精度与性能仍需服务器验证。
本轮不声称覆盖可变batch、多head、其他head_dim、causal、GQA或反向；
这些需要另给规格或另选兼容kernel，不能在固定实现上直接改变标签后报告结果。
