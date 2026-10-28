# Q128流水线深度实验

规格保持B1/H1/S4096/D128，BF16输入输出，AscendC复用metadata。
本轮五侧同输入对比：

| 标签 | softmax | num_stages |
|---|---|---:|
| tilelang_q128 | 原始 | 3 |
| tilelang_nostore | 减少写回，当前最佳 | 3 |
| tilelang_s2 | 原始 | 2 |
| tilelang_nostore_s2 | 减少写回 | 2 |
| ascendc | 原生FlashAttn | 不适用 |

原始/nostore各自比较stages2与3，区分调度收益和softmax收益。
Q/KV分块均128、自动调度，其他计算宏不改；不叠加manual/FMA/逐行融合。
num_stages可能改变buffer版本和编译器调度，不代表实际重叠程度。
本轮不盲目增加至4，先避免更大buffer占用导致资源不足。
本机仅验证实验配置、五侧轮换顺序和Python源码，NPU编译/精度/性能尚未验证。

## 服务器

沿用当前zwh/CANN/q4fa环境，无需重建AscendC。先执行：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_stages.sh check
```

全部精度检查通过后执行：

```bash
bash operators/mha/tuning/run_stages.sh bench
```

3个种子，五侧分别对CPU参考及全部两两核对，每个进程15项精度记录。
计时3个进程、各10轮、每轮100次；每侧在每个执行位置出现2次。
所有比例均大于1才表示对应分母候选更快，需同时超过同轮nostore才能认定推进当前最佳。
若编译或精度失败就停止，保留日志，不跳过失败候选。
确认API收益后再采集profile/pipe。本轮不混入输出归一化/转换融合。

下一轮再单独测试输出归一化和BF16转换融合，以本轮结果选择相同的调度配置做对照。

## 本轮API结果与设备复核

90项精度通过。nostore stages2约51.903us，相对原始stages3降低1.45%，
相对nostore stages3降低0.42%。完整结果见reports/mha_stages/report.md。
额外收益较小，先比较nostore stages3、nostore stages2与AscendC设备计时：

```bash
bash operators/mha/tuning/run_stages.sh profile_best
```

该模式自动打印设备CSV，不需要重新运行check/bench。将输出反馈到error.txt。
