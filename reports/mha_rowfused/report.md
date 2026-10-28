# Q128逐行融合结果：不采用

单head、4096x128、BF16输入/输出，原生AscendC复用metadata。
三种子检查及三次计时前检查，共36项精度通过；候选与基线在已测输入上逐元素相同。

| 实现 | API事件耗时：3进程中位数 μs |
|---|---:|
| 原始Q128 | 52.666841 |
| 逐行融合 | 61.969790 |
| 原生AscendC | 40.029650 |

候选比原始TL慢17.66%，不采用。最佳已测候选仍为nostore，约1%收益。
没有采集本候选设备profile，不把API退化归因于某一管线。
逐行统计更新的指令开销、寄存器压力、调度变化均是待查假设。
下一步导出三版编译后device C++，检查实际生成的VF与指令结构，再决定是否需要汇编/寄存器报告。
C++源码本身不能证明spill或物理寄存器数量。

证据：https://github.com/zhangwenhao91/Q4/blob/e4f03a7/error.txt

服务器沿用当前环境，无需重建AscendC：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
OUT="results/mha-tuning/codegen-$(date +%Y%m%d-%H%M%S)-$$"
python operators/mha/tuning/export_codegen.py --output "$OUT"
cat "$OUT/codegen.txt"
```

将codegen.txt内容反馈到error.txt。该步骤只编译/导出，不执行kernel或计时。
