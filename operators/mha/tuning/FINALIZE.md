# 最终归一化与BF16转换融合

两阶段nostore设备均值51.765us，三阶段52.079us，降低约0.60%，每侧30次。
本轮在nostore stages2上，仅改finalize_cast：FP32除法结果直接转换BF16并写入O_cast_ub，
省去归一化结果写回FP32 O_ub及下一VF重新读取。两段VF合为一段。
QK/softmax/PV和输出精度不改，保留原始BF16存储参数与O_cast_ub别名。
原有VF提示1780+193合为1973，仅为提示，不是测量值；实际调度可能变化。
保留递增行/列遍历，避免in-place压缩覆盖尚未读取的FP32数据。
本机验证别名存储顺序和源码范围，不代替NPU同步/精度验证。

四侧：原始stages3、当前最佳nostore stages2、融合候选、原生AscendC复用metadata。
3进程、各8轮、每轮100次，四侧在每个位置各出现2次；精度门限不变。

服务器现有zwh/CANN/q4fa环境，无需重建AscendC：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_finalize.sh check
```

全部通过后执行：

```bash
bash operators/mha/tuning/run_finalize.sh bench
```

任何编译或精度失败都停止，保留日志并反馈error.txt。计时确认收益后再采集profile。
比较matched_best_over_finalize，而不只看是否超过原始基线。

## API结果后复核

60项精度通过，API观察到小幅改善。下一步：

```bash
bash operators/mha/tuning/run_finalize.sh profile
```

自动打印kernel-details.txt，将输出反馈error.txt。无需重复check/bench或重建AscendC。
