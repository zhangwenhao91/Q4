# 单次apply仿真：采集成功

2026-11-03，证据f0ebbb0工作树中的error.txt（本轮日志随此前提交推送），
目录results/mhc-simulator/20261103-093657-all-3856816。
Ascend950PR_9579，N128、hidden5120，目标apply调用一次，解析core0。
原版H5120、嵌套1280三阶段、连续1280三阶段均采集成功，
输出CPU校验passed=True，max_abs均0.015625。

| 实现 | core0.veccore0 duration(us) | core0.veccore1 duration(us) |
|---|---:|---:|
| 原版 | 4.84 | 4.72 |
| 嵌套流水 | 5.16 | 5.49 |
| 连续流水 | 4.86 | 4.94 |

仿真不是此前真机API测量，不能作为完整三kernel加速比。
三个版本均已产出trace.json及visualize_data.bin，支持进一步管线分析。
控制台缺少debug_line导致源码行映射为空，不表示指令流水采集失败，
但当前无法从控制台直接判定MTE2/Vector/MTE3重叠方式。
stars_log任务dump也不等于指令时间轴。

使用collect_simulator.py离线提取trace schema、lane标签、事件样例、
各lane区间并集及两两重叠，先验证哪些lane代表真实执行，避免将父级span误算为流水重叠。
默认实现不变，无需重跑NPU或仿真。完整trace/Insight时间轴仍是分析依据。
