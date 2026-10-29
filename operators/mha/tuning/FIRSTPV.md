# 首个PV结果直接初始化O

采用DSA文档中的首块处理思路，直接对照已确认的最佳组合：
nostore + stages2 + 最终归一化/BF16转换融合 + K/V版本2；S/alpha仍为3版本。
Q搬运仍在KV循环内，不叠加此前无稳定收益的Q常驻实验。

当前所有KV块先写O_tmp，再执行O=alpha*O+O_tmp，循环前先清零O。
本轮首块PV直接dual_copy进O_ub，跳过首块累积和自动路径的O清零；
后续31块仍写O_tmp并按原公式累积。softmax及m/l初始化保持原样。
第一块在正常有限分数下alpha=0。不能只删清零而仍执行首块的旧O读取。

这只节省首块的部分工作，不能提前承诺提速。条件分支和新的FIX->O依赖可能改变调度。
保留原始pv宏和manual路径清零，实验只运行自动调度；自动路径是否正确同步需NPU验证。
完整生成C++及源码哈希会保存，终端CODEGEN列出PV的L0C->UB复制和KV循环位置。

规格仍为B1/H1/S4096/D128，BF16输入输出，原生AscendC复用metadata。
四侧：原始TL、最佳K/V双版本、新候选、原生AscendC。
3种子检查；3进程计时、各8轮、每轮100次，四侧轮换。
CPU参考及全部两两核对，精度门限保持不变。

## 服务器

沿用现有zwh/CANN/q4fa环境，无需重建AscendC：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_firstpv.sh check
```

全部检查通过后：

```bash
bash operators/mha/tuning/run_firstpv.sh bench
```

若编译或精度失败，停止并反馈完整日志；不要跳过失败配置。
将含CODEGEN行的输出更新到error.txt。若有API收益，再profile复核设备时间。
