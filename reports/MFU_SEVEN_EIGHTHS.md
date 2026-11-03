# 按 28/32 缩放输入的同轮 MFU 对照

已完成一轮36进程测试，结果见 [2026-11-03实测报告](mfu_20261103/report.md)。
以下保留实验设计说明，不表示仍未执行。

日期：2026-11-03。按用户及前辈建议准备；本机只有代码与CPU测试验证，尚无新NPU测量。

## 要验证的假设

MHA 原规格 S4096、Q tile128，共32个逻辑Q任务；S3584变为28个逻辑Q任务。
可检验任务数与28 AIC匹配时吞吐/MFU是否改善。
本次也同步缩短KV：KV循环从32次变28次，因此并非只改变调度任务数的单变量实验。
即使观测改善，也需要时间轴或进一步固定KV实验，才能严格归因于减少调度尾部。

mHC 采用 shifted 旧 pre、hidden5120 的 tilelang_coeff_bound 三kernel预绑定，
只将token N缩放为原来的7/8。N与每个kernel任务数并非必然成正比；
投影tile padding、apply分块、尾部token块与三个kernel自身调度均影响结果。
这是经验规模对照，不预先认定32→28是所有kernel的最优规模。

## 规模、FLOPs及边界

| 项目 | 原规模 | 新规模 | 新/原有用FLOPs |
|---|---|---|---:|
| MHA B1/H1/D128，Q/K/V长度 | 4096 | 3584 | 49/64 = 0.765625 |
| mHC shifted H5120，N | 64/128/256/512/8192 | 56/112/224/448/7168 | 7/8 = 0.875 |

MHA有用FLOPs为4*S²*128；新规模6,576,668,672，原规模8,589,934,592。
mHC有用FLOPs为2*N*20480*24。分母均为完整API event延迟，峰值仍用28 AIC
Cube BF16 378 TFLOPS / HF32按TF32档189 TFLOPS；不再按7/8缩放峰值。
完整BF16输出、非因果语义、scale、metadata复用和既有精度门限保持一致。
输入CPU参考、原生metadata、TL编译和输出view都使用实际S，禁止静默pad回4096。

MHA比较原始TL、KV2、S2、原生AC；mHC只计用户指定三kernel预绑定。
N56有token尾块；投影padding和尾部kernel行为必须重新验证，不能沿用历史精度结论。

## 服务器执行

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
bash operators/mfu/run.sh all both
```

共12种规格/实现集合、每种3进程（36个子进程）。MHA每进程8轮、mHC9轮，
每轮100次、预热100次。每个子进程先执行CPU参考精度检查，mHC另验投影/平方和。
形状顺序在第2次重复反转；不是同进程双形状配对采样，剩余设备状态影响由3进程分布评估。
脚本保留每个过程日志、实际shape、源与编译哈希、硬件信息、逐轮时间。

无需重建已装原生算子，首次新shape的TL JIT在计时外完成。
将最终表格或失败日志更新至error.txt，并保留MFU OUTPUT目录。

## 结果判读

同时看延迟、有效TFLOPS、MFU与进程间波动。
例如MHA延迟下降20%，但计算量下降23.4375%，则吞吐反而下降，不能称为效率提升。
新旧shape结果是不同工作量，不能只以旧延迟/新延迟称作算法加速比。
以新结果同轮原规模对照为主，历史结果只作背景，避免混入日前的N512波动。
