# 小N预绑定三kernel实验

先验证设备并行度与额外启动成本的取舍，不同时改数值内核、token布局或split-K。
当前预绑定两kernel在N=512/8192已超过AC，但N=128约29.11us，AC约28.30us。

## 比较对象

| 标签 | 每次设备路径 | kernel数 |
|---|---|---:|
| tilelang_apply_bound | 投影/平方和 + 融合旧pre apply与系数尾部 | 2 |
| tilelang_coeff_bound | 独立旧pre apply + 投影/平方和 + 融合系数尾部 | 3 |
| ascendc_shifted_native | 已验证原生shifted Pre + Sinkhorn | 2 |

两个TL路径都在计时外使用公开compile接口绑定JITKernel，计时内每次分配四个张量，
所有设备计算重新执行，没有输出预分配或数值缓存。相同输入、CPU参考、HF32模式、
epsilon、20轮Sinkhorn及四项输出契约保持不变。
独立apply使用已验证上游mhc_pre_apply_mix_fwd_asc，hidden tile=5120，与原wrapper一致。
系数尾部沿用已验证coeff_tail.py，token tile=16、stage=2。
本实验只改变kernel组合和调用分发，不修改这些设备数值源码。

## 执行

在既有zwh/CANN环境中执行，无需重建AscendC或重装Python包：

```bash
cd /home/z30086261/Q4
git pull --ff-only
bash operators/mhc/tuning/run_split.sh smoke
```

smoke先检查N=128三侧对CPU参考和全部两两组合，预计24项输出和2项原始投影检查。
全部通过后：

```bash
bash operators/mhc/tuning/run_split.sh check &&
bash operators/mhc/tuning/run_split.sh bench &&
bash operators/mhc/tuning/run_split.sh profile
```

check共16个case：N=128/512各三种子，N=1/17/64/127/129/256/513，及N=129三种特殊输入。
预计384项输出检查与32项原始投影检查；门限不变，失败停止。
bench为N=64/128/256/512/8192各3进程，预热100次、12轮各100次；三侧均衡轮换顺序。
profile覆盖同样五个N的三侧，每侧30次目标调用，核对2/3/2个kernel。
先按完整API event/wall判断胜负，设备Duration用于解释；不用kernel合计代替API延迟。

结果位于results/mhc-split，分步跑完后可离线汇总：

```bash
python operators/mhc/matched/collect_results.py --root results/mhc-split
```

将汇总输出复制到error.txt即可。保留两kernel基线，不自动根据N切换实现；
待多形状实测后再决定是否有合适的分派阈值。
MHC_WEIGHTS仍可指定有效FP32 fn/scale/base；默认合成权重。
本机只有CPU调用合约与语法验证，设备精度和性能待服务器确认。

已完成服务器验证：1104项输出检查与92项原始投影检查通过；三kernel版在全部五个
已测N上超过AC。N=128/256/512三kernel稳定更快，N=8192两kernel稳定更快；
N=64两/三kernel胜负有波动。详见 [结果报告](../../../reports/mhc_split/report.md)。
