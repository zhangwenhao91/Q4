# 小N shifted mHC融合实验

目标：改善N=512完整API约79.54us而适配AscendC约40.87us的差距。
本实验不修改已验证baseline、已安装TileLang/DeepGEMM或原生AscendC包。

## 四侧同进程比较

| 标签 | 路径 | 预期kernel数/API |
|---|---|---:|
| deepgemm_tilelang | 原始apply、投影、RMS、split、Sinkhorn | 5 |
| tilelang_coeff_fused | 原始apply、投影、融合RMS/split/Sinkhorn | 3 |
| tilelang_apply_fused | 投影、融合系数尾部及旧pre apply | 2 |
| ascendc_shifted_native | 已验证MhcShiftedPreV2、Sinkhorn | 2 |

两个候选衍生自固定TileKernels融合实现，来源和哈希见source.json，MIT许可随目录保留。
系数部分的数值计算顺序沿用上游，仍为1e-20 RMS eps、1e-6 pre/Sinkhorn eps、
20轮Sinkhorn、post乘数2。是否与分解路径数值一致须设备验精度，不提前宣称bitwise一致。

apply_tail.py通过独立prev_ub读取旧pre，pre_ub计算的新pre单独输出。
coeff_tail.py完全不处理特征apply。完整token块使用普通DMA，尾块通过有界SIMT加载，
将无效token显式置零；没有将DMA行宽pad_value用于动态token尾块。输出边界由动态shape约束；
非16倍数N需要设备检查通过才能宣称支持。
初始token tile=16、缓冲版本2、融合apply hidden tile=256，暂不同时扫描参数。

候选每次重新分配BF16 layer_input和FP32系数slab；slab划分为互不重叠的
pre/post/comb输出。每组起点32字节对齐，奇数N在pre/post段末尾保留必要补齐，
返回view仍严格为4N/4N/16N个元素，不使用跨调用预分配。
加上投影的两个输出，每次共4次张量分配；原始TL保持既有路径，约10次。
本候选同时包含kernel融合和分配减少，不能将全部收益归于算术或单项改动。
投影、平方和、apply、归一化均在每次计时重新执行，输入生成和CPU参考在计时外。

## 服务器执行

无需重建AscendC vendor，沿用已验证q4mhcmatch包与DeepGEMM兼容补丁。

```bash
conda activate zwh
cd /home/z30086261/Q4
git pull --ff-only
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
bash operators/mhc/tuning/run_fusion.sh smoke
```

smoke先在N=512检查四侧对CPU及六组两两比较，首次调用会编译新TL kernel。
失败即停止，将打印出的runner.log完整报错复制到error.txt。
smoke通过后执行完整精度检查：

```bash
bash operators/mhc/tuning/run_fusion.sh check
```

check：N=512三种子；N=1/17/127/128/129/513；N=129全零残差/全零旧pre/单流旧pre。
12个case，每个40项输出检查及2项原始投影检查，预计480+24项。
阈值沿用baseline，失败不放宽。MHC_WEIGHTS可选，格式与matched实验相同；默认合成权重。

全部通过后：

```bash
bash operators/mhc/tuning/run_fusion.sh bench
bash operators/mhc/tuning/run_fusion.sh profile
```

bench：N=128/512/8192各3个独立进程，预热100次、16轮各100次。
四侧轮换起始位置并反转顺序，16轮覆盖两次完整位置/顺序周期。
每个case先检查全部四侧精度。以API event/wall判断收益，设备Duration单列。
profile：N=512/8192各四侧，每侧30次目标调用；核对全部kernel，不能只统计融合尾部。
不将kernel Duration合计直接当作存在间隙/重叠时的API延迟。

每次结果在results/mhc-fusion的新目录，末尾summary.txt可直接复制到远端error.txt。
如分步跑完，可一次离线汇总：

```bash
python operators/mhc/matched/collect_results.py --root results/mhc-fusion
```

本机只有CPU合约、源码及脚本检查；NPU编译、精度与性能均待验证，不预设候选一定更快。

已测融合结果见 [reports/mhc_fusion/report.md](../../../reports/mhc_fusion/report.md)。
小N后续的预绑定调用实验见 [BOUND.md](BOUND.md)，保持两kernel的数值源码不变。
