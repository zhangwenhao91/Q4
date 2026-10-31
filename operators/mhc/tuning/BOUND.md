# 预绑定已编译kernel：减少主机分发开销

此前两kernel版N=512的API约45.00us，AC约40.86us；设备Duration合计分别约30.37/41.09us。
本实验保持同一投影kernel和同一融合apply尾部，只改变主机调用方式。

TileLang a7cb753d的JITImpl.__call__每次执行parse_args并查询kernel缓存；
其公开compile方法返回可直接接受运行时张量的JITKernel。
新候选在计时前编译并持有两个JITKernel，每次调用直接传运行时tensor。
不缓存任何投影结果、系数或输出，不使用跨调用预分配；每次仍分配四个张量。
完整块平方和避免多余slice view，尾块仍按真实N取view。
sample buffers只用于编译时的shape/dtype推导，不在后续执行中复用。

依据：固定版本tilelang/jit/__init__.py的compile、__call__；tilelang/jit/kernel.py的__call__。
未修改TileLang包、原生AscendC或融合kernel源码，不使用私有缓存接口。
metadata记录编译对象生成的设备源码哈希，profiler核对仍为2个kernel/API。

## 服务器

在已配置zwh/CANN的终端执行，无需重建任何原生包：

```bash
cd /home/z30086261/Q4
git pull --ff-only
bash operators/mhc/tuning/run_bound.sh smoke
```

smoke通过后依次执行：

```bash
bash operators/mhc/tuning/run_bound.sh check &&
bash operators/mhc/tuning/run_bound.sh bench &&
bash operators/mhc/tuning/run_bound.sh profile
```

三侧为已验证tilelang_apply_fused、新tilelang_apply_bound及ascendc_shifted_native。
check保持三种子、动态N及旧pre特殊输入，共12个case，每个24项输出检查。
bench为N=128/512/8192各3进程，12轮各100次，三侧均衡轮换位置和正反顺序。
profile为N=128/512/8192各三侧，每侧30次调用。其他计时和精度条件保持一致。
所有侧的首次编译与参考计算都在计时外，内部设备计算/分配/调用仍计时。

汇总已有结果：

```bash
python operators/mhc/matched/collect_results.py --root results/mhc-bound
```

本机仅验证CPU调用合约和语法；设备编译、数值与性能待服务器确认。
预绑定能否补上小N差距由API实测决定，不预设收益。

已测结果：N=512预绑定约30.36us、AC约40.87us；N=8192约544.14us对705.89us。
N=128约29.11us对28.30us，尚未超过AC。完整证据见
[预绑定结果报告](../../../reports/mhc_bound/report.md)。
