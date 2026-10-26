# mHC 反向性能覆盖

状态：本机完成接口核对、脚本与投影梯度公式检查；尚未在 NPU 验证或测得反向性能。延续已验证的上游提交及 DeepGEMM 两项兼容修复，不重新安装 DeepGEMM。

| suite | TileKernels 一侧 | AscendC 一侧 | 梯度 |
|---|---|---|---|
| post | mhc_post_bwd | MhcPostBackward | 单层输出、残差、post 权重、残差混合矩阵 |
| sinkhorn | mhc_sinkhorn_bwd（重算迭代） | MhcSinkhornBackward（读取原生前向缓存） | logits |
| pre | TL apply/split/RMS 反向 + PyTorch 投影梯度 | MhcPreBackwardV2，mode=1 HF32 | x、phi、alpha、bias |
| pre_sinkhorn | 上述链 + TL Sinkhorn 反向 | MhcPreSinkhornBackward | x、phi、alpha、bias |

Pre 两组标签为 `tilekernels_plus_torch_projection_grad`。当前锁定的 DeepGEMM 没有 HC 投影反向接口，不能宣称这是纯 TileLang 完整反向。计时包含两次 GEMM、平方和梯度、cast、合并、参数梯度归约、分配及 apply-mix 累加缓冲区清零。Pre 前向使用之前已验证的 DeepGEMM TileLang 融合投影，但为反向保存中间量，后续采用 split/RMS/apply 路径；不是之前 inference-only big-fuse 计时。

两种边界必须分开比较：

- `--mode backward`：每侧各自前向准备缓存，不计入反向时间；TL Sinkhorn 的内部重算仍计入。缓存策略不同，不能仅由此判断完整训练成本。
- `--mode train`：每次包含前向（含缓存生成）和反向，不含优化器更新。AscendC 前向打开原生缓存，不能复用无缓存推理时间。

独立 Sinkhorn 缓存使用原生前向实际生成，保持其不透明的分核布局；PreSinkhorn 使用自身原生前后向配对，不能混用两种缓存。无 gamma、无外部 grad_x_post；BF16 特征、FP32 权重，Sinkhorn 20 轮、eps=1e-6。Pre 两侧 HF32。CPU FP32 autograd 为独立参考，保留 BF16 输入和输出 cast。每个梯度都与参考及另一侧核对，另外重复调用防止累加污染。BF16 rtol/atol=0.02/0.02，FP32=1e-3/1e-4；失败保存 correctness.json 并停止计时，不自动放宽阈值。这是初步精度门限，需结合输出误差判断适用范围。

## 1. 服务器构建新增反向算子

在 zwh 环境、Q4 根目录执行。新 vendor 同时包含前向和反向，避免反向错误地链接其他版本的前向。编译仍用已验证的源码提交和 GCC13，不更新第三方仓库。

```bash
conda activate zwh
cd /home/z30086261/Q4
git pull --no-rebase
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$PWD/third_party/torch-extensions-gcc13"
set -o pipefail
cd third_party/ops-transformer
bash build.sh --pkg --soc=ascend950 --vendor_name=q4mhcbwd \
  --ops=mhc_post,mhc_post_backward,mhc_sinkhorn,mhc_sinkhorn_backward,mhc_pre,mhc_pre_backward,mhc_pre_sinkhorn,mhc_pre_sinkhorn_backward \
  2>&1 | tee ../../mhc-backward-build.log
```

先确认构建成功，再安装：

```bash
mkdir -p /home/z30086261/Q4/third_party/q4mhcbwd-opp
bash build_out/cann-ops-transformer-q4mhcbwd_linux-x86_64.run \
  --install-path=/home/z30086261/Q4/third_party/q4mhcbwd-opp \
  2>&1 | tee ../../mhc-backward-install.log
cd /home/z30086261/Q4
find "$PWD/third_party/q4mhcbwd-opp" -name set_env.bash -print
```

以安装日志给出的实际 vendor 路径为准。默认预期如下；先确认目录和 libcust_opapi.so 存在：

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/third_party/q4mhcbwd-opp/vendors/q4mhcbwd_transformer"
test -d "$ASCEND_CUSTOM_OPP_PATH/op_api/lib"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
ls "$ASCEND_CUSTOM_OPP_PATH/op_api/lib"
```

包名或目录与上述不同则按安装日志调整，不猜测其他包代替。已有 Python wrapper 含 Post/PreSinkhorn 反向；独立 Sinkhorn 与 PreV2 用本目录薄 C++ ACLNN 桥，首次运行会编译该桥。新 vendor 的库路径必须在旧 q4mhc 路径之前。

## 2. 小规模精度验证

```bash
bash operators/mhc/backward/run_backward.sh check-basic
bash operators/mhc/backward/run_backward.sh check-pre
```

每个阶段检查三种种子，512x4096，两种计时边界；任一失败先处理，不执行性能扫描。每个实际测试进程会再次校验自身输入后才计时。首轮可以只执行 check-basic 并将日志更新到 error.txt。

## 3. 性能与设备 kernel 分析

精度阶段通过后：

```bash
bash operators/mhc/backward/run_backward.sh bench
bash operators/mhc/backward/run_backward.sh profile
```

bench 覆盖 tokens=512/8192、hidden=4096/7168（Sinkhorn 无 hidden），四个 suite、两种模式、三个独立进程重复。每进程预热100次、9轮每轮100次，两侧轮流先执行。CPU autograd 大尺寸参考需要较多主机内存；初始化和 JIT 均在计时之外。脚本结果路径包含时间与 PID，防止覆盖历史结果。需要单测可运行：

```bash
python operators/mhc/backward/bench_backward.py --suite post --mode backward \
  --tokens 512 --hidden 4096 --check-only --output results/mhc/bwd-post-smoke-01
```

保存 correctness.json、metadata.json、timings.csv 和 summary.json。event_us 是同一 stream 上完整调用链的事件间隔，包含调度间隙；wall_us 为同步主机耗时，都不是单个 kernel 执行时间。summary 的 ascendc_over_tilelang>1 表示 TL 一侧调用链较快。最终汇总采用三个进程中位数，不以最快单次为结论。profile 包含每侧各30次活跃调用；从 kernel_details.csv 按实际调用次数求每次调用的 kernel 总设备时间，再与 API 事件耗时一起报告。Pre 混合路径须将 PyTorch kernel 全部计入，不能只加总名称含 mhc 的 kernel。
