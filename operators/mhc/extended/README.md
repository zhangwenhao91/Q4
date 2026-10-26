# mHC 扩展比较：Sinkhorn / Pre / Pre+Sinkhorn

本目录新增三组前向测试，沿用锁定的 TileKernels 与 ops-transformer 提交，不修改上游 kernel。

| suite | 比较边界 | TileKernels 路径 | AscendC 路径 |
|---|---|---|---|
| sinkhorn | logits → softmax+eps → 定次行列归一化 | mhc_sinkhorn_fwd | aclnnMhcSinkhorn，无反向缓存 |
| pre | BF16 x、FP32 phi/alpha/bias → h_in、h_post、残差 logits | PyTorch FP32 投影/RMS + TL split/apply | aclnnMhcPre，FP32 Cube，无 gamma/反向缓存 |
| pre_sinkhorn | 同上，残差 logits 再做 Sinkhorn | PyTorch 投影/平方和 + TL pre_big_fuse | MhcPreSinkhorn，HF32 Cube，out_flag=False |

重要：Pre 两组是 **PyTorch 投影 + TileKernels 的完整实现链** 对比 AscendC 融合算子，结果标签为 torch_projection_plus_tilekernels。计时包含 cast、GEMM、平方和、分配以及 TL 后处理，绝不将预计算 GEMM 排除后与完整 AC 相比。此结果不能代表纯 TileLang vs AscendC 编译器的比较。若要测生产级纯 TileLang 链，下一阶段需接入并单独验证 DeepGEMM-Ascend。

## 参数与精度

- N=4；eps=1e-6；Sinkhorn 默认迭代 20；post multiplier=2；不启用 gamma，不保存反向缓存。
- pre 默认 FP32，与 aclnnMhcPre 入口匹配；pre_sinkhorn 将 torch.npu.matmul.allow_hf32=True，与 AC 文档规定的 HF32 对齐。两边具体舍入仍可能有差异，必须先验精度。
- Pre+Sinkhorn 的 950 D 取交集 4096/7168；tokens 使用 16 的倍数。
- 输入 CPU 生成后复用；phi 限制在 ±1/sqrt(4D)，CPU FP32 参考。分别核对两侧与参考，以及两侧互相比对。
- BF16 初步阈值 rtol=0.02/atol=0.02；FP32 rtol=1e-3/atol=1e-4。失败即停止，保留误差记录，不自动放宽。
- 同一输入的 eps 与迭代定义见上游实际 aclnn 文档和 kernel；独立 Sinkhorn README 叙述存在出入，不据它判断语义。
- correctness.json 记录最大绝对误差/RMSE；timings.csv 保留全部轮次；metadata.json 保存环境和计时参数。
- 不同输入种子可以 --seed 指定。smoke 不是完整精度认证。

## 服务器准备

在 Q4 根目录执行。首次扩展构建选择新 vendor q4mhc，避免覆盖已经验证的 Post 包。沿用服务器已有 zwh、GCC13 与 CANN：

```bash
conda activate zwh
cd /home/z30086261/Q4
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$PWD/third_party/torch-extensions-gcc13"
set -o pipefail

cd third_party/ops-transformer
# 若企业代理证书问题尚未修复，先按之前的证书方案处理下载。
bash build.sh --pkg --soc=ascend950 --vendor_name=q4mhc \
  --ops=mhc_post,mhc_pre,mhc_sinkhorn,mhc_pre_sinkhorn \
  2>&1 | tee ../../mhc-extended-build.log
```

构建成功后，安装新包：

```bash
mkdir -p /home/z30086261/Q4/third_party/q4mhc-opp
bash build_out/cann-ops-transformer-q4mhc_linux-x86_64.run \
  --install-path=/home/z30086261/Q4/third_party/q4mhc-opp \
  2>&1 | tee ../../mhc-extended-install.log
```

**实际 vendor 目录可能带 _transformer 后缀，以安装日志给出的 source 命令为准。** 可用下面命令查找：

```bash
find /home/z30086261/Q4/third_party/q4mhc-opp -name set_env.bash -print
```

source 上一条输出的实际脚本路径，然后进入 Q4 根目录。旧整包 Python wrapper 已包含 PreSinkhorn；Pre 与独立 Sinkhorn 缺少同样的 Python wrapper，因此此目录提供薄 C++ ACLNN 桥，首次运行会用 GCC13 编译，kernel 保持上游实现。依赖已安装的 cann_ops_transformer 包的 common header 和 CANN-first 编译配置；缺头文件或 API 符号时停止，保留完整日志。

## 先精度，再计时

先分别执行，某一步失败即停止。每次 output 必须是新目录，防止覆盖历史结果。

```bash
cd /home/z30086261/Q4
python operators/mhc/extended/bench_forward.py --suite sinkhorn \
  --tokens 512 --check-only --output results/mhc/ext-check-sinkhorn
python operators/mhc/extended/bench_forward.py --suite pre \
  --tokens 512 --hidden 4096 --check-only --output results/mhc/ext-check-pre
python operators/mhc/extended/bench_forward.py --suite pre_sinkhorn \
  --tokens 512 --hidden 4096 --check-only --output results/mhc/ext-check-pre-sinkhorn
```

全部通过后，逐组执行计时：

```bash
python operators/mhc/extended/bench_forward.py --suite sinkhorn \
  --tokens 512 8192 --iters 1000 --rounds 9 --output results/mhc/ext-sinkhorn
python operators/mhc/extended/bench_forward.py --suite pre \
  --tokens 512 8192 --hidden 4096 7168 --iters 100 --rounds 9 --output results/mhc/ext-pre
python operators/mhc/extended/bench_forward.py --suite pre_sinkhorn \
  --tokens 512 8192 --hidden 4096 7168 --iters 100 --rounds 9 --output results/mhc/ext-pre-sinkhorn
```

对于 Pre，CPU 参考及设备 FP32 cast 占用较大内存；先跑 smoke 再扩展。不同 suite/shape 均独立验精度。API 为热缓存重复调用，输出新分配；不等于纯 kernel 时间。

## profiler

同一脚本支持 --profile tilelang 或 --profile ascendc，先验精度、预热，再采集。示例：

```bash
python operators/mhc/extended/bench_forward.py --suite sinkhorn \
  --tokens 512 --profile tilelang --output results/mhc/ext-prof-sinkhorn-tl
python operators/mhc/extended/bench_forward.py --suite sinkhorn \
  --tokens 512 --profile ascendc --output results/mhc/ext-prof-sinkhorn-ac
```

Pre/PreSinkhorn 同理改变 suite。每个活跃区间 30 次 API 调用，但混合 Pre 路径每次会启动多个 kernel。统计全部 kernel，不能只选 pre_big_fuse；保留原始 CSV，不把 Wait Time 加入 kernel Duration。

## 覆盖范围

本阶段覆盖三个前向边界；不包含 backward。后续应单独对齐缓存与梯度输出，再补 Post backward、Pre backward、Sinkhorn backward。当前提供的是可在服务器验证的脚本，本机无 NPU，仅做静态核查与 CLI/语法验证。

## 源码依据（锁定的 ops-transformer ba5162411）

- mhc/mhc_sinkhorn/docs/aclnnMhcSinkhorn.md 与 op_kernel/arch35/mhc_sinkhorn_arch35.h：softmax 初始化、eps 和定次迭代；可不输出 normOut/sumOut。
- mhc/mhc_pre/op_host/op_api/aclnn_mhc_pre.cpp：aclnnMhcPre 选择 FP32 模式；gamma/缓存为可选。
- mhc/mhc_pre_sinkhorn/torch_extension/mhc_pre_sinkhorn.py：无 requires_grad 时 out_flag=False，返回 3 个主输出；h_res 为平铺 16，脚本统一 reshape。
- TileKernels tests/mhc/test_pre_big_fuse.py：GEMM 不在原有 fused-tail benchmark 中。

## 融合投影路径（后续阶段）

已接入 DeepGEMM-Ascend 的原生 TileLang HC 融合投影/平方和 kernel。安装、精度模式、独立投影校验及重测步骤见 [FUSED_PROJECTION.md](FUSED_PROJECTION.md)。旧 PyTorch 路径保留，新路径使用 `--projection deepgemm`，Pre 对手改为 HF32 MhcPreV2，不能直接与旧 FP32 实验混为同一条件。
