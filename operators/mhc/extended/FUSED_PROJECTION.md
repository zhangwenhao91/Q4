# 融合投影接入：DeepGEMM-Ascend + TileKernels

## 实现与边界

已找到 DeepGEMM-Ascend 的 `deep_gemm.tf32_hc_prenorm_gemm`，固定提交：

`3a35326271b069b8b5bf1a754435acd6cc91575b`

上游 `deep_gemm/tilelang_ops/hc_prenorm.py` 本身就是 TileLang Ascend 实现。在 kernel 内完成 BF16→FP32 转换、`A @ phi.T` 和 `sum(A²)`，直接接受 BF16 A[M,K]、FP32 phi[24,K]；输出 mul[1,M,24]、sqrsum[1,M]（num_splits=1，内部已完成 split-K 归约）。

新路径不再用 PyTorch 的 cast/matmul/square/sum 完成设备数值计算；PyTorch 仍负责张量、分配、view、调用及 CPU 参考。这是接入上游已有实现，不是重新编写投影 kernel。

- Pre+Sinkhorn：DG 融合投影/平方和 → TL pre_big_fuse。预期可消除此前独立的 cast、square、reduce 和普通 matmul kernel；实际 kernel 数量以 profiler 为准。
- Pre：DG 融合投影/平方和 → TL reduce/RMS → TL split → TL apply。
- `--projection torch` 保留原混合基线；`--projection deepgemm` 标签为 deepgemm_tilekernels。不存在导入失败后静默回退到 PyTorch 的逻辑。
- 所有投影、输出分配、后处理都在每次计时内执行；不缓存/预计算投影值，不用重复输入值省略工作，不给新路径独享预分配。

## 精度模式（必须单独标明）

DG 源码显式 `T.set_hf32_mode('nearest_even')`。因此新 Pre 使用 `aclnnMhcPreV2(opImplMode=1)` 作为 AC 对手；Pre+Sinkhorn 的 AC 本身为 HF32 模式。新路径强制 `--pre-mode hf32`，不能伪装为旧 Pre 的 FP32 对比。

原 `--projection torch --suite pre` 仍默认 FP32，与旧实验一致。**做前后性能对比时，应重测 torch 基线并显式加 `--pre-mode hf32`**；不能将旧 FP32 的全部差距当成融合收益。run_fused.sh bench 已按相同 HF32 条件重测两条路径与 AC。

DG 的 HF32 舍入与 AC 的具体内部舍入不保证位级一致，仍需独立 CPU 参考和对比测试。默认 deterministic=False 与上游测试一致，可用 --dg-deterministic 另测；不同确定性模式的结果不可混合。

## 第一阶段：在服务器安装 DG

不需重建 q4mhc 算子包：其中 mhc_pre 已导出 MhcPreV2。薄 C++ 桥会因新增 pre_hf32 在首次运行时重编。

```bash
conda activate zwh
cd /home/z30086261/Q4
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
# 单独指定新 vendor，避免重复 source 累积旧 vendor；LD_LIBRARY_PATH 仍由 CANN/本脚本管理。
export ASCEND_CUSTOM_OPP_PATH=/home/z30086261/Q4/third_party/q4mhc-opp/vendors/q4mhc_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=4
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$PWD/third_party/torch-extensions-gcc13"
set -o pipefail
bash operators/mhc/extended/install_deepgemm.sh 2>&1 | tee mhc-deepgemm-install.log
```

安装脚本：先验证 GCC13 的 C++20 `<format>` 支持；克隆并锁定上游、获取递归子模块；安装 tree-sitter 构建依赖；使用 `--no-deps --no-build-isolation` 构建完整 DG `_C` 扩展，保留已验证的 torch/torch_npu/tilelang。不使用 DG_SKIP_BUILD，不修改其他用户的 TileLang 源码。若现有 DG 安装不属于本实验，请先自行确认环境归属；此处在 zwh 中安装 deep_gemm 包。

CANN 的 bisheng、ld.lld 需可用。源码安装可能耗时，失败时保留从第一个错误起的日志，不用另一个包替代。版本/子模块记录位于 third_party/deepgemm-revision.txt，安装后依赖记录位于 third_party/deepgemm-install-freeze.txt。

## 第二阶段：先核对投影，再核对完整输出

```bash
bash operators/mhc/extended/run_fused.sh check
```

先在 512×4096 上使用 42/43/44 三个种子，覆盖 Pre、Pre+Sinkhorn。每个 case 在计时前单独核对 raw projection 与 sqrsum，防止归一化把共同尺度误差隐藏：

- 与 CPU FP32 参考比较，记录 max_abs、RMSE、relative_energy_diff。
- relative_energy_diff = sum((actual-ref)²)/sum(actual²+ref²)，是上游 calc_diff 的代数等价式，用 FP64 统计避免消减误差；要求 <1e-6，出现 NaN/Inf 直接失败。
- 继续核对 h_in/h_post/h_res：分别对 CPU 参考和 AC；沿用既有 BF16/FP32 阈值，不自动放宽。
- 原始投影检查在计时之外；投影计算本身每次计时重新执行。

输出目录必须为新目录，保留全部历史实验。check 通过后才执行 bench。大形状在 bench 前也会做相同精度检查。

## 第三阶段：重测与设备采集

```bash
bash operators/mhc/extended/run_fused.sh bench
bash operators/mhc/extended/run_fused.sh profile
```

bench：每个 suite、每个 backend，三次独立进程，200 次预热、每轮 1000 次、9 轮；tokens=512/8192，D=4096/7168，均为 HF32。第一/三次先跑旧基线，第二次先跑 DG。两侧 API 都重新分配输出；不将 check、编译和 profiler 时间放进正常计时。

profile：8192×4096 上分别采集新 TL 路径和 AC，30 次 API。Pre 是多 kernel 路径，汇总每次 API 的全部设备工作量，不能只选 prenorm 或 fused-tail；预期旧 cast/square/mean/sum kernel 不再出现，但以设备 CSV 为证据。profile 导出汇总在该次结果目录的 kernel-details.txt。

每次运行记录 Q4/TileKernels/ops-transformer/DG SHA、工作区状态、编译器和 pip freeze；metadata.json 含 DG 版本/模块路径、预期提交、精度模式。DG 库实际源码身份以安装脚本记录和版本信息交叉确认；expected_revision 字段本身不是运行时二进制来源证明。

## 本机验证情况

本机没有 NPU/PyTorch。已完成上游源码接口/精度模式核对、适配器 CPU mock 合约测试（输入/输出布局、num_splits=1、无静默回退）、Python AST 与 CLI 检查。没有声称 DG 的设备编译、精度或性能已经验证。shell 脚本需服务器 bash 执行；此前本机 MSYS shell 无法启动。

参考：
- https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/deep_gemm/tilelang_ops/hc_prenorm.py
- https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/tests/test_hyperconnection.py
- 锁定 ops-transformer 的 mhc/mhc_pre/op_host/op_api/aclnn_mhc_pre_v2.h（HF32 mode=1）。

## 已知上游问题：x86_64 主机的 adv_api 目录发现

锁定的 DeepJIT 65f5139 在 backend.hpp 中写死 aarch64-linux/asc/include/adv_api；当前服务器为 x86_64，运行 get_num_sms 时因此失败。补丁只调整主机工具链目录发现，依次检查本机架构目录与通用 asc/include/adv_api，不改设备数值 kernel，也不往 CANN 系统目录创建软链接。

在原有 zwh、CANN、q4mhc vendor、ASCEND_RT_VISIBLE_DEVICES=4 环境下执行：

```bash
bash operators/mhc/extended/repair_deepjit_host.sh 2>&1 | tee mhc-deepjit-repair.log
```

修复脚本先验证锁定子模块 SHA 和现有头文件目录，再按精确上下文应用补丁并重编 deep_gemm `_C`。目录不存在时停止，不猜测安装路径；源码上下文不匹配时停止。补丁可重复应用，记录原始头文件、diff 和前后 SHA256 到 third_party/deepjit-host-fix/。源树会有可解释的修改，安装版本可能标为 +local；此时不要再执行要求干净源码树的 install_deepgemm.sh。

重装后验证真实调用 deep_gemm.get_num_sms()，覆盖此前仅验证导入没有触发的延迟 JIT 初始化。出现 `DeepGEMM JIT initialization passed` 后，再执行 run_fused.sh check；数值和性能仍需设备验证。
