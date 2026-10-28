# MHA 注意力核心：BF16 输入、BF16 输出，双侧原生前向

用户确认的新比较规格。原《MHA和mHC规格表》及 FP32 输出历史结果保留；本轮结果单独放在 results/mha-bf16，不与 FP32 输出的 52.734us 混用。

| 项目 | 双侧相同规格 |
|---|---|
| 范围 | 给定 Q/K/V 的注意力核心；不含 QKV 投影、输出投影 |
| batch/head | B=1、Hq=Hkv=1，沿用现成示例的单 head 范围 |
| Q/K/V | [4096,128]，BF16，同一份 CPU 生成数据复制到 NPU |
| 输出 | [4096,128]，BF16，均由内核直接写出，没有 PyTorch 后置 cast |
| scale | 1/sqrt(128) |
| mask/dropout/bias | 无；非 causal；无 sink、无 KV cache，关闭 LSE 输出 |
| TileLang | 固定源码 core.py::flash_attention_fwd(out_dtype='bfloat16')，FP32 accumulation |
| AscendC | 固定 ops-transformer 的 FlashAttn + FlashAttnMetadata，未修改原生 kernel |

TileLang 源码提交：80364b3d9758854a9692cc0f14089025f54d3f3f；AscendC 源码提交：ba5162411c919e15fbabb28298e8b420c11ad905。不重新安装现有 TileLang/torch/DeepGEMM。

AscendC 通过无复制 view 使用 BSND=[1,4096,1,128]，与 TL 的二维张量共享完全相同存储。Python/PyTorch 只负责分配、view、原生接口封装和测量，没有数值补算。

## 计时边界

- `--ac-metadata cached`：静态形状 metadata 在计时外由原生 metadata 算子生成，之后复用。结果明确注明 setup 不计入，setup 首次主机时间（含 metadata JIT）另存。
- `--ac-metadata per_call`：每次 AscendC 调用都生成原生 metadata，开销包含在 API 和 kernel 总时间中。TL 不需要此步骤。

两种边界分别报告。双方都计入调用内部输出/临时张量分配，不计输入准备、CPU参考和首次编译。精度先于计时，CPU FP32 分块 attention 计算后舍入一次至 BF16；逐侧与 CPU 参考及互相比对。沿用 MHA 初步门限 relative_l2<=1%、max_abs<=0.02，并严格核对 dtype/shape/finite；失败保留 correctness.json 并停止，不自动放宽。

## 1. 服务器编译原生 AscendC attention

在现有 zwh 环境中执行。独立 Git worktree 复用本地提交对象，不改已有 ops-transformer 源码或安装的 mHC 包；独立 vendor q4fa。构建时跳过无关 LFS 测试数据下载，无递归删除或环境重装。默认8线程，可通过 Q4_BUILD_JOBS 调整。

```bash
conda activate zwh
cd /home/z30086261/Q4
git pull --no-rebase
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export ASCEND_NPU_ARCH=dav-3510
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$PWD/third_party/torch-extensions-gcc13"
set -o pipefail
bash operators/mha/build_ascendc.sh build 2>&1 | tee mha-bf16-build.log
```

先确认构建成功，再安装：

```bash
bash operators/mha/build_ascendc.sh install 2>&1 | tee mha-bf16-install.log
```

以安装日志给出的 vendor 实际路径为准。预期如下：

```bash
export ASCEND_CUSTOM_OPP_PATH="$PWD/third_party/q4fa-opp/vendors/q4fa_transformer"
test -d "$ASCEND_CUSTOM_OPP_PATH/op_api/lib"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
ls "$ASCEND_CUSTOM_OPP_PATH/op_api/lib"
python operators/spec_v1/prepare_mha.py
```

该 shell 切到 attention vendor；后续重新运行 mHC 时需要恢复原 q4mhc vendor 路径。

## 2. 先运行一个种子的精度检查

```bash
python operators/spec_v1/bench.py --suite mha_bf16 --ac-metadata cached --check-only \
  --output "results/mha-bf16/first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee mha-bf16-first.log
```

使用现有 cann_ops_transformer Python 包，它已包含 FlashAttn wrapper；第一次会 JIT 编译该 wrapper。若出现符号/版本问题，先保留完整日志，不切换到 torch SDPA 或其他数值 fallback。

## 3. 通过后执行完整检查与计时

依次执行，前一步失败就停止：

```bash
bash operators/mha/run_bf16.sh check
bash operators/mha/run_bf16.sh bench
bash operators/mha/run_bf16.sh profile
```

check：种子42/43/44，两个 metadata 模式。bench：两个模式各3个独立进程，每进程预热100次、9轮每轮100次，轮内交替TL/AC先后顺序。profile：各模式各实现30次活跃调用。每次新建独立结果目录，不覆盖历史结果。

将 benchmark 摘要、profile 的 kernel-details.txt 更新到 error.txt。API 使用事件间隔与同步主机耗时；kernel 时间另从CSV统计，注意 mixed kernel不能将AIC/AIV重复计数，per_call模式需要把metadata设备耗时计入。最终按三个进程的中位数汇总；AC/TL>1表示TL路径较快。

## 本机验证范围

本机无NPU：Python语法、CLI入口、原生调用的参数和 metadata 计时边界测试可验证。实际精度、AscendC 包构建与性能由服务器执行，不能提前声明通过。

## 构建下载 JSON 时的 SSL 证书错误

若 third_party_json 下载 json-3.12.0.tar.gz 因企业代理 CA 失败，更新后的构建脚本会优先复用已成功 mHC 构建的
third_party/ops-transformer/third_party/pkg/json-3.12.0.tar.gz，将其复制到新 worktree 的同一缓存位置。
先检查归档能被 tar 读取，并打印 SHA256；后续仍走上游原有校验/构建流程。
不删除 build、不关闭 TLS 校验、不改动 AscendC kernel。重新运行同一 build 命令即可增量继续。

如果日志提示 No prior JSON cache，先查看已有归档位置：

```bash
find /home/z30086261/Q4/third_party -type f -name json-3.12.0.tar.gz -size +0c -print
```

此时把结果更新至 error.txt，选择可信且完整的同版本缓存后再继续。不要把本次失败下载留下的空归档当作有效缓存。

## Eigen 及其他依赖的统一离线缓存修复

仅复用 JSON 不够；新 worktree 也缺少 Eigen、Abseil、Protobuf 等依赖缓存。
现按上游 cann/cmake 的 master-059 依赖定义，查找旧 mHC 构建 third_party 下的同版本归档和补丁，
用上游声明的 SHA256 验证后，复制到新 worktree 中各模块优先查找的本地位置。
覆盖 JSON 3.12.0、Eigen 5.0.0、Abseil 20230802.1、Protobuf 25.1 及其 Abseil CVE 补丁。
不会复用编译后的共享库，不修改原生算子源码，不关闭 TLS 校验。

构建脚本已自动调用 seed_dependency_cache.py。也可以先单独执行它：

```bash
python operators/mha/seed_dependency_cache.py 2>&1 | tee mha-dependency-cache.log
```

看到 All dependency caches ready 后再运行原 build 命令。
如果某个归档缺失或校验不匹配，脚本在构建之前停止，并列出缺失项与候选位置，避免再次盲目触发网络下载。
依赖来源：https://atomgit.com/cann/cmake/tree/master-059/third_party

## 服务器缺少 Eigen/Abseil 归档时

日志已确认旧目录没有 eigen-5.0.0.tar.gz 与 abseil-cpp-20230802.1.tar.gz。
两份官方原始归档现已随 Q4 放在 operators/mha/dependency_archives（合计约5.1MB），
本机已核对上游 SHA256，归档包含原有许可证。更新后原 seeder 自动使用这两个离线副本，
其余依赖仍复用已有缓存。先重新运行 seed_dependency_cache.py，成功后再构建即可。

## AICPU 收到 x86 ISA 参数时

已定位日志：flash_attn_metadata_aicpu.cpp 编译出现 unknown architecture nocona、
unknown value haswell。该目标使用 CANN 的 ARM 交叉编译器，但全局 flags 携带 Conda 的 x86 参数。
构建脚本现仅在本次子进程内清除 CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/ASFLAGS，
并备份后清空根 CMakeCache.txt 中继承的 C/C++/ASM 和链接全局 flags。
保留编译器路径、目标特定参数与依赖缓存，上游 CMake 继续设置项目所需选项。
不修改算子源码、不禁用 AICPU、不修改 Conda 环境配置或已安装的 mHC 包。

脚本新增 --incremental，保留已完成的依赖和 kernel 产物；更改编译命令涉及的目标仍可能重新编译。
重新 git pull 后执行原 build 命令即可。完整日志应保留，以便确认 ARM 编译命令不再带 nocona/haswell。
