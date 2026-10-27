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
