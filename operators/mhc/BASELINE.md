# mHC Post / shifted Pre 基线采集

只调用现有 spec_v1 实现，不包含新融合候选，不运行 MHA，不安装依赖。
固定 hidden=5120、四路残差；tokens=512/8192。shifted Pre 的 AscendC 对手为
PreV2 + Sinkhorn + PyTorch apply 组合，包含未使用的原生输出与缓存开销。

## 服务器环境

```bash
conda activate zwh
cd /home/z30086261/Q4
git pull --ff-only
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
export ASCEND_NPU_ARCH=dav-3510
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$PWD/third_party/torch-extensions-gcc13"
export ASCEND_CUSTOM_OPP_PATH="$PWD/third_party/q4mhc-opp/vendors/q4mhc_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
bash operators/mhc/run_baseline.sh preflight
```

vendor 必须使用实际已安装的 mHC 包，不能沿用仅含 MHA 的 q4fa vendor。
如果服务器重新克隆导致 third_party 丢失，先运行 preflight 并回传报错；
本脚本不自动重装框架，不会用替代实现跳过依赖问题。
DeepGEMM 及其两项兼容补丁见 extended/FUSED_PROJECTION.md。

若预检报 `No module named 'tile_kernels'`，可单独恢复锁定的 TileKernels：

```bash
bash operators/mhc/restore_tilekernels.sh
bash operators/mhc/run_baseline.sh preflight
```

恢复脚本采用固定提交和 `--no-deps --no-build-isolation`，不升级框架；
已有源码存在修改时停止并保留。其他缺失项以 preflight.log 的完整清单为准。

## 执行

环境检查通过后，可一次完成全部流程，任一步失败即停止：

```bash
bash operators/mhc/run_baseline.sh all
```

也可分步运行，每步使用独立结果目录：

```bash
bash operators/mhc/run_baseline.sh check
bash operators/mhc/run_baseline.sh bench
bash operators/mhc/run_baseline.sh profile
```

check：N=512，两个 suite 各三个种子。bench：两个 N 各三个独立进程，
预热100次，9轮各100次；每个进程先验精度。profile：两个 N、双方实现，
每组30次目标调用。bench/profile 单独运行也先检查对应形状精度。

结果位于 results/mhc-baseline/，结束打印 tar.gz 路径，回传该包即可。
失败时回传对应目录的日志。API event/wall 与设备 kernel Duration 分开解释；
profile 应汇总一次 API 的全部 kernel，不能仅统计投影或 Sinkhorn。
preflight 仅检查依赖、设备和 JIT 初始化，不代表设备 kernel 编译或精度通过。

本脚本在无 NPU 的本机仅作静态检查，设备正确性和性能待服务器验证。
