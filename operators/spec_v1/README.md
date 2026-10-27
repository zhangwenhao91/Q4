# 按《MHA和mHC规格表》执行：前向 v1

本目录独立于此前 mHC benchmark。原规格表不修改，mHC 反向继续挂起。当前仅完成本机静态检查，**尚无 NPU 精度/性能结果**。

## 固定比较范围

- MHA：严格使用单 head 的二维 Q/K/V [4096,128]、BF16 输入、FP32 输出、scale=1/sqrt(128)、无 causal/dropout。随仓库附带并固定 TileLang 示例提交 `80364b3d9758854a9692cc0f14089025f54d3f3f`。它是注意力核心，未加入 QKV/输出投影。
- 当前 stock AscendC FlashAttn 的 BF16 输入对应 BF16 输出，FlashAttentionScoreV2 文档同样要求输出 dtype 与 query 一致。因此 **MHA 首轮只有 TileLang 严格规格单侧测试**。不会将 BF16 转 FP32 冒充匹配实现，不会生成双侧加速比。后续需要找到或适配 BF16 输入/FP32 输出的 AscendC 实现。
- mHC Post：hidden=5120、4条流，BF16 特征、FP32 post/comb 系数、BF16 输出。
- mHC shifted Pre：旧 `prev_pre_mix` 用来生成当前 layer_input；新 pre 系数单独输出为 next_pre_mix。fn=[24,20480]、scale=[3]、base=[24]。norm_eps=1e-20、pre_eps=1e-6、Sinkhorn eps=1e-6，20轮，post乘数2。输出4项均校验。
- Pre 计时包括旧 pre 的 apply、融合投影与平方和、RMS、split、Sinkhorn。按表中四个输出为边界，**不包括末尾子层 RMSNorm、FP8 cast 或前一个 Post**。
- 表中 N 未指定测试值，本轮选 N=512/8192，且当前融合投影适配要求 N 为128的倍数。测试数据采用同一随机种子生成的合成数据，**不是实际 checkpoint 权重**。fn 按已折入 norm 权重的有效布局生成，不再次乘 gamma。

## mHC AscendC 组合路径的限制

当前锁定的 MhcPreSinkhorn 不支持 hidden=5120，且原生 PreV2 不接受 shifted 的 prev_pre_mix。本目录组合：

1. PyTorch NPU FP32 cast + multiply + sum + BF16 cast，应用旧 pre；
2. 原生 aclnnMhcPreV2 HF32 模式，分别传入 norm_eps 与 pre_eps，输出新 pre/post/comb logits；
3. 原生 aclnnMhcSinkhorn。

标签为 `ascendc_pre_plus_torch_apply`，**不是纯 AscendC 融合 shifted 算子**。PreV2 还会计算未使用的当前新 pre 对应 h_in，以及为输出 hPre 必须同时输出的 invRms/hMix 缓存。这些额外计算与分配全部计时，报告中记录，不将组合路径性能当成语言优劣结论。TL 一侧使用此前已验证的 DeepGEMM TileLang 融合投影，两项兼容补丁继续保留。

## 服务器第一步：环境及 MHA 精度

不重装已验证的 torch、tilelang、DeepGEMM 或 q4mhc 包。MHA 示例源码和许可证已随 Q4 附带。准备脚本仅校验 SHA256 并从本地复制文件，完全不联网，不改变已安装编译器。若现有 TileLang 缺少所需 API，先保留报错，再决定兼容处理；不要直接升级整个环境。

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
export ASCEND_CUSTOM_OPP_PATH="$PWD/third_party/q4mhc-opp/vendors/q4mhc_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
set -o pipefail
python operators/spec_v1/prepare_mha.py 2>&1 | tee spec-v1-prepare-mha.log
```

离线准备成功后先运行一个种子的最小检查：

```bash
python operators/spec_v1/bench.py --suite mha --check-only \
  --output "results/spec-v1/mha-first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee spec-v1-mha-first.log
```

将这一步的输出更新到 error.txt。此命令只验证，不测性能；无需先构建新 AscendC attention 包。

## 精度通过后再逐步运行

```bash
# MHA 三种输入种子：仍为单侧测试
bash operators/spec_v1/run.sh check-mha
# mHC Post/shifted Pre：512x5120，三种种子，逐项与CPU参考及另一侧比较
bash operators/spec_v1/run.sh check-mhc
# 上述通过后：三次独立进程重复，9轮，每轮100次；全部先校验再计时
bash operators/spec_v1/run.sh bench
# 各条调用链30次活跃profile，含PyTorch/额外开销
bash operators/spec_v1/run.sh profile
```

Post 沿用已装 q4mhc 库；PreV2+Sinkhorn 薄桥首次运行用 GCC13 JIT 编译。若当前 shell 使用新 q4mhcbwd vendor 且其构建已成功，可以使用该完整 vendor 的实际路径；不要混入不同版本库。

每次结果使用独立目录。metadata.json 记录规格、依赖版本、输入种子、源码提交/本地修改、HC源码哈希和实现边界；correctness.json 记录 shape/dtype/finite/max_abs/RMSE/relative_l2。MHA 使用分块 CPU FP32 softmax attention 参考，初步门限 relative_l2<=1%、max_abs<=0.02；mHC BF16 rtol/atol=0.02/0.02，FP32=1e-3/1e-4。门限为初步 smoke 标准，失败停止，不自动放宽。输入分配、CPU参考和JIT预热不计时；每次调用内部输出/临时张量分配计时。

timings.csv 中 event_us 为完整调用链的 stream event 间隔（可能包含主机发射间隙），wall_us 为含同步的主机时间；都不能替代 kernel 时间。summary.json 给出各进程轮次中位数，最终再汇总3进程。MHA comparison_status.json 标明单侧限制。profile 的 kernel_details.csv 应统计全部相关 kernel，不能丢掉 cast/zero/reduce 或只按 mhc 名字筛选。

## 本机检查范围

本机没有 torch/NPU/CANN：已检查 Python 语法、CLI --help、固定上游示例内容/哈希，以及 ACLNN 调用参数对照。不能在本机编译验证 ACLNN 桥，不能提前声称 NPU 精度通过。Git Bash 在本应用内被 Windows 权限限制，shell 脚本须在服务器验证。

## 首步 SSL 证书报错修复

旧准备脚本通过 urllib 从 raw.githubusercontent.com 下载文件，服务器代理证书链触发 CERTIFICATE_VERIFY_FAILED。现改为使用本目录 upstream_mha 中固定提交的原样源码（含许可证和来源清单），无需关闭 TLS 校验或重装环境。先 git pull --no-rebase，再重新执行同一个 prepare_mha.py 命令；成功后继续上面的单种子精度检查。已存在但被修改的源码会被保护并停止，不自动覆盖。
