# mHC TileLang / AscendC 核查与性能验证

核查日期：2026-10-26。本机无 NPU，已完成源码与接口核查、测试脚本准备及 Python 语法检查；未执行设备编译、正确性测试或性能测量。

## 1. 版本锁定与存在性

- TileKernels main：c1cf914a82e6901c679445f872051ce44ac2cc63。
- ops-transformer master：ba5162411c919e15fbabb28298e8b420c11ad905。
- 已下载核对相关源码。Windows 全量检出存在 LFS 与超长文件名限制；服务器使用下列 Linux 指令重新检出锁定版本。

两种实现均存在：

| 范围 | TileKernels | ops-transformer 常规目录 |
|---|---|---|
| Post 前向/反向 | tile_kernels/mhc/post_kernel.py、post_asc.py | mhc/mhc_post、mhc/mhc_post_backward |
| Pre | norm_fn、pre_split_mixes、pre_apply_mix、pre_big_fuse | mhc/mhc_pre、mhc/mhc_pre_backward |
| 含 Sinkhorn 的 Pre | 分解链或 pre_big_fuse（输入为 GEMM partials） | mhc/mhc_pre_sinkhorn、对应 backward |
| Sinkhorn | sinkhorn_kernel.py、sinkhorn_asc.py | mhc/mhc_sinkhorn、mhc/mhc_sinkhorn_backward |

ops-transformer 还保留 experimental/mhc/mhc_pre、mhc_post、mhc_res，包含静态权重版本。不要仅凭同名将实验目录里的 Post 与 TileKernels 的融合 Post 比较。

TileKernels README 要求 Python>=3.12、PyTorch>=2.13、TileLang>=0.1.15；Ascend 后端要求 Ascend 950、CANN>=9.2.0。还需配套 torch_npu。版本安装来源和兼容性以服务器实际发行包为准，不能让 pip 静默替换成无 NPU 支持的框架。
AscendC MhcPost 支持 A2/A3/950；这不代表 TileKernels 支持 A2/A3。只有 910B/A2/A3 时，应先解决 TileLang 移植适配，不能将运行失败当成性能劣势。

## 2. 首选比较：Post 前向

计算：Y[b,s,j,d] = sum_i C[b,s,i,j]*R[b,s,i,d] + H[b,s,j]*X[b,s,d]。

| 语义 | TileKernels mhc_post_fwd 参数 | AscendC mhc_post 参数 |
|---|---|---|
| R [B,S,4,D] BF16 | residual | x |
| X [B,S,D] BF16 | x | h_out |
| C [B,S,4,4] FP32 | comb_res_mix | h_res |
| H FP32 | post_layer_mix [B,S,4,1] | h_post [B,S,4] |

即 TL(X,R,H.unsqueeze(-1),C) 对比 AC(R,C,X,H)。矩阵方向一致，不需要转置。两侧都使用返回新输出的 API，不给 TL 独享预分配输出。

附带 bench_mhc_post.py：同一份输入、CPU FP32 参考、BF16 输出核对、预热、交替测试顺序、NPU Event 与同步 wall time、逐轮 CSV。默认 B=1，S=512/8192，D=4096/5120/7168，N=4，20 次预热，7 轮各 100 次。
精度阈值 rtol=0.02/atol=0.02 仅作为初始 BF16 冒烟门槛，不能代替业务验收；输出最大绝对误差也写入 CSV。正式验收可补相对误差、归一化 RMSE、零值/极值/非对称混合矩阵测试。失败后停止计时，先查精度。

## 3. NPU 服务器执行

以下 bash 命令在 Linux 服务器运行。从 Q4 仓库根目录开始，以下命令锁定源码并在 third_party 中准备环境。优先使用已配套 Python/PyTorch/torch_npu/CANN/TileLang 的环境或官方镜像。

```bash
bash operators/mhc/prepare_sources.sh
cp operators/mhc/bench_mhc_post.py third_party/
cd third_party
npu-smi info | tee npu-smi.txt
# 按实际安装路径 source CANN 环境，例如：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c 'import sys,torch,torch_npu,tilelang; print(sys.version); print(torch.__version__,torch_npu.__version__,tilelang.__version__); print(torch.npu.is_available()); print(torch.npu.get_device_name(0))' | tee framework.txt
python -m pip freeze > pip-freeze.txt

# 已确认框架版本满足要求后，避免依赖解析替换现有框架
python -m pip install --no-deps -e './TileKernels[dev]'
# dev 所需 pytest / pytest-xdist / pytest-repeat 等如缺少，再单独安装
python -m pip install pytest pytest-xdist pytest-repeat build

cd ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=mhc_post
ls build_out/*.run
# 执行上一条列出的 mhc_post 自定义算子 .run 包；先检查其 --help，按服务器安装目录部署
# 仓库标准安装命令形式：bash build_out/<实际生成的包名>.run
# 若指定自定义安装目录，按 compile.md source 对应 vendors/custom/bin/set_env.bash
# 默认安装时，也需确认该 vendor 已进入 load_priority，防止调用到系统旧版本

# Python wrapper 单独打包，整包便于使用标准导入路径
bash build.sh --torch_extension
python -m pip install --no-deps build_out/cann_ops_transformer-*.whl
cd ..

# 首次 JIT 编译发生在预热前，小规模先验证
python bench_mhc_post.py --tokens 512 --hidden 4096 --rounds 3 --iters 20 --output smoke.csv | tee smoke.log
# 全部六个形状
python bench_mhc_post.py --output mhc_post.csv | tee mhc_post.log
```

.run 包安装选项与目标目录依赖服务器权限，故没有硬编码未知包名或安装路径。构建或运行失败请保留完整日志；不能跳过安装步骤，用系统包结果冒充锁定源码的结果。用 profiler 的 kernel 名称、已加载库路径和 vendor 配置验证来源。

可独立跑 TileKernels 原生测试，验证仓库自身是否通过：

```bash
cd TileKernels
python -m pytest tests/mhc/test_post.py -m 'not benchmark' -q
TK_BENCHMARK_FAIL_MODE=none python -m pytest tests/mhc/test_post.py --run-benchmark -m benchmark --benchmark-output ../tile_post.jsonl -q
cd ..
```

原生测试还有 backward，且计时工具与附带脚本不同。其结果用于自检，正式横向对比以同一脚本/同一计时方法为准。性能时不要使用多 worker 或多进程并发争用一张卡。

## 4. 如何解释性能

- 核心指标：各 shape 的每次调用延迟中位数（微秒），同时保留 7 轮原始结果和范围。多次独立运行验证稳定性。
- 加速比 = AscendC 延迟 / TileLang 延迟；>1 表示 TileLang 更快。分别列出每个 shape，避免仅给平均数。
- 附带脚本的 Event 是设备时间区间，可能包含主机提交造成的设备空闲；wall time 包括 Python、分配、dispatch 与同步。这两者都不是纯 kernel 时间。
- 要下“算子内核性能”结论，额外用 torch_npu profiler 或 msprof 抓设备 kernel 时间；使用当前安装版本支持的选项（先 msprof --help）。分离 profiler 与最终计时，不在 profiler 开启时统计普通延迟。
- 一个 API 可能启动多个 kernel，统计其全部设备工作；不能只挑最快的一个。查看 timeline 判断 CPU 发射受限、搬运和同步。
- 950 支持且双方均可运行时，可另测等价图捕获/重放以降低 Python 提交影响，但两边必须使用同一模式，结果单独标注。
- 保持同卡、同版本、相同精度、相同输入分布/形状/布局；记录卡型号、驱动、固件、CANN、框架、编译器、源码 SHA、频率/功耗设置。测试期间卡上无其他负载，避免热降频。
- 本脚本是热缓存重复调用。同样工作集、相同输入地址的结果要标明 hot-cache；如需冷缓存/真实模型工作集，另做轮换输入实验，勿混合统计。
- 不把编译/JIT 首次耗时计入稳态延迟；如果关心部署冷启动，单独列。

## 5. 扩展到 Pre、Sinkhorn 和反向

Pre 是第二阶段，必须先画清边界。TileKernels test_pre_big_fuse.py 中明确写明 GEMM 已移至 DeepGEMM，测试用 torch 构造 partials 后只计融合尾部。因此该数字不能直接对比包含 RMSNorm/矩阵乘/映射/Sinkhorn 的 AscendC MhcPreSinkhorn。

可选择：
1. 全链路对比：TL 把真实可在目标 NPU 运行的投影 GEMM、平方和、归约、映射、Sinkhorn、apply 都计入；AC 使用完整 MhcPreSinkhorn。不能拿 CPU/预计算 GEMM 把 TL 主计算排除。
2. 尾部对比：只有 AC 暴露相同 partial 输入的同等实现时才比较；否则明确需要额外实现，不能拿完整 AC 作分母。

参数对齐：N=4、D 优先 4096/7168；MhcPreSinkhorn 文档当前只支持 numIters=20，TL 默认测试为 10，需要统一；post multiplier 统一为 2（TL 测试默认 1），hc_eps/norm_eps/Sinkhorn eps 逐项核对；gamma 不启用；对齐 phi 方向、split 顺序、HF32/FP32 模式，以及 needBackward 是否输出缓存。没有相同精度/缓存功能时单独标记，不能直接归因于语言。

Sinkhorn 不能只看名字：TL 参考从 logits 做 softmax+eps 后定次行列归一化；AC 独立 MhcSinkhorn README 描述非负输入/收敛阈值，与融合 MhcPreSinkhorn 的定次定义不同。应核对 golden、host 和 kernel 的实际初始化、eps 插入位置、迭代计数及输出缓存；不满足等价条件先不做直接性能比较。

反向另外比较 Post backward、Pre backward、Sinkhorn backward；核对梯度输出顺序与累加规则，并统一前向缓存及其是否计入成本。需要评估训练时，再汇总 forward+backward，不能用前向数字代表训练速度。

## 6. 参考源码

- https://github.com/deepseek-ai/TileKernels/tree/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc
- https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tests/mhc/test_post.py
- https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tests/mhc/test_pre_big_fuse.py
- https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post
- https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_pre_sinkhorn
- ops-transformer 锁定版本内 docs/zh/install/compile.md、torch_extension/README.md、mhc/*/README.md 与 torchapi 文档已在本机读取核对。


## 7. 仓库内辅助脚本

从 Q4 根目录执行 `bash operators/mhc/prepare_sources.sh` 会将锁定源码放到 `third_party/TileKernels` 和 `third_party/ops-transformer`，第 3 节使用的就是这两个目录。

完成安装并 source CANN/vendor 环境后，从 Q4 根目录执行 `bash operators/mhc/run.sh smoke` 或 `bash operators/mhc/run.sh full`。脚本记录 npu-smi、pip freeze、源码提交和测试日志，结果位于 `results/mhc/`。CANN、驱动、固件详细版本和 profiler 结果仍需另行保留。
