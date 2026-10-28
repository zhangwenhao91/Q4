# Q64 候选：分块与任务分配优化实验

状态：Q64已在NPU编译并通过三种种子精度，但性能退化，当前不采用。汇总见 reports/mha_q64/report.md。原始Q128继续作为最佳TL基线。

## 三条原生路径，同进程、相同输入

- tilelang_q128：已验证的原始固定版本，Q=128、KV=128、stages=3，32个逻辑任务。
- tilelang_q64：候选 Q=64、KV=128、stages=3，64个逻辑任务；BF16输入、BF16内核直接输出。
- ascendc：此前已验证的原生 FlashAttn，复用metadata，使用同一 q4fa vendor，不重新构建或安装。

B=1、H=1、S=4096、D=128，scale=1/sqrt(128)，无mask/dropout/bias，与前一轮一致。CPU FP32分块参考最后舍入一次到BF16。
每侧与CPU参考比较，再执行三组两两比较。门限保持 relative_l2<=1%、max_abs<=0.02，严格检查shape/dtype/finite，失败即停止。

## 必须显式记录的存储补齐

源码的 ROWS=block_q/2。Q64时逻辑ROWS=32，但部分 S.vld/S.vsts 使用64个FP32元素的完整SIMD宽度。
直接把Q改为64可能使 m/l/alpha/old_m/row_sum 的读写超出32元素分配。本实验生成独立 core_q64.py，
仅将这些统计数组的存储长度改为 max(ROWS,64)，保留所有逻辑索引与输出范围。
公式、KV分块、stages、同步、精度及BF16输出路径不变。原始core.py与其manifest不修改。

改动受原始源码SHA256保护；候选保存于 third_party/mha-q64-candidate/core_q64.py，哈希和padding信息写入metadata.json。
本机AST检查验证：移除新增存储长度定义并还原统计数组shape后，与原始AST完全相同。
这保证改动范围，不替代NPU编译、实际精度和越界检测；候选是否更快仍须实测。

## 服务器：第一步只做单种子校验

沿用已有zwh、CANN、GCC13、q4fa环境。**不需要重编译AscendC包。**

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
export ASCEND_CUSTOM_OPP_PATH="$PWD/third_party/q4fa-opp/vendors/q4fa_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
set -o pipefail
python operators/spec_v1/bench.py --suite mha_q64 --ac-metadata cached --check-only \
  --output "results/mha-tuning/q64-first-$(date +%Y%m%d-%H%M%S)-$$" \
  2>&1 | tee mha-q64-first.log
```

候选需要一次TileLang JIT；编译失败或精度失败时更新error.txt，暂不计时。没有数值fallback，不自动选择Q128冒充Q64。

## 校验成功后再依次运行

```bash
bash operators/mha/tuning/run_q64.sh check
bash operators/mha/tuning/run_q64.sh bench
bash operators/mha/tuning/run_q64.sh profile
```

check：种子42/43/44。bench：3个独立进程，每进程预热100次、9轮、每轮100次，三侧同输入；
轮次顺序轮换并反转，9轮中每侧在每个位置各出现3次，避免候选始终位于中间。
profile：每个实现30次活跃调用，分别输出CSV，不因kernel都叫main_kernel而混合统计。

summary.json 保存三侧event/wall中位数和两个比值：
- q128_over_q64>1：候选比原始TL快。
- ascendc_over_q64>1：候选比复用metadata的原生AC快。

以同轮实测的Q128和AC为参照，不仅与历史52.56/39.69us比较；最终结论结合3进程API中位数、设备kernel时间和精度。
只对当前单head固定形状得出结论，不能推广为全部attention场景。

## 下一步：收集管线利用率，先定位再改流水线

Level0 已确认退化发生在设备kernel内，但没有足够的利用率指标确定搬运、Cube或Vector瓶颈。
新增 metrics 模式显式请求 torch_npu Level1 + AiCMetrics.PipeUtilization + Text导出，
不修改kernel、不再计时扫描参数；每侧采集30次调用。不支持接口时明确报错，不降级成Level0。

```bash
bash operators/mha/tuning/run_q64.sh metrics
LATEST=$(find results/mha-tuning -maxdepth 1 -type d -name 'q64-*-metrics-*' | sort | tail -n 1)
cat "$LATEST/kernel-details.txt"
cat "$LATEST/csv-files.txt"
```

将这两份输出更新error.txt。优先核对Cube/Vector/MTE等实际导出的字段，避免将缺失字段或N/A当作0利用率。
指标采样可能影响耗时，Level1数据用于瓶颈定位，不直接替换之前Level0性能结论。
官方接口来源：https://github.com/Ascend/pytorch/blob/master/torch_npu/profiler/experimental_config.py
