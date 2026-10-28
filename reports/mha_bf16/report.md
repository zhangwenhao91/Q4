# MHA BF16 双侧原生前向比较结果

日期：2026-10-28。服务器：本任务既有 Ascend950PR_9579，物理设备4。
范围：B=1、Hq=Hkv=1、S=4096、D=128，BF16输入/输出，scale=1/sqrt(128)，无mask、dropout、bias、sink；不含QKV和输出投影。
双方内核直接写BF16，没有PyTorch数值补算或后置类型转换。该范围为单head注意力核心，不是完整MHA层，也不代表其他序列长度/head数量。

## 精度

三种种子42/43/44、两种metadata模式检查通过；加上六个计时进程的检查，共36项精度记录，无失败。
参考为分块CPU FP32 attention，最终舍入一次至BF16。逐侧对参考及双侧互比均检查shape、dtype、finite、relative L2<=1%、max_abs<=0.02。门限为本轮smoke验收标准，不代表模型级精度认证。

## API耗时

各模式3个独立进程；每进程预热100次，9轮、每轮100次。下表为三个进程轮次中位数的中位数，单位μs。

| 计时范围 | TL event | AC event | TL wall | AC wall | AC/TL |
|---|---:|---:|---:|---:|---:|
| 复用静态metadata | 52.677 | 39.960 | 53.040 | 40.434 | 0.759 |
| 每次生成metadata | 52.722 | 81.933 | 53.212 | 82.551 | 1.554 |

复用模式的metadata由原生算子在计时外准备；每次生成模式将其完整调用计入。两侧均计入调用内部输出/临时分配，排除输入准备、CPU参考和首次JIT。
API event包含stream上调用链及可能的发射间隙，不是纯kernel时间。两种模式分进程测量，不能把中位数相减当作严格独立的metadata微基准。

## 设备kernel耗时

每组单次profile，3个活跃step，每step10次调用。每类kernel均核对到30次，mixed attention每次只计一条记录。数值为平均Duration(us)。

| 模式 | TL attention | AC attention | AC metadata | AC设备时间合计 |
|---|---:|---:|---:|---:|
| cached | 52.561 | 39.689 | 计时外 | 39.689 |
| per_call | 52.644 | 39.836 | 43.106 | 82.942 |

固定形状且复用metadata时，AscendC attention kernel约快1.324倍，API约快1.318倍。
每次准备metadata时，AscendC多出约43.106μs设备工作，完整调用的API耗时由TL路径获得约1.554倍加速。
两种结论分别适用于不同的metadata复用条件，不能合并成不带条件的语言优劣结论。

Profiler曾提示ACL→NPU flow关联解析失败；设备kernel_details.csv已完整生成且调用次数符合预期，本报告使用设备Duration，不据此分析CPU调用关联或AIC利用率。

## 版本和证据

- TileLang示例源码：80364b3d9758854a9692cc0f14089025f54d3f3f，core.py::flash_attention_fwd(out_dtype=bfloat16)。
- AscendC源码：ba5162411c919e15fbabb28298e8b420c11ad905，原生FlashAttn和FlashAttnMetadata；独立q4fa vendor。
- 修复限于构建环境和依赖准备：离线依赖按上游SHA256校验；清理泄漏到ARM AICPU的Conda x86全局flags。未修改attention算子源码。
- [API原始日志](https://github.com/zhangwenhao91/Q4/blob/be6a517/error.txt)。metadata模式按已执行runner的固定交错顺序映射；完整metadata.json仍保存在服务器各结果目录。
- [Profile原始日志](https://github.com/zhangwenhao91/Q4/blob/17ce416/error.txt)。各组实际CSV路径保存在summary.json。
- 本目录summary.json保留每进程摘要、汇总值、kernel计数/分布；kernel_summary.csv用于后续汇总。
- 历史BF16输入/FP32输出的TL单侧测试是不同规格，不与本报告交叉计算加速比。
