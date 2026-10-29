# 从平均耗时转向逐核负载与kernel内部流水线分析

当前最佳观测组合约50.922us，原始TL约52.633us，AscendC约39.846us。
S双版本的额外设备收益仅约0.105%，后续仍保留已验证KV2对照，不把微小差异视作固定收益。

## 要回答的问题

1. 当前32个Q tile在设备可用AIC上的实际分布、逐核耗时和收尾情况。
2. 当前最佳kernel中QK、softmax、P搬入L1、PV和输出累积之间，哪些阶段存在等待。
3. 优先改变任务划分，还是改变L0/L1搬运与同步。

先纠正证据边界：普通torch_npu的kernel_details.csv及主机/设备调用trace只给出task级时间，
不能用32个Block Num直接计算逐核尾部时间，也不能从聚合管线占比恢复kernel内部指令时间线。
32任务/28核有潜在负载问题，不代表已经测得两波等长执行或确定尾部比例。

## 第一步：核对安装工具能力

当前CANN为9.3 weekly，不直接套用旧版工具的芯片支持表或参数。
探测脚本读取msprof版本、msprof和msprof op帮助、设备属性及Python包版本；
没有编译、运行kernel或进行profile，不改CANN/驱动或安装新工具。
帮助中列出的指标也不代表该设备一定支持，正式采集仍需检查工具反馈。

服务器现有zwh/CANN/q4fa环境：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_timeline.sh probe
```

将完整输出更新到error.txt。结果也保存在results/mha-tuning/timeline-probe-*/probe.txt。
当前probe模式仅探测，不会把普通Level1指标当成逐核/指令级采集的降级替代。

## 根据能力选择后续采集

- 若工具及芯片支持逐核负载数据，比较每个物理AIC的工作量和耗时。
  若同时有逐任务/逐核起止时间，再量化收尾；只有计数器则不能重建任务分配时间线。
- 若支持片上流水线或指令时间线，采集当前最佳TL、KV2对照和原生AscendC，
  对照生成C++辨认QK/PV，不把CPU API空隙当作Cube等待。
- 采集可能涉及重放、采样或插桩，产生的时间只用于诊断，不替代现有非插桩延迟基准。
- 若950PR或该工具版本不支持所需能力，明确缺失项，再选择代码诊断、可用硬件计数器
  或单独的模拟分析；不能把模拟时间冒充设备实测。

单算子工具可能默认只采首个kernel，正式采集须只运行目标实现或准确过滤kernel名称，
避免采到内存初始化、CPU参考辅助算子或另一个同名main_kernel。正式命令按下方探测结果固定。

参考官方工具说明：

- [MindStudio Ops Profiler用户指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)
- [CANN算子调优工具概述](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0082.html)

以上网页描述多种版本与芯片，不作为当前设备支持所有功能的依据。

## 已探测环境与正式负载采集

探测日志固定于Q4提交c4dd807。设备确认Ascend950PR_9579、28 AIC/56 AIV；
安装的msprof op帮助提供Default、Occupancy、PipeTimeline、InstrTimeline等选项。
msprof本身不接受--version，但msprof op提供--version；之前此项返回255不代表工具损坏。
帮助中有选项不等于硬件采集已经成功，本步骤仍可能收到芯片支持限制。

先执行逐核负载采集：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_timeline.sh load
```

无需重建AscendC。依次运行tilelang_s2buf、tilelang_kv2、ascendc。
每侧先独立检查CPU参考精度；采集应用也在MSTX之外完成检查和100次预热。
Q4_TIMELINE区间仅有一次所选attention调用，原生metadata预先缓存，
排除辅助算子，并避免FlashAttn名称前缀误选FlashAttnMetadata。
未执行其余TL baseline，尽管公共make_case可能编译未使用的原始版本。

使用实测帮助参数：Default,Occupancy、launch-count=1、warm-up=50、replay-mode=kernel、
mstx=true、mstx-include=Q4_TIMELINE，以及main_kernel/FlashAttn目标名称。
工具可能重放kernel，不能把本轮输出时间直接替代常规API或设备基准。

输出目录为results/mha-tuning/timeline-load-*：

- 每侧validate.log、collect.log、command.txt、app/metadata.json及精度检查。
- TL生成源码selected_core.py、selected_kernel.cpp及其哈希。
- 原始op/OPPROF_*数据全部保留。
- inventory.txt自动打印文件清单、CSV字段和有限数量原始行；inventory.json保留更多原始样本。

将完整终端输出更新到error.txt。若采集失败，脚本在第一处失败停止并打印现有文件，
不切换到不具备逐核信息的普通profile。若没有逐核数据，明确反馈缺失，不能自行推算尾部。

负载采集成功并核对输出后，再执行流水线采集：

```bash
bash operators/mha/tuning/run_timeline.sh pipeline
```

pipeline使用Default,PipeTimeline，其余范围及对照保持一致。
首次先运行load并反馈，不要求立刻运行pipeline。

## 首次采集失败后的修正

首次日志7f57ca3显示PyTorch MSTX前端返回无效ID，三侧没有有效profiler数据。
工具在子进程失败时仍返回0；旧脚本只检查退出码，错误报告完成，现已修正。
具体记录见reports/mha_timeline/attempt_01.md。

当前capture应用改用CANN原生MSTX C接口，而非依赖torch_npu的MSTX前端状态。
采用空stream host范围，目标调用后同步设备，再关闭范围。
RTLD_DEFAULT优先，保留工具预加载符号；必要时从ASCEND_HOME_PATH/lib64等已配置路径加载
libms_tools_ext.so。没有安装新包、改变全局配置或取消采集范围。

采集后timeline_verify.py要求capture完成、日志无明确错误、op目录有非空CSV/bin结果，
否则即使msprof op返回0也停止。不把app源码或metadata当作profile数据。
有效数据存在也不等于已有足够逐核字段，后续仍按inventory核对能力覆盖。

服务器重试相同load命令，无需重建AscendC；先不运行pipeline。
