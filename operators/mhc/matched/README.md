# 原规格 mHC：TileLang / 适配 AscendC

本实验只测试mHC前向，hidden=5120、4路流、24系数通道、BF16特征、FP32系数，
保留旧baseline及其vendor。AscendC侧是**根据固定上游适配的实现**，不称为未修改官方算子。
当前无本机NPU，设备编译、精度、动态N和性能均须依次在服务器验证。

## 对齐的计算边界

- Post：双方原生Post，输入输出按规格；batch=1的view不搬运数据。
- shifted Pre TL：旧pre apply + TileLang融合投影/平方和 + RMS + split + Sinkhorn。
- shifted Pre AC：新增原生MhcShiftedPreV2 + 原生Sinkhorn；在Pre内用旧pre计算layer_input，
  同时输出next_pre/post/logits。没有PyTorch cast/mul/sum数值路径，没有计算未使用的新pre h_in。
- norm_eps=1e-20，pre_eps=1e-6，Sinkhorn eps=1e-6，20轮，post乘数2。
- 双侧投影均为HF32模式；规格表只限定输入张量dtype，未要求全FP32乘法。
  不宣称位级等价；继续使用已有精度门限，失败不自动放宽。
- 所有设备计算和每次输出/工作区分配均计时；输入准备、CPU参考、原始投影检查和编译在计时外。

## AscendC适配内容

基于ops-transformer `ba5162411c919e15fbabb28298e8b420c11ad905`的mhc_pre，
生成独立 `mhc_shifted_pre` 目录/算子名/ACLNN符号，只注册950。
输入slot 4由可选gamma改为必需prev_pre[N,4]，fn按有效折入norm权重的布局传入。
内部沿用的gamma指针字段携带prev_pre，但hasGamma始终为0，不对投影应用gamma。
必需输出next_pre；invRms/hMix反向缓存不提供、不写出；保留算法必需的内部工作区与描述符。
先输出新pre，显式等待MTE3完成，再复用UB读旧pre供h_in计算，保留其他跨核同步。
没有为了优化跳过输入计算或预计算权重投影。

脚本校验固定提交、上游输入文件内容及生成目录；遇到未知修改即停止。
`third_party/mhc-matched-source/`保存生成源码哈希与adaptation.diff，编译源码保留上游许可证。
独立worktree/vendor不覆盖q4mhc或q4fa。

## 动态N路径

原Q4投影补丁的设备kernel已将平方和存储显式表示为[ceil(N/128),2,64]，
但公开wrapper对view使用了N/128并要求整块。
新实验不修改已安装源码，直接调用同一patched kernel：只对平方和输出分配补齐存储，
残差和投影输出仍使用真实N，输出有效平方和取前N项view。
底层既有tail加载/写出处理能否完整覆盖，必须通过新增N=1/17/127/128/129/513测试确认。
这是一条待设备验证的动态N适配，检查失败时不能声称已满足完整动态N规格。

## NPU服务器执行

沿用已恢复的zwh、CANN、TileKernels、DeepGEMM及其兼容补丁，不重装Python包。

```bash
conda activate zwh
cd /home/z30086261/Q4
git pull --ff-only
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4
bash operators/mhc/matched/build.sh
```

只在构建安装成功后执行：

```bash
bash operators/mhc/matched/run.sh check
```

check先跑N=512三种子，然后测动态尾部，再测全零残差、全零旧pre、单流选择旧pre。
每个新Pre case单独核查原始投影和平方和，再核查四项输出对CPU参考及双方互比。
默认仍为合成数据；可选 `export MHC_WEIGHTS=/absolute/path/weights.pt`，文件应包含
已折入norm权重的FP32 fn[24,20480]、scale[3]、base[24]，在计时外加载。
未提供真实权重时不得声称实际模型权重已验收。

check全部通过后：

```bash
bash operators/mhc/matched/run.sh bench
bash operators/mhc/matched/run.sh profile
```

bench为N=512/8192各3个独立进程，预热100次、9轮各100次；双方轮换先后顺序。
profile每条路径30次目标调用，需确认新增原生Pre与Sinkhorn实际出现；不预设kernel数量。
每个模式都先验证该case精度；all模式按check→bench→profile顺序执行。
结果在results/mhc-matched，末尾summary.txt可直接复制到远端error.txt，无需服务器push。
构建失败回传results/mhc-matched-build中的日志；检查失败回传结果目录runner.log。

本机验证仅包括源代码生成、上下文/哈希检查、Python/Bash语法和CPU合约测试；
AscendC与TileLang实际编译、运行以及收益尚未验证。
