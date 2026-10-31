# 原版 AscendC 两条路径计时

新增未经修改的上游融合尾部测试（固定版本c1cf914）：

- tilelang_upstream_big_fuse_h4096：公开mhc_pre_big_fuse_fwd调用，含JIT分发。
- tilelang_upstream_big_fuse_bound_h4096：同一原版数值kernel，公开compile预绑定。

两侧都完整计入DeepGEMM-Ascend投影，每次两个kernel，不是仅计尾部。
两侧每次分配特征输出、打包系数输出、原始投影、平方和共四个张量。
原版公开wrapper及Ascend数值源码与固定版本SHA256核对，数值代码不修改。
投影使用现有Q4补丁版，因此整体标签为“上游原版尾部+DeepGEMM投影”。
N非16倍数时明确跳过上游两候选，未添加padding数值adaptor。
五个性能N均为16倍数，4096每进程四侧轮换顺序，16轮覆盖两个完整位置/顺序周期。
本轮all预期81份正确性文件、30份API汇总、25份profile。

新增 TileLang hidden=4096：在 mhc_original_fused 测试内，与原版融合4096共享
同一组设备输入，逐项CPU校验和两侧互验通过后交替计时。标签为
tilelang_current_pre_h4096，采用预绑定三kernel路径：投影、系数/Sinkhorn、新pre混合。
三个主输出与原版一致；内部pre为每次重新计算的工作区，不是预先给定或缓存的数值。
此路径使用新pre，与已有shifted TL5120的旧pre语义不同。
原版4096的profile和TL4096的profile分别采集，TL预期每次三个kernel。
当前API每进程16轮，支持四侧顺序平衡；all文件数见上文。
既有5120 shifted设备路径不变。

同一脚本同时运行原版融合 `PreSinkhorn`（hidden=4096）和原版
`PreV2 + Sinkhorn`（hidden=5120），N 均为64/128/256/512/8192。
smoke 先检查两条路径的 N=128；all 在每个 N 上先后测试两条路径，
分别保留三进程 API 基准及 profiler。

融合版使用上游扩展的 needBackward=False 路径，三个主输出，不输出 next_pre，
不保存反向缓存；5120 版保留四项输出及 PreV2 所需缓存。
固定 norm_eps=1e-20、hc_eps=1e-6、20次 Sinkhorn；均使用新 pre 混合输入。
扩展在计时前绑定，计算、输出及工作区分配均在计时内。
因此这是不同 hidden、不同原生输出边界的性能对照，不是相同规格的加速比。

```bash
bash operators/mhc/tuning/run_original.sh smoke
bash operators/mhc/tuning/run_original.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-original
```

原版 vendor 已安装则无需重建；缺失时执行 `bash operators/mhc/restore_vendor.sh`。
原版与 shifted vendor 在独立进程中加载，避免同名 Sinkhorn 动态库冲突。
固定上游提交 ba5162411c919e15fbabb28298e8b420c11ad905，并要求 tracked 源码干净。
记录已安装库 SHA256；源码检查和库哈希不单独证明该二进制的构建来源。

5120 路径测量未修改的 PreV2 + Sinkhorn。
PreV2 自身输出缓存与工作区分配均计时，不加入 Python 数值 adaptor。
PreV2 的 layer_input 使用本次新 pre；按此语义独立校验四项输出，
不能将它标为与 shifted 版本完全等价。权重、输入种子和随机生成顺序与 split 测试一致。
融合 PreSinkhorn 原版不支持 hidden=5120，其4096结果单独标记。

all 包含正确性、三进程 API 基准和五个形状的 profiler；每进程16轮采样，
每轮100次、预热100次。
若需同一设备会话的最新 TL/适配 AC 对照，另运行：

```bash
bash operators/mhc/tuning/run_split.sh bench
python operators/mhc/matched/collect_results.py --root results/mhc-split
```

复制两份汇总到 error.txt 即可。原版 API 为完整 PreV2 + Sinkhorn；
profiler 同时展示两个阶段各自的设备 Duration，不能与 API 延迟直接相加比较。
