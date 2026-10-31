# 原版 AscendC 两条路径计时

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

all 包含正确性、三进程 API 基准和五个形状的 profiler；每进程九轮重复采样，
每轮100次、预热100次。
若需同一设备会话的最新 TL/适配 AC 对照，另运行：

```bash
bash operators/mhc/tuning/run_split.sh bench
python operators/mhc/matched/collect_results.py --root results/mhc-split
```

复制两份汇总到 error.txt 即可。原版 API 为完整 PreV2 + Sinkhorn；
profiler 同时展示两个阶段各自的设备 Duration，不能与 API 延迟直接相加比较。
