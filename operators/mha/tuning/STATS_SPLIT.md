# 分别检查S和alpha的buffer版本数

按第三优先级定位此前S/alpha同时降为双版本时的退化。
直接对照当前已验证最佳：nostore+stages2+输出融合+K/V双版本。
保留默认Q/O缓存策略、P原配置；不叠加Q常驻、首块PV或P双版本。

| 标签 | S_ub版本数 | alpha_ub版本数 |
|---|---:|---:|
| tilelang_kv2，当前最佳 | 3 | 3 |
| tilelang_s2buf | 2 | 3 |
| tilelang_alpha2buf | 3 | 2 |

S_ub保存QK分数，alpha_ub供后续输出累积读取，二者生命周期不同。
本轮分别更改，避免将此前的组合退化直接归因于某一个buffer。
原始TL与原生AscendC也保留，同轮五侧。B1/H1/S4096/D128、BF16输入输出、AC复用metadata。
3种子检查，每进程15项精度记录；3计时进程、各10轮、每轮100次，执行位置平衡。

服务器无需重建AscendC，沿用zwh/CANN/q4fa环境：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_stats_split.sh check
```

全部精度通过后：

```bash
bash operators/mha/tuning/run_stats_split.sh bench
```

任一候选编译或精度失败即停止，反馈完整日志。
完整生成C++位于third_party/mha-stats-split-candidate/；源码和生成代码哈希记录于metadata.json。
CODEGEN打印流水循环结构供核对，不据此直接推断性能。
将包含CODEGEN的输出更新到error.txt。仅在出现API收益后再profile确认设备时间。

## API结果后设备复核

90项精度通过。S双版本观察到小幅API改善，alpha双版本单独导致明显退化。
设备复核仅测原始、KV2最佳、S双版本和AscendC：

```bash
bash operators/mha/tuning/run_stats_split.sh profile_best
```

自动打印设备CSV，反馈error.txt；无需重复check/bench或重建AscendC。
