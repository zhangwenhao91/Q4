# Q常驻L0A实验

直接对照当前最佳nostore + stages2 + 最终归一化/BF16转换融合。
单head、4096x128，BF16输入输出，原生AscendC复用metadata。

原始自动调度qk宏每轮KV都将同一Q分块从L1复制到L0A，每个Q分块32次。
本轮将该复制移至Q从GM进入L1之后、KV流水线之前，目标是每个Q分块只复制一次。
源码仍分别声明QK和PV的L0A buffer，其他算术及buffer版本不变。
实际驻留和同步由编译器决定；更长生命周期可能影响资源安排，不承诺提速。
仅测试自动调度；manual路径未作为本轮性能配置。

对照四侧：原始TL、当前最佳tilelang_finalize、新候选tilelang_qresident、原生AscendC。
精度检查仍为CPU参考和全部两两比较；3种子，每进程10项。
计时3个进程、每进程8轮，每轮100次，四侧在每个位置各出现两次。
比较matched_best_over_qresident，大于1才表示超过当前最佳。

## 服务器

沿用zwh/CANN/q4fa环境，无需重建AscendC：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_qresident.sh check
```

脚本输出CODEGEN行，列出候选与对照的L1->L0A搬运和KV循环位置。
完整device C++保存到third_party/mha-qresident-candidate/，哈希记录在metadata.json。
这些行辅助检查，不能仅凭行号证明最终机器指令数或物理寄存器用量。

全部精度检查通过后：

```bash
bash operators/mha/tuning/run_qresident.sh bench
```

编译或精度失败即停止，将包含CODEGEN行的输出更新到error.txt。
若API有收益，下一步再用profile复核设备耗时。
