# 显式buffer版本数实验

基线为当前最佳nostore + stages2 + 最终归一化/BF16转换融合。
不叠加未观察到稳定收益的Q常驻L0A改动。
B1/H1/S4096/D128，BF16输入输出，原生AscendC复用metadata。

当前num_stages=2，但显式K_shared/V_shared/S_ub/alpha_ub仍各3个版本。
本轮分别测试版本数调整，而不将流水线阶段数等同于buffer版本数：

| 标签 | K/V版本数 | S/alpha版本数 |
|---|---:|---:|
| tilelang_finalize，当前最佳 | 3 | 3 |
| tilelang_kv2 | 2 | 3 |
| tilelang_stats2 | 3 | 2 |
| tilelang_buffers2 | 2 | 2 |

另保留原始tilelang_q128及原生AscendC，同轮共六侧。
QK/PV累加、概率打包、O_tmp等其他buffer版本保留，所有算术宏不改。
减少版本可能降低内存占用，也可能限制重叠、产生额外等待或被编译器拒绝。
实际分配和性能以生成代码及NPU验证为准。

本机检查修改范围、源码固定版本及执行顺序，不代表NPU编译/同步/精度通过。
每进程六侧对CPU参考及全部两两核对，共21项精度记录，门限不变。
计时3进程、各12轮、每轮100次，六侧在每个位置各出现2次。

## 服务器

沿用现有zwh/CANN/q4fa环境，无需重建AscendC。先执行：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_buffers.sh check
```

全部通过后：

```bash
bash operators/mha/tuning/run_buffers.sh bench
```

如任一候选编译或精度失败，立即停止并反馈完整日志，不跳过失败配置。
脚本输出各候选CODEGEN循环信息；完整device C++位于third_party/mha-buffers-candidate/。
与生成Python源码的哈希均记录在各结果目录metadata.json。
将包含CODEGEN的输出更新到error.txt。
判断是否有收益需比较matched_best_over_candidate，而不仅仅比较原始TL。
