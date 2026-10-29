# P的L1缓冲显式双版本实验

按优先级先检查P_shared，直接对照已验证的KV2最佳组合：
nostore + stages2 + 最终输出融合 + K/V版本2，S/alpha仍3版本。
本轮只在两处buffer注解中增加P_shared:2，所有计算宏及其他版本保留。
不叠加首块PV、Q常驻或缓存策略改动。

原始注解没有显式P_shared版本，不能据此断言编译器一定使用单版本。
较早nostore生成代码的P在固定L1位置，但当前最佳还需独立核对。
检查两侧生成C++中UB->L1写入P、L1->L0A读入P的地址及索引是否随迭代交替，
确认增加P槽位；地址偏移/循环结构的变化也可能伴随调度变化。
完整C++位于third_party/mha-pbuffer-candidate/，源码/生成代码哈希记录于metadata.json。
终端打印相关CODEGEN行和CODEGEN_IDENTICAL_PBUFFER布尔值。
如果两侧C++完全相同，不应将计时波动视作缓冲优化收益，也不需要继续bench。
C++不同不等于机器指令一定不同或已经提速，最终仍靠NPU计时。

仍为B1/H1/S4096/D128、BF16输入输出，AscendC复用metadata。
四侧：原始TL、当前KV2最佳、P双版本候选、原生AscendC。
3种子检查；计时3进程、各8轮、每轮100次，四侧执行位置平衡。
CPU参考及全部两两精度门限不变。

## 服务器

无需重建AscendC，沿用当前zwh/CANN/q4fa环境：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_pbuffer.sh check
```

若编译或精度失败，停止并反馈日志；若CODEGEN_IDENTICAL_PBUFFER为True，
也先反馈check输出，不继续计时。其余情况全部精度通过后执行：

```bash
bash operators/mha/tuning/run_pbuffer.sh bench
```

将含CODEGEN行的输出更新到error.txt。
比较matched_best_over_pbuffer；确认API收益后再profile。
后续按优先级独立测试Q/O的L2策略，不与本轮混在一起。
