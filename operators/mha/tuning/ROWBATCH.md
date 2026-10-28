# 单遍softmax + 批量行统计

导出代码确认：rowfused将原始一次64-lane的alpha/l更新放进了32次循环，
每次两行，从而每个KV块执行64次alpha exp、64次l乘法/加法，而原始各一次。
这是生成C++中的结构性开销；不能仅凭源码证明寄存器溢出或精确分摊17.66%的退化。
固定证据：https://github.com/zhangwenhao91/Q4/blob/1d97603/error.txt

本轮保留单遍分数处理，恢复old_m快照及最终64-lane批量alpha/l更新。
行max和sum暂存UB，保留必要的VST_VLD屏障。概率BF16打包、输出和GEMM不变。
相比rowfused增加统计量UB访问，但消除逐行重复统计更新；收益仍需NPU验证。
Q128/KV128/stages3，自动调度，不叠加FMA或manual。保留原始实现作为固定对照。

服务器现有zwh/CANN/q4fa环境，无需重建AscendC：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
bash operators/mha/tuning/run_rowbatch.sh check
```

全部精度检查通过后：

```bash
bash operators/mha/tuning/run_rowbatch.sh bench
```

编译或精度失败则停止；将输出反馈到error.txt。API确认收益后再profile。

实测结果：36项精度通过，但比同轮基线慢约6.17%，不采用。详见reports/mha_rowbatch/report.md。
