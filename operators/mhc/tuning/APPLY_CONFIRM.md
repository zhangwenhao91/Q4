# H1280小幅收益复核

```bash
bash operators/mhc/tuning/run_apply_confirm.sh smoke
bash operators/mhc/tuning/run_apply_confirm.sh all
python operators/mhc/matched/collect_results.py --root results/mhc-apply-confirm
```

仅复测H1280和H5120重复配置对照，不改数值kernel。性能N=64/128/256/512，
五个独立进程、32轮（四次完整四侧位置/正反顺序周期），每轮100次、预热100次。
候选进程顺序随rep奇偶反转，每进程都含默认两kernel、默认三kernel、候选与适配AC。
判断H1280收益应使用每进程同组默认三kernel/H1280比值，不能挑最快进程或跨组AC。
H5120候选与默认三kernel数值及配置相同，用于观察重复路径和分配/缓存噪声。
需要检查event与wall方向、五个进程方向、重复配置差值，再决定是否采用。
若收益与重复配置噪声接近，保持默认值，不宣称优化完成。

check覆盖五个N（含129尾数）和三个种子。profile分别采集两候选，
不以设备Duration和API相减作为精确host开销。
all预期78份正确性文件、40份API汇总、8个profile目录。
不再测试已确认退化的8192，不引入通用N分派阈值；旧pre四输出、
精度门限、每次重新计算和分配边界均保持不变。
当前默认实现仍未修改。
