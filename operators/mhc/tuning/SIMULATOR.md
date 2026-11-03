# Ascend950 apply仿真流水

按eb58cda:error.txt确认的工具参数构建：soc=Ascend950PR_9579，
PipeUtilization、dump=true、core-id、launch-count、timeout均来自服务器help。

```bash
bash operators/mhc/tuning/run_simulator.sh original
bash operators/mhc/tuning/run_simulator.sh all
```

第一条先验证仿真环境；第二条采集三个独立进程：原apply H5120、
嵌套1280三阶段、连续1280三阶段。kernel编译在启动前完成，目标只调用一次，
无预热/计时循环，无投影和系数尾部。输入由相同CPU种子产生，输出回CPU校验。
msprof仅按目标kernel筛选一次采集，默认只解析0号核，不代表所有核利用率。
源码、配置、哈希、CPU校验结果随仿真数据保存。

可用MHC_SIM_TOKENS、MHC_SIM_CORE、MHC_SIM_SOC覆盖默认128、0及已查询的soc名。
以服务器实际安装为准：本脚本不升级CANN/TileLang，不猜测替换simulator动态库，
不主动LD_PRELOAD未知库；若当前工具无法自动准备仿真环境，保留完整日志后按
报错定位缺失组件/库路径，不能声称已验证模拟器兼容。

结果在results/mhc-simulator，runner.log包含各进程控制台；复制至error.txt。
保留完整目录及visualize_data.bin，可导入MindStudio Insight检查指令流水。
仅凭控制台或kernel Duration无法证明MTE2/Vector/MTE3重叠。
仿真性能不可直接替换此前真机API加速比；主要用于定位事件等待和buffer生命周期。
本地只完成Python/Shell语法验证，仿真环境和数据解析必须在服务器验证。
