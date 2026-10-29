# 逐核采集首次尝试：没有有效数据

日志来源：Q4提交7f57ca3的error.txt。

- 三个目标的独立精度检查通过。
- 在msprof op进程中也通过了CPU参考检查，但torch_npu的MSTX range_start路径返回无效值。
- 应用在目标采集区间开始前退出；工具报告Child process exited with return value 1及空结果。
- msprof op仍以0退出，旧shell脚本只检查退出码，错误打印了Collector completed并继续后两侧。
- 清单只有应用源码、metadata、日志等35个文件，没有有效profiler输出。

因此本次不能用于比较逐核负载、尾部时间或管线等待，不是新的性能测量结果。
无关kv_quant_sparse_flash_attention导入告警不是本次停止的直接原因。

## 修正

1. 独立使用CANN公开C ABI的mstxRangeStartA/mstxRangeEnd，通过ctypes调用，
   使用空stream的host范围覆盖唯一目标调用及设备同步。
   先从RTLD_DEFAULT解析符号，以保留采集器的预加载hook；未找到才加载CANN库。
   保持uint64范围ID，不在Python中将其截成32位，不需要重新构建算子或安装包。
2. 区间开始返回0仍拒绝无范围采集，打印所用backend帮助进一步定位。
3. 工具退出码为0后，再检查应用capture_succeeded、collect.log中的明确失败信息，
   以及op目录内非空CSV/bin数据。源码/日志不算采集结果。
4. 如果验证失败，立即停止其余目标，不报告成功，也不降级成普通profile。

这些修正通过10项本机检查，服务器采集效果待重试。

参考：

- [官方msOpProf扩展功能示例](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/extended_functions.md)
- [官方PyTorch所附MSTX C头文件](https://gitee.com/ascend/pytorch/blob/f5f187d722b876491c27dfdee7df1b1bf7eed5c2/third_party/mstx/ms_tools_ext.h)

官方单算子示例使用独立mstx范围和空stream。此次不把单算子CLI采集等同于开启torch_npu.profiler。
