# 第二次采集：原生 MSTX 仍未启动

证据：Q4 提交 a7a50a0 的 error.txt，目录 timeline-load-20261029-170940-2520119。

- tilelang_s2buf 精度通过，relative L2 0.00242606697，max abs 0.00048828125。
- 原生 MSTX 已从 CANN lib64/libms_tools_ext.so 加载，但 range_start 未返回有效 ID。
- 未执行区间内目标调用；没有非空 profile 数据，不能据此分析逐核负载或更新加速比。
- 工具仍返回 0，新增结果校验成功拦截并在第一侧停止，未再次误报成功。
- 现有日志不足以确定 MSTX 回调为何未启用，也不能证明 main_kernel 名称错误。

下一步绕开 MSTX，使用安装工具帮助已提供的 kernel 名称过滤和单次 launch。
独立进程先检查精度；采集应用只调用一次指定 TL 实现，由工具执行 50 次预热和重放。
暂采 S2 与 KV2 两个 TL 实现，AscendC 名称与 metadata 排除规则尚待确认。
本机可验证控制流程，真实采集及 Ascend950 指标支持情况仍需服务器验证。
