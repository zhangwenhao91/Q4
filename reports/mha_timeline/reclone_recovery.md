# 删除仓库并重新克隆后的恢复检查

42f0d2d:error.txt 确认 export_timeline.py 找不到旧采集输入目录。
当前远端没有 reports/mha_timeline/captures 中的导出文件；此前服务器提交 495f70b
未成功推送。普通重新克隆不能找回仅存在于被删除旧仓库中的提交和未跟踪文件。

保留的内容：已推送的脚本、补丁、报告，以及 Git 历史中的 error.txt 和精度/耗时记录。
可能丢失的内容：未上传的 trace.json、原始 dump，以及被 .gitignore 排除的
results/、third_party/。后者包含固定源码、q4fa 算子包和编译缓存，需实际检查。
zwh 环境位于仓库外，删除 Q4 本身通常不删除它；但 editable 安装仍可能指向已删除源码。

先执行只读检查，不直接重装环境：

```bash
cd /home/z30086261/Q4
git pull --no-rebase
python operators/mha/tuning/check_reclone.py
```

把输出更新至 error.txt。如果删除前有移动旧目录、打包或文件系统快照，应保留并告知路径，
可优先恢复采集数据。若旧目录确实删除且无备份，需恢复缺失依赖后重新采集；
旧时间轴无法从摘要日志重建，也不能把新采集冒充为旧采集。
重新采集将生成新的时间戳目录，导出时必须使用新路径，不能继续指定旧目录名。

后续按检查结果只恢复缺失项：固定 TileLang 源码、固定 ops-transformer 源码与
部分编译生成的 FlashAttn/FlashAttnMetadata 算子包、必要的本地扩展缓存。
不把未知环境状态当作重新安装全部 CANN/PyTorch 的理由。
