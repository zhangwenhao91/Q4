# 删除仓库并重新克隆后的恢复检查

## 已确认状态与恢复命令（2650e36）

zwh 的 torch 2.13.0+cpu、torch-npu 2.13.0、tilelang 0.1.15+cpu.gita7cb753d、
cann-ops-transformer 1.0.0 均仍有安装元数据。后两者不是 editable 安装，
安装来源路径消失并不代表已安装文件消失；恢复脚本先实际导入 torch/torch_npu/tilelang 检查。
固定源码、q4fa vendor 库、扩展缓存和采集结果全部缺失。
ASCEND_CUSTOM_OPP_PATH、TORCH_EXTENSIONS_DIR、ASCEND_RT_VISIBLE_DEVICES、ASCEND_NPU_ARCH 尚未设置。

已补齐 JSON、Protobuf、Abseil CVE 补丁三个离线文件，全部与既有固定 SHA256 一致。
本机已验证五项依赖完全不依赖旧目录即可播种，shell 语法检查通过；尚未在 NPU 服务器构建。

```bash
cd /home/z30086261/Q4
git pull --no-rebase
conda activate zwh
source /home/CANN/9.3.0.weekly.20260930/ascend-toolkit/set_env.sh
bash operators/mha/recover_reclone.sh
```

恢复脚本导入检查失败、下载失败、构建或安装失败时立即停止。它恢复 bundled TileLang
源文件，克隆 ops-transformer，并使用旧的固定提交 ba5162411c919e15fbabb28298e8b420c11ad905
和既有构建脚本部分编译 flash_attn,flash_attn_metadata，再安装至 Q4-local q4fa vendor。
不升级或重装 Python 包。完整构建日志保存在 results/mha-recovery/，首次编译需要时间。

仅在显示 `MHA source and native package restored` 后执行：

```bash
source operators/mha/env_recovered.sh
bash operators/mha/tuning/run_timeline.sh pipeline
```

环境脚本默认恢复之前使用的物理设备 4（进程内逻辑 0）；若已指定其他非空设备选择则保留。
pipeline 仍先检查 S2、KV2 精度，再采集，首次调用将重建缺失的 native wrapper 扩展缓存。
将完整输出更新至 error.txt；使用新输出目录导出，不能再使用已丢失的旧目录。
新采集完成前，不将旧报告中的精度和性能结果当作重建后的验证结果。

## 最初诊断与恢复边界

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
