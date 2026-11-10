# PR3896 MHA性能验证

固定PR：https://github.com/tile-ai/tilelang/pull/3896

源码提交：`86dccfce8266818a7f6278de84b5c28b9851fa46`。
PR提交的三个示例文件及LICENSE已下载到upstream，source.json记录SHA256；服务器无需访问GitHub下载PR。
仅替换测试用例源码，不重新安装TileLang编译器。若现有编译器不支持PR使用的动态接口，应先保留错误日志，再判断是否升级。

## 对比口径

- B=1、H=1、D=128、S=4096和3584，非因果，无dropout。
- 四条路径：此前固定原版TL、PR静态TL、PR动态TL、固定AscendC FlashAttn。
- 四方均BF16输入、内核直接BF16输出。PR示例mha入口原本指定FP32；本测试直接使用PR原生core的BF16输出选项，以匹配AC。这是BF16性能比较，不是默认FP32入口比较。
- 原版TL固定80364b3d；它是此前基线，不冒充PR的直接父提交。因此旧TL/PR包含两个固定版本间的全部差异。
- 每个进程动态内核只编译一次，先运行4096，再复用于3584；编译不计入耗时。本轮不覆盖非整块长度的性能与尾块验收。
- AC固定ba516241，分别测metadata缓存和每次生成。AC/TL>1表示TL更快；旧TL/PR>1表示PR更快。
- 精度沿用Q4：CPU FP32参考计算后转BF16，relative L2<=0.01、max_abs<=0.02，检查dtype、shape和有限值。各实现均通过后才测性能。
- 每个metadata模式测3个独立进程，每进程预热100次、10轮、每轮100次；保留Event与wall耗时，汇总Event进程中位数及逐进程加速比。
- 记录实际设备，不将28核测试当作32核950DT结果。

## NPU服务器执行

沿用已有zwh、CANN和q4fa环境。在Q4目录执行，先确保CANN的set_env.sh已加载；可见卡号选择实际空闲设备，以下示例为物理卡4。

```bash
conda activate zwh
git pull --ff-only
export ASCEND_RT_VISIBLE_DEVICES=4
bash operators/mha/pr3896/run.sh smoke
```

smoke成功后执行：

```bash
bash operators/mha/pr3896/run.sh all
```

末尾会打印结果目录，将其中summary.txt内容粘贴到error.txt。若中途失败，提供runner.log末尾的完整异常，先不要重装环境。

可在无NPU机器上重新汇总已上传的原始结果：

```bash
python operators/mha/pr3896/summarize.py --input results/mha-pr3896/<实际目录>
```

本地仅完成源码获取和脚本静态检查，NPU兼容性、正确性与加速比待服务器验证。

## 非整块长度

新增长度：1、63、64、65、127、129、193、255、3583、3585、4095、4097。
同一进程中单次编译的PR动态内核复用全部12个长度；输入为真实长度，不做Host补齐或输出转换。旧TL和PR静态路径不支持这些长度，不参与本轮。AC按真实长度生成metadata，仍分别测试cached及per_call。

```bash
bash operators/mha/pr3896/run.sh tails-smoke
# 上一步全部精度检查通过后：
bash operators/mha/pr3896/run.sh tails-all
```

tails-all预期12个长度、每个长度50项输出精度记录，共600项；每个长度6份性能汇总。若某一长度的AC或TL不支持，保留完整异常，不以补齐实现替代。本轮检查返回张量数值、形状及类型，不等同于内存越界检测器验证。
