# 经上游 SHA256 校验的离线构建依赖

保存固定构建所需的原始源码归档和补丁，没有修改或重新压缩。归档内保留上游许可证。
本机从下列官方 CANN 镜像下载，TLS 校验正常，SHA256 与 cann/cmake master-059 的 third_party 模块一致。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| eigen-5.0.0.tar.gz | 2963022 | 93f7f0462988b934e632a9fba58af55192ffceae38e8f46233f4f62cb1e79370 |
| abseil-cpp-20230802.1.tar.gz | 2157158 | 987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed |
| json-3.12.0.tar.gz | 9678593 | 4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187 |
| protobuf-25.1.tar.gz | 5878124 | 9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42 |
| backport-CVE-2025-0838.patch | 4618 | a4e4ef6fc4f4c1ed871a3472c5e650068aaadaad2adfef2c8ccb00918bb36d8c |

来源：
- https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/eigen/eigen-5.0.0.tar.gz
- https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/abseil-cpp/abseil-cpp-20230802.1.tar.gz
- https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/json/json-3.12.0.tar.gz
- https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/protobuf/protobuf-25.1.tar.gz
- https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1-h0/backport-CVE-2025-0838.patch
- https://atomgit.com/cann/cmake/blob/master-059/third_party/eigen.cmake
- https://atomgit.com/cann/cmake/blob/master-059/third_party/abseil-cpp.cmake

seed_dependency_cache.py 优先检查目标缓存及旧构建缓存；缺少归档时使用这里的副本。复制前后均验证 SHA256。
2026-10-30 删除并重克隆后，旧缓存已缺失，因此补入 JSON、Protobuf 和 CVE 补丁。
这五项现在均可从本目录恢复。无需关闭任何 HTTPS 证书校验；源码克隆及构建工具其他依赖仍可能访问网络。
