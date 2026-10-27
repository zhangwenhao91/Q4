# 经上游 SHA256 校验的离线构建依赖

仅保存服务器旧缓存中缺少的两份原始源码归档，没有修改或重新压缩。归档内保留上游许可证。
本机从下列官方 CANN 镜像下载，TLS 校验正常，SHA256 与 cann/cmake master-059 的 third_party 模块一致。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| eigen-5.0.0.tar.gz | 2963022 | 93f7f0462988b934e632a9fba58af55192ffceae38e8f46233f4f62cb1e79370 |
| abseil-cpp-20230802.1.tar.gz | 2157158 | 987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed |

来源：
- https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/eigen/eigen-5.0.0.tar.gz
- https://cann-3rd.obs.cn-north-4.myhuaweicloud.com/abseil-cpp/abseil-cpp-20230802.1.tar.gz
- https://atomgit.com/cann/cmake/blob/master-059/third_party/eigen.cmake
- https://atomgit.com/cann/cmake/blob/master-059/third_party/abseil-cpp.cmake

seed_dependency_cache.py 优先检查目标缓存及旧构建缓存；缺少归档时使用这里的副本。复制前后均验证 SHA256。
JSON、Protobuf 和 CVE 补丁仍复用服务器已有的已校验缓存。无需关闭任何 HTTPS 证书校验。
