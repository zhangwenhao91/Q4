# Pinned upstream vLLM-Ascend HcPre v3 comparison

```bash
bash operators/mhc/vllm_v3/run.sh all
```

Requires the existing zwh/CANN/TileKernels/DeepGEMM environment. This builds
HcPre and HcPost from vLLM-Ascend e838a32888113cd20e1b5bad2053d1b4a06bf935 using its own
csrc/build.sh, in an isolated q4vllmv3 vendor. It does not install vLLM or upgrade
Python packages. modes: build, smoke, bench, all (default). Full logs go to
results/mhc-vllm-v3; error.txt is a compact result or failure report for copying
manually to GitHub. The build may download upstream framework dependencies.

Source download uses the fixed-commit GitHub codeload archive, SHA256
98cef93d6983f9cb5b193e200a1e7b6b18ee11daef17eb35e2acfa432b8e82cc,
then verifies relevant extracted files against the verified archive on every
run. It avoids Git partial-clone lazy fetches that returned HTTP 403 on the NPU
host. The earlier failed clone is preserved; the new source directory ends in
`-archive`. This changes only source acquisition, not the pinned implementation.

The verified 19.6 MB archive is bundled in dependency_archives so NPU hosts
can obtain it through their working Q4 git pull. build.sh prefers this copy;
it does not invoke curl for upstream source or disable TLS checks. Additional
build-framework dependencies may still need their existing CANN caches/network.
If regex is missing, the build installs the bundled PyPI regex 2024.11.6 wheel
for CPython 3.12/Linux x86_64, with SHA256 verification and --no-index --no-deps.
An already importable regex package is kept. Other Python/platform versions
stop with an explicit dependency message instead of installing an invalid wheel.
The wrapper clears inherited Conda CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/ASMFLAGS
before mixed host/AArch64 builds and backs up/resets global flag entries in
root and tiling-sink CMake caches using Q4's existing clean_build_flags helper.
Compiler paths, target-specific options and upstream source remain unchanged.

The AscendC kernel, tiling and ACLNN adapter are unchanged upstream files.
prepare.py verifies their bytes against the fixed commit (allowing Git CRLF
checkout normalization), extracts original HcPre helper functions, v3 function,
and torch operator registration verbatim into a small extension, and adds only
includes/namespace and an API-address diagnostic. This avoids unrelated vLLM
runtime dependencies. The upstream EXEC_NPU_CMD header and NPU bridge/storage
sources are used unchanged. The original licenses remain in the fetched source
and extracted binding. Full-package vLLM extension loading is not measured.

The runtime checks source hashes, installed library/build records, and that the
upstream adapter resolves aclnnHcPre to the isolated library. This provides
source/build provenance; it is not a formal binary equivalence proof. Do not
import another vllm_ascend extension into the benchmark process.

Compare ascendc_vllm_v3, tilelang_coeff_bound, tilelang_all_token_h5120s3 on
identical freshly generated inputs for each N/seed. N=128/256/512/2048/8192,
three processes each, 100 warmup calls, 12 rounds x 100 calls. Ordering rotates
and reverses across rounds. Compilation/source validation/input prep/CPU
reference are excluded; every call's output/workspace allocation and dispatch
are included. No precomputed device result, output cache or graph replay.

The runner reports Pre alone, Post alone and a directly timed Pre + Post call.
Post alone receives identical CPU-reference post and comb coefficients on both
sides. The combined call feeds each implementation's newly computed coefficients
into its Post and checks the resulting residual in addition to all Pre outputs.
Both modes use the same synthetic BF16 layer output, generated outside timing.
The combined scope covers mHC only and excludes the intervening attention/MLP.

Semantic contract: BF16 x[N,4,5120], FP32 old pre[N,4,1], fn[24,20480], scale[3],
base[24]; old pre produces y while new pre/post/Sinkhorn coefficients are
computed from this call's x. norm_eps=1e-20, hc_eps=1e-6, 20 Sinkhorn iterations,
post multiplier 2. Upstream v3 returns y,post,comb,pre; metadata-only views and
tuple reordering adapt to Q4's y,next_pre,post,comb contract. No numerical adapter.

Upstream HcPre uses HF32 round-toward-zero; Q4 TL retains its existing projection
rounding. CPU and all pairwise checks retain Q4's original allclose gates
(BF16 rtol/atol .02/.02; FP32 1e-3/1e-4). Failure stops before timing, without
relaxing gates or silently substituting the previous Q4-modified AC library.

The local Windows checks cover source extraction, Python/Shell syntax, and
summary formatting. NPU build, H5120 execution, numerical checks and speedup
must be verified on the server; no result is claimed before that run.
