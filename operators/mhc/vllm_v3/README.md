# Pinned upstream vLLM-Ascend HcPre v3 comparison

```bash
bash operators/mhc/vllm_v3/run.sh all
```

Requires the existing zwh/CANN/TileKernels/DeepGEMM environment. This builds only
HcPre from vLLM-Ascend e838a32888113cd20e1b5bad2053d1b4a06bf935 using its own
csrc/build.sh, in an isolated q4vllmv3 vendor. It does not install vLLM or upgrade
Python packages. modes: build, smoke, bench, all (default). Full logs go to
results/mhc-vllm-v3; error.txt is a compact result or failure report for copying
manually to GitHub. The build may download upstream framework dependencies.

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
