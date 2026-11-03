# Three-kernel N-pipeline experiment

Run in the existing zwh/CANN environment:

```bash
bash operators/mhc/tuning/run_all_token_simulator.sh
```

This is an experimental candidate, not a verified performance improvement.
The script compiles all kernels before launching one three-kernel mHC call.
There are no projection-check launches consuming the capture quota. It writes
CPU correctness checks, generated C++ for each kernel, projection configuration,
full logs and a compact report copied to error.txt. Simulation and numerical
validation require the NPU server. Keep full traces even after sharing error.txt.

- Apply: original full hidden tile 5120; three-stage pipeline across tokens.
- Projection: derived from DeepGEMM-Ascend commit
  3a35326271b069b8b5bf1a754435acd6cc91575b, file
  deep_gemm/tilelang_ops/hc_prenorm.py (LICENSE_DEEPGEMM). Its token-stage loop
  becomes two-stage T.Pipelined; existing hidden-block pipeline, split-K,
  square-sum accumulation and inter-core store barriers remain. Explicit AIV
  output storage follows the existing Q4 adapter. Installed dependencies are
  not modified. Nested pipeline compilation and synchronization need server
  validation; do not disable synchronization checks to force compilation.
- Coefficient tail: derived from coeff_tail.py (LICENSE_TILEKERNELS); replace
  Persistent N-tile traversal with explicit strided T.Pipelined traversal,
  keeping 16-token tiles, two buffer stages and numerical SIMD body.
  The original tail already requested Persistent pipelining; this change may
  generate equivalent scheduling and does not imply a new overlap benefit.

Default N=2048 gives more work per core than N=128/512, particularly for the
16-token coefficient tiles. Only positive multiples of 128 are accepted in this
experiment. Override with MHC_SIM_TOKENS; comparisons with previous runs must
use the same N. All cores are parsed so Cube/Vector core mapping is visible.

The three files show instruction pipelines within each kernel, not concurrent
execution of the three kernels. In Insight check MTE2/Vector/MTE3 for Apply and
tail; for projection also inspect transfer, conversion, Cube, FIX and C/V sync.
Actual overlapping instruction intervals must be inspected before concluding
that computation and transfers overlap across token blocks.
