# Hardware assumptions

The picker selects an assumed per-GPU hardware profile. It is not device detection or a measurement. All units below are decimal, using published SXM specifications. Tensor peaks are **dense**, not sparse.

| Profile | Memory GB | HBM TB/s | FP32 TFLOP/s | Dense BF16/FP16 TFLOP/s | Dense TF32 TFLOP/s |
|---|---:|---:|---:|---:|---:|
| B200, HGX/DGX | 180 | 8 | 75 | 2250 | 1125 |
| H100 SXM | 80 | 3.35 | 67 | 989.5 | 494.5 |
| A100 SXM | 80 | 2.039 | 19.5 | 312 | 156 |

B200 compute comes from the eight-GPU [HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/), divided by eight; tensor figures are additionally divided by two to convert sparse peaks to dense. Memory capacity and bandwidth come from [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/), divided by eight. Other sources: [H100](https://www.nvidia.com/en-us/data-center/h100/) and [A100](https://www.nvidia.com/en-us/data-center/a100/). Verified September 2026.

The source-level FP32 matrix estimate uses `2*M*N*K` FLOPs and `(M*K + K*N + M*N)*4` minimum tensor bytes. The ideal intensity is compared with the selected FP32 peak divided by HBM bandwidth. It assumes each input is read once and each output written once, with no TF32 conversion. Reuse, caches, actual traffic, library choice, small shapes and additional operations can invalidate a whole-program bottleneck inference. The UI therefore labels this an **ideal operator roofline**, not an achieved utilization or runtime prediction.

Unknown shapes/dtypes, incompatible matrix dimensions, CPU operands and unsupported operator semantics suppress this numerical estimate. Scalar CUDA code is never assigned a tensor-core peak automatically. Imported hardware disables built-in numerical roofline combinations to avoid mixing different devices' limits. Interconnect bytes are never converted into communication time without topology or runtime evidence.

The qualitative review still runs without shapes. It can identify serialization, limited explicit reuse, host synchronization, repeated setup and policy tradeoffs. `requires` describes what is still needed for stronger claims; a selected GPU alone cannot establish occupancy or workload parallelism.
