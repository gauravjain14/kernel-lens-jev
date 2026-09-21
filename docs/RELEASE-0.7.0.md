# Kernel Lens 0.7.0

Kernel classification now uses 24 independent axes and named execution buckets. Compute path, lane participation, global access, transfer width, reuse, shared layout/staging, copy/compute scheduling, buffering, MMA completion cadence, async protocols, synchronization, accumulator placement, register pressure, shared footprint, launch parallelism, instruction costs/dependencies, reductions, epilogue, atomics, control flow, compute/traffic balance and execution assumptions are assessed separately.

The CUDA adapter recognizes pipeline copy/commit/wait operations, WMMA, inline MMA/WGMMA/tcgen05, TMEM operations and mbarrier protocols. Unique reachable same-file helpers contribute bounded instruction contracts with original source evidence. Names alone never establish helper behavior. Async copies carry source/destination addresses; calls carry arguments and enclosing control relationships. Shared allocations retain all dimensions and element width; unresolved dimensions remain symbolic. Loop-counter increments no longer count as reductions.

Jev chooses semantic buckets directly. Probabilities are no longer split over four competing source anchors. Supported buckets retain deterministic source evidence; unsupported or uncertain choices remain unknown. The complete current function, operation sequence, helper instructions, allocation facts and assumed hardware inform one request. Helper contracts are bounded to 7,000 serialized characters and execution summaries to 180 operations; omitted operations are explicitly counted. Unknown callees, external templates and unresolved layouts remain limitations, not invented facts.

The sidebar displays individual buckets by pipeline, compute, memory and resources. Each expands to evidence, uncertainty, requirements, an actionable check and the relevant Nsight Compute section. A function-level CodeLens summarizes the current kernel. Execution assumptions remain separate. Runtime impact remains unmeasured unless supplied.

Training/inference analysis is preserved. This release does not attempt a complete CUDA compiler, full PTX interpretation, inferred runtime occupancy or automatic profiling.

## Validation

- TypeScript checks and the local regression suite pass. New cases cover helper resolution, unsupported evidence, all 24 axes, source provenance, late operations, symbolic allocation sizes and cache invalidation.
- The packaged VS Code check passes: six distinct kernel classifications, a visible function summary, complete source across all 233 body lines, stable caching and unfinished-edit handling. The sidebar browser check passes for visible buckets, expandable evidence, source navigation, stale-state controls and a 260-pixel layout.
- The first successful live Jev request on `phase1_cpasync_2stage.cu` produced 15 classifications, including Blackwell MMA, elected-thread issue, vector transfers, shared reuse/swizzle, two storage stages, per-tile MMA completion waits and repeated TMEM epilogue waits. That request took 764 ms and reported 22,326 input tokens. These are request observations, not a latency guarantee.
- The same live session retained the RMSNorm serial-reduction finding. Intermittent Gateway HTTP 503 responses interrupted the remaining live cases and a later check after adding explicit call/copy arguments. Full live-suite validation is therefore incomplete; the deterministic and controlled-host checks are distinct from live-model validation.

Reproducible checks:

```sh
npm run typecheck
npm test
npm run eval:kernels -- --env-file ~/.env
npm run test:host -- --scenario cuda-asm --vsix artifacts/kernel-lens-0.7.0.vsix
node --import tsx scripts/prepare-kernel-preview.ts
```

CUDA semantic references: [PTX issue granularity and instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instructions), [asynchronous copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html), and [Nsight Compute sections](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Install or share

```sh
code --install-extension artifacts/kernel-lens-0.7.0.vsix --force
```

Run **Developer: Reload Window**. Saved credentials and workspace permission carry over. Recipients install the same VSIX and use their own Gateway key. No Marketplace publication is performed.
