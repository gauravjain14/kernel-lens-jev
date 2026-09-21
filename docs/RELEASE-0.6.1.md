# Kernel Lens 0.6.1 release validation

This beta restores whole-function review and adds hardware-aware fixed-dimension Jev predictions across kernel, training and inference paths. It retains deterministic execution facts and the evidence gate. No GPU workload performance or optimal implementation is claimed.

## Release cases

| Case | Required result |
|---|---|
| User's RMSNorm kernel | Serial tail and reduction barriers; additional source-based investigation; no shared-memory-as-global-output false positive. |
| User's coalesced scalar GEMM | Limited explicit operand reuse and scalar contraction investigations; local accumulation recognized. |
| Warp shuffle reduction | Shuffle positive; no serial-tail or repeated-global-output issue. |
| Training loop | Recreated DataLoader iterator, per-step host reads and graph-attached logging. |
| Improved training loop | Iterator advances normally; detached logging; no old iterator/host-read/retention concerns. |
| Inference loop | Host round trips and serialized per-request work. |
| Improved batched inference | Inference-mode positive; old transfer/retention/batching concerns absent. |

Live results: `artifacts/release-live-0.6.1.json`. All seven case contracts passed. Jev findings can vary; contracts test required structures and removal of specific concerns, rather than exact prose or probability snapshots. The adapter identifies the DataLoader iterator fact independently; Jev classifies higher-level concerns such as host synchronization and graph retention.

## Local checks

- 100 unit tests pass, including both user kernels at every line in the function, decorator scope, shared-memory classification, unsupported evidence suppression, model evidence gating, hardware cache changes, known-shape roofline math and iterator controls.
- The real sidebar renderer passes Playwright checks for evidence, confidence, runtime labels, separate assumptions, hardware selection, escaping, stale-state handling and a 260px viewport.
- The packaged release host passes eight scenarios using recorded Jev responses: all seven code cases plus hardware-change cache invalidation. Every body line is visited without dropping findings or calling Jev again.
- The controlled VS Code host passes nine workflow snapshots covering diagnostics, CodeLens, hovers, cache reuse, stale response rejection, auth recovery, provider cooldown, overload fallback, retry recovery, incomplete edits and pause cleanup.

## Performance and limits

The seven live case results use 10–13 questions per request. Most successful responses in this run completed in roughly 0.3–1.0 seconds; a training recheck took 3.6 seconds. Gateway also returned 503 and 529 overload failures. A final live VSIX run passed the two kernels before failing on training with a 503; another rerun encountered a 529. The live packaged suite therefore did not complete cleanly, despite the separate seven-case live suite and recorded-response packaged suite passing. The extension retries once automatically; if the provider remains unavailable it shows direct source observations, keeps unassessed model hypotheses hidden, and leaves Analyze available for retry. These are observations, not a service latency guarantee. The default typing pause is an additional 700ms.

Synthetic local preparation, including IR and payload generation, had median times of 0.6–4.5ms and p95 of 1.3–7.9ms across a small tensor loop, CUDA fragment and 250-function file. See `artifacts/systems-local-latency-0.6.1.json` for scope and environment.

Hardware profiles are explicit assumptions with published ceilings. Numerical roofline estimates are limited to compatible, explicitly shaped GPU FP32 matrix operations. The beta does not infer exact occupancy, stall percentages, achieved throughput, arbitrary model shapes or distributed topology. Supported framework semantics and false-positive controls are documented in [coverage](SYSTEMS-COVERAGE.md).

## Reproduce

```sh
npm run check
npm run eval:release -- --env-file ~/.env
npm run package
npm run test:host -- --scenario systems --vsix artifacts/kernel-lens-0.6.1.vsix
npm run test:host -- --scenario release --vsix artifacts/kernel-lens-0.6.1.vsix
npm run test:host -- --scenario release --vsix artifacts/kernel-lens-0.6.1.vsix --live --env-file ~/.env
npm run preview
# With the preview running:
npm run test:ui
```

The release host's controlled mode replays distributions from the live case artifact, testing controller/rendering integration without further model charges. Live mode calls Gateway for the two user kernels, training and inference. It verifies cursor navigation does not trigger new calls. Hardware change invalidation is tested in controlled mode.

Public Marketplace publishing, CUDA execution/profiling, Cursor and Remote-SSH validation are outside these checks. Share the VSIX for installation in VS Code; each recipient supplies a Gateway key.
