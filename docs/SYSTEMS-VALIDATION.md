> Current release: [0.6.2 validation](RELEASE-0.6.2.md). The report below records the historical 0.6.0 checks.

# Systems refactor validation · 0.6.0

Validated locally on 2026-09-20 Pacific / 2026-09-21 UTC, using Linux, Node 22.14 and the installed VS Code executable. These are implementation and smoke checks, not a measured accuracy claim across arbitrary repositories.

## Results

| Check | Result |
|---|---|
| TypeScript | `npm run typecheck` passed. |
| Unit/regression tests | 92 passed, zero failures, including 48 preserved legacy tests and 44 Systems IR/evaluation tests. |
| Packaged VS Code host, controlled Jev | Passed, 9 recorded snapshots. |
| Packaged VS Code host, real Gateway/Jev | Passed, 8 recorded snapshots; 5 Jev requests and no generated reviews. |
| Sidebar browser test | Passed against the production renderer and a fixture produced by the actual static analyzer. |
| Live evaluator smoke corpus | 8/8 scenario expectations passed. Policy cases may remain unknown or suppressed; these are not eight proven performance defects. |

`npm test` passed all five test files. A second run with `node --import tsx --experimental-test-isolation=none --test test/*.test.ts` recorded all 92 individual cases in `artifacts/unit-tests-0.6.tap`; the default child-process reporter in this environment printed file-level totals only.

The packaged host exercises automatic CUDA serial-tail detection, correction to a shuffle reduction, inline diagnostics/CodeLens/hover, source evidence, cache reuse, stale-request rejection, consent gating, authentication recovery, rate-limit recovery, incomplete syntax, five Python/framework workflows and pause cleanup. Optional review/handoff uses a controlled response and leaves source unchanged. No paid live Astra review was requested in this validation.

The browser test checks exact evidence disclosure, confidence, next checks, requirements, unmeasured labels, positive structures, separate assumptions, source navigation, review/handoff controls, metadata-import intent, stale-state controls, escaped source content and a 260-pixel sidebar. The native host screenshot confirms editor decorations and the sidebar in VS Code. The OS file-selection dialog for evidence import is not automated; schema validation, revision binding and renderer intent are covered separately.

## Precision controls

Positive and negative examples cover serial/shuffle/barrier CUDA reductions, global versus shared memory, changed addresses, one-iteration loops, guarded access, launch-width assumptions, host transfers, JIT boundaries, shape-dependent branches, collective types, materialization, sharding round trips, overlap candidates and explicit serving policies.

Unsupported evidence is tested explicitly: comments and strings, unrelated methods, shadowed imports and parameters, visible monkey-patching, reassigned tensors, distinct producer/consumer values, unknown callees, nested function definitions, dead branches and truncated context. Compiler/runtime quantities require matching provenance. Execution assumptions cannot become a positive performance rating.

The first live smoke passes exposed two integration issues: splitting model support between compatible `retain` and `likely` outcomes made a supported concern appear unknown, and asking the model to veto an obvious serial structure mixed structural certainty with workload importance. The final implementation preserves the full model distribution, combines compatible concern probability and keeps that direct serial observation independent of a model verdict. Earlier smoke results remain in the artifacts directory for comparison.

## Observed extension overhead

Local block/context preparation plus Systems IR and static lenses, 80 synthetic edit samples per case without report-cache reuse:

| Fixture | Characters | Median | p95 |
|---|---:|---:|---:|
| CUDA fragment | 595 | 0.9 ms | 2.3 ms |
| Tensor loop | 112 | 0.3 ms | 0.7 ms |
| 250-function Python file, focused block | 28,902 | 4.7 ms | 8.1 ms |

These figures exclude the default 700 ms typing pause, UI rendering and cloud requests. They measure the extension's overhead, not the performance of the analyzed program. The live smoke corpus observed 222–625 ms for its five actual Jev requests; the direct serial, shuffle and CPU-control scenarios made no model call. Network timings vary and are not a service guarantee.

## Reproduction and artifacts

```sh
npm run typecheck
npm test
npm run package
npm run test:host -- --vsix artifacts/kernel-lens-0.6.0.vsix --scenario systems --capture
npm run eval:live -- --env-file ~/.env
npm run test:host -- --vsix artifacts/kernel-lens-0.6.0.vsix --scenario systems --live --env-file ~/.env --capture
npm run bench
npm run preview
# In another terminal, with the preview running:
npm run test:ui
```

Local artifacts:

- `artifacts/unit-tests-0.6.tap`
- `artifacts/systems-host-mocked-0.6.json`
- `artifacts/systems-host-live-0.6.json`
- `artifacts/systems-live-0.6.json`
- `artifacts/systems-live-0.6-first-pass.json` and `systems-live-0.6-second-pass.json`
- `artifacts/systems-local-latency-0.6.json`
- `artifacts/systems-native-0.6.png`
- `artifacts/kernel-lens-sidebar-0.6.png`, `kernel-lens-improved-0.6.png`, `kernel-lens-narrow-0.6.png`
- `artifacts/systems-tested-runtime-0.6.json` and `editor-release-verification-0.6.json`

The VSIX contains the bundled runtime and UI; test tooling and local evidence artifacts are excluded. The separate source ZIP includes the implementation, tests and documentation. No Marketplace publishing, remote deployment, CUDA compilation, Compute Sanitizer run, workload profiling, or Cursor/Remote-SSH validation is claimed.

See [implemented coverage](SYSTEMS-COVERAGE.md) for the supported structures and current boundaries.
