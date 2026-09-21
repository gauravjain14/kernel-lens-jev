# Kernel Lens 0.6.2

Fixes false “Finish this block” messages in CUDA code using GNU extended inline assembly, including the downloaded `phase1_cpasync_2stage.cu` regression fixture.

The C++ grammar previously interpreted `asm volatile(...)` as a function declaration. Error recovery also corrupted string masking, so PTX braces could affect the block-completeness check. The extension then withheld the Jev request even when the enclosing kernel was closed.

The parser now masks C++ literals independently of grammar recovery and normalizes complete assembly statements as opaque calls for structural parsing. Original PTX, operand constraints, source locations and line numbers are preserved in the evidence and Jev context. Assembly remains an unknown side-effect boundary; this patch does not claim full PTX semantics or compilation correctness.

Opening a CUDA file on comments or constants selects a nearby kernel instead of a partial line window through a helper. Complete functions receive priority within the existing context budget. The supplied kernel's entire 233-line body now fits without truncation.

Regression tests cover every cursor line in the 334-line file, stable kernel context, raw strings, quoted PTX braces, GNU asm qualifiers and constraints, comments, numeric separators, memory side effects and incremental edits. Unfinished strings, comments, constraints and braces still wait. The local suite passes 105 tests.

Both controlled and live Jev runs passed in the packaged VS Code extension. The live run made one request, returned reuse/overlap/matrix-instruction classifications, and made no additional requests while traversing all 233 kernel lines. See `artifacts/cuda-asm-host-live-0.6.2.json`.

The dedicated VS Code host scenario verifies that opening this exact source triggers an evaluation request automatically, visits every kernel body line without extra requests, checks a PTX helper, and confirms incomplete assembly does not trigger evaluation:

```sh
npm run check
npm run package
npm run test:host -- --scenario cuda-asm --vsix artifacts/kernel-lens-0.6.2.vsix
# Optional: one real Jev assessment, using the existing Gateway key.
npm run test:host -- --scenario cuda-asm --vsix artifacts/kernel-lens-0.6.2.vsix --live --env-file ~/.env
```

The inherited performance review, hardware profiles, precision controls and Gateway-outage behavior are documented in [0.6.1 validation](RELEASE-0.6.1.md). This patch addresses readiness and context handling; undefined external symbols or unknown hardware workload details remain separate analysis limitations.
