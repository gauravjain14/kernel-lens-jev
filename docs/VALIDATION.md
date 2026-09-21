> Historical record through 0.5. See [current validation](SYSTEMS-VALIDATION.md) for the 0.6 systems refactor.

# Beta validation

Jev supplies semantic assessments. Local parsing selects context and reports syntax readiness. In 0.5, each model-selected risk also has a fixed consequence and next-check definition; these are not generated explanations or local semantic verdicts. Astra remains the optional generated-review path.

## 0.5.0 editor interaction and independent examples, September 20, 2026

The editor now displays gutter markers, concise inline findings, clickable CodeLens titles, and hovers with a consequence and next check. Jev can select the exact source line in a separate bounded request after the primary result is visible. Uncertain locations stay explicitly block-level. Strong concern-to-supported changes display a specific improvement; unknown or non-applicable outcomes never count as fixes. Sidebar probabilities, usage and the full profile are collapsed under Assessment details.

- **48/48 local regression tests** passed, including direct execution of all three test files. Strict TypeScript and the production bundle passed. New cases cover stable scope identity, historical edit isolation, real source candidates, low-confidence anchor rejection, conservative deduplication and improvement semantics.
- The packaged extension passed the mocked-Gateway native editor suite, including imported setup without a language server, delayed location cancellation, resuming location selection from a cached primary result, and existing permission/cache/error/review controls.
- The packaged **CUDA edit regression passed with both controlled responses and live Jev**: shuffle supported (0.92), serial replacement concern (1.00), restored shuffle supported (cached), and the completed serial inner block inside an unfinished kernel concern (1.00). The correction produced a specific improvement. The sidebar stayed closed and no Analyze or Astra call was used. See `artifacts/reduction-live-host-0.5.json` and `artifacts/reduction-live-editor-0.5.png`.
- The native workflow suite exercises **12 automatic snapshots** across disconnected training loss, optimizer lifetime, decode reuse, padding masks, functional dropout and compiled control. It verifies clickable findings and hovers with the sidebar closed, source locations, stale gating and specific improvements. The live variant invokes Jev without Analyze or Astra. Reports and native screenshots use the `-0.5` artifact suffix. One repeat stopped after four snapshots when Gateway returned HTTP 503 after the bounded retry; stale results stayed labeled and old editor annotations remained cleared. This service failure is preserved separately.
- **All six workflow pairs were validated with live Jev on the final runtime.** The final full run completed 11 snapshots before repeated provider cooldowns blocked the last compiled-code correction. That remaining pair passed in an isolated native run after the connection recovered. `artifacts/workflows-live-host-0.5.json` and `artifacts/workflows-live-host-0.5-focused.json` retain both runs; this is not a claim that the last full run completed without interruption. The 12-snapshot controlled-response workflow suite also passed.
- Across the nine successful uncached edits in the final full run, excluding initial setup and the known cache hit, **edit-to-primary-result was 1,490 ms median / 3,542 ms maximum**, including the default 700 ms typing pause. The source-detail stage completed at **2,019 ms median / 4,115 ms maximum**. These are extension-state timings; they do not measure the browser's paint timestamp. They exclude the rate-limited final edit, which remained pending beyond 150 seconds. A cache hit returned in 156 ms. Provider limits can therefore dominate the otherwise short response time.
- Browser checks passed at **380 px and 260 px**: primary findings, source excerpts, consequences, next checks, collapsed details, tentative states, improvements, review controls, source navigation, stale gating and literal HTML-like input rendering. The preview is explicitly labeled sample data and is excluded from the VSIX.
- Local context preparation for a synthetic **25 KB / 250-function file** measured **4.5 ms median / 7.3 ms P95**. The CUDA fragment measured **0.5 / 2.1 ms**. These are local parser/context times, not network or edit-to-result latency. See `artifacts/local-latency-0.5.json`.

An accelerated workflow test following the CUDA run reached the Gateway's rate limit on its tenth snapshot. Its 45-second test deadline expired during the server's requested 59-second cooldown; the extension remained in its automatic retry state. That record is `artifacts/workflow-failure-1789953814123.json`. The normal-budget run subsequently received repeated cooldowns on its final edit, preserved in `artifacts/workflow-failure-1789954053430.json`. Live workflow checks use the application's normal 30-request budget and allow up to 150 seconds for provider pacing. They do not waive prediction or interaction assertions. A later fixed-sample Gateway check passed, followed by the successful focused correction run.

```bash
npm run test:host -- --scenario workflows --vsix artifacts/kernel-lens-0.5.0.vsix --live --env-file ~/.env --capture
# Reproduce the isolated final pair:
npm run test:host -- --scenario workflows --workflow compiled_control --vsix artifacts/kernel-lens-0.5.0.vsix --live --env-file ~/.env --capture
```

VS Code 1.98.2 logged an internal `editor.contrib.inlineChatHints` line-number error during automated CUDA bulk replacements. The stack resolves to VS Code's built-in inline-chat hint code; both edit suites completed and their annotation assertions passed. This was not an exception in the extension bundle.

### Public-source audit

`scripts/prepare-external-audit.py` freezes pinned source URLs, source hashes, the rubric hash and expected labels before running Jev. Inputs use official [PyTorch MNIST](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/mnist/main.py), [PyTorch language generation](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model/generate.py), and [Triton fused softmax](https://github.com/triton-lang/triton/blob/84fa223cb8fe44f671a0fcd543236f44df975dda/python/tutorials/02-fused-softmax.py). Five controlled mutations remove a gradient reset, reverse update/backward, enable inference gradients, replace reduction padding, or remove a load mask. Two variants of the offline CUDA benchmark complete the **10 cases / 15 selected labels**.

**14/15 labels matched; 11 matched above the default 0.75 threshold.** The three tentative matches were MNIST's loss contract, evaluation behavior in language generation, and the original Triton masks. The missed case was the unmasked Triton load: Jev selected supported at 0.44, so it was not presented as a strong supported practice, but the expected risk did not surface. The three unmodified public examples produced no strong correctness concern. Their performance findings included blocking transfers, repeated host/device work and growing concatenations; their runtime impact was not established by this audit.

The questions and expectations were **not changed after this run**, and the missed prediction was not replaced by a rerun. `artifacts/external-cases-0.5.json` contains frozen inputs; `artifacts/external-results-0.5.json` contains all returned profile labels. This is a small independent check of our prompts, not a general accuracy rate or an assertion about Jev's training data. Most dimensions have no asserted ground-truth label. One transient 503 recovered during this audit.

```bash
python scripts/prepare-external-audit.py
npm run eval:live -- --env-file ~/.env --cases-file artifacts/external-cases-0.5.json --output artifacts/external-results-0.5.json
```

### Offline runtime check

`test/performance/reduction.cu` compares the user's serial 32-value reduction with a warp-shuffle reduction on an **RTX 4060 Ti**, compiled with CUDA 12.1, `-O3 -arch=sm_89`. Both use 32 threads per block and resident FP32 input. CUDA events time 21 alternating rounds of 100 launches per variant after 100 warmup launches each. Both matched the deterministic CPU reference, with zero maximum absolute error for these inputs.

| Blocks | Serial median | Warp median | Serial / warp |
| --- | --- | --- | --- |
| 1 | 1.464 µs | 1.464 µs | 1.0004 |
| 256 | 1.719 µs | 1.719 µs | 1.0002 |
| 32,768 | 32.078 µs | 31.725 µs | 1.0111 |

Jev identified the serial work and distributed reduction as expected. **That structural distinction did not imply a large measured speedup**: the largest case differed by about 1%, while the two smaller cases were effectively tied. These measurements validate keeping source-level performance findings conditional. They do not establish occupancy, SM utilization, profiler counters, or a fastest variant. Results are in `artifacts/reduction-runtime-0.5.json`.

This harness is an offline release check. The extension contains no compiler, GPU executor or automatic profiler hook.

```bash
nvcc -O3 -std=c++17 -arch=sm_89 test/performance/reduction.cu -o /tmp/kernel-lens-reduction-benchmark
/tmp/kernel-lens-reduction-benchmark
```

## 0.4.0 workflow audit, September 20, 2026

This release expands training/inference coverage and adds common PyTorch questions to both profiles and generic tensor helpers. Context collection now retains enclosing decorators, relevant setup declared below a function, and same-class initialization. A change to the bounded enclosing context invalidates the workload route. Tentative concerns remain visible as **Possible:** inline and in the sidebar, without becoming strong diagnostics or automatic-review triggers. No local semantic verdict or generated model call was added to the Jev path.

- **39/39 local regression tests** and strict TypeScript passed. Coverage assertions require both a concern case and a sound control for each training, inference, shared tensor and shared preprocessing question. The [coverage matrix](COVERAGE.md) lists the distinctions exercised.
- The first expanded live run matched **54/56 asserted outcomes across 54 cases**. It missed a reset that wiped gradients before stepping and misclassified valid probability targets for cross entropy. Both question formulations were clarified, and the failed results remain in `artifacts/workflow-audit-first.json`.
- The subsequent full run matched **110/110 assertions across 95 cases**: 108 selected labels plus two controls that reject unexpected strong concerns anywhere in a clean profile. **13 selected outcomes were tentative** below 0.75, including a restarted iterator (0.64), enabled autograd during inference (0.48) and an inference device mismatch (0.74). These remain predictions with uncertainty. Results are in `artifacts/workflow-audit-final.json`.
- Two additional preprocessing cases matched in a separate live run: synchronous tokenization on the GPU submission path was a concern (0.92); worker-based preprocessing was supported but tentative (0.47). One timeout recovered on retry. `artifacts/workflow-preprocessing.json` preserves the results. Combined, the current **97-case corpus matched 112/112 assertions**, with 14 selected outcomes tentative.
- Median classification round trip: **275 ms** (P95 400 ms). Median assessment round trip: **362 ms** (P95 500 ms, max 675 ms). These record successful calls and exclude the default typing pause, local preparation, evaluation pacing and failed attempts. Several 503 errors and two 504 timeouts recovered with retries; these medians are not an end-to-end availability/latency promise.
- The packaged **0.4.0 VSIX passed the mocked-Gateway native editor suite**, the CUDA reduction edit regression, and a new **12-snapshot workflow edit test** in one Python document. The latter switches training → inference → compiled prediction and corrects six issues without pressing Analyze, invoking a generated review or rewriting source. Controlled-response reports are in `artifacts/workflows-mocked-host.json`.
- The **same packaged runtime passed all 12 workflow snapshots against live Jev in native VS Code**. Disconnected loss, optimizer recreation, full-prefix recomputation, discarded padding masks, active functional dropout and compiled Python branching all changed to the expected supported outcomes after edits. Reports are in `artifacts/workflows-live-host.json`; actual screenshots are `artifacts/optimizer_lifetime-live-editor.png` and `artifacts/decode_cache-live-editor.png`. No Analyze command or Astra request was used. The installed runtime is checked byte-for-byte against this tested package.
- Sidebar browser checks passed at 380 px and 260 px, including the PyTorch selector, visible tentative concerns, probability details, review/handoff controls, stale gating and literal rendering of hostile HTML-like strings.
- Local context preparation for a synthetic 25 KB, 250-function Python file measured **4.0 ms median / 7.6 ms P95** over repeated edits; a small CUDA fragment measured 0.5 / 1.6 ms. These machine-specific figures exclude debounce and network requests. No runtime dependency, compiler or local semantic analyzer was added.

The corpus is synthetic and was used to refine the questions; it is regression evidence, not a held-out accuracy benchmark. Most cases assert selected questions, not every returned profile label. The two clean controls cover only their visible programs. CUDA, Triton and previous training/inference regressions are included in the 95-case run. Live Astra access was not changed or retested; the account limitation documented below still applies.

## 0.3.2 reduction regression, September 20, 2026

The reported shuffle-to-serial edit exposed a missing CUDA thread-participation question. The existing SM work-distribution question only evaluated launch-level parallelism. The new question distinguishes a serial cooperative computation in a narrow thread-dependent branch, a distributed reduction, and a single-thread final output store. Jev supplies the verdict; there is no local source-pattern override.

Block selection now retains a completed child on a following blank line or complete trailing C++ statement, even while the outer kernel is unfinished. Long-function context retains the enclosing branch of an inner loop. Inline activity and results appear near the cursor. Transport failures and timeouts use the existing single transient retry, while canceled revisions remain canceled.

- All **34 core tests** and strict TypeScript passed, including completed blocks inside unfinished kernels, context-budget preservation of a late loop's branch, transport redaction and cancellation.
- The packaged extension also passed the existing mocked-Gateway editor suite for training updates, canceled/stale results, caching, permission, auth and rate-limit behavior, automatic transport recovery, reviews, exclusions and pause/resume.
- The packaged **0.3.2 VSIX passed a native VS Code test with live Jev**: open the shuffle reduction, delete it, pause with an incomplete expression, finish the user's serial replacement, restore the shuffle reduction, then finish the inner block inside an unfinished kernel. **No Analyze command or Astra call was made.**
- Live thread-participation outcomes were supported (0.92), concern (1.00), supported (cached), and concern (1.00), respectively. Assessment round trips were 395, 460 and 603 ms, excluding typing pause and routing. These are model probabilities, not measured accuracy or GPU utilization.
- `artifacts/reduction-live-host.json` preserves these reports; `artifacts/reduction-live-editor.png` shows the actual inline warning and sidebar after the serial replacement.
- Four direct live regression cases matched their selected labels above the display threshold: serial reduction, shuffle reduction, the inner serial loop, and a lone final store. Results are in `artifacts/reduction-after.json`. The prior run, without this question, is `artifacts/reduction-before.json`.
- An initial live host run completed the replacement/restoration but failed on a later connection error. The successful rerun above followed the bounded transport-retry change. The direct corpus also encountered one transient 503. The tests do not establish a universal latency or availability guarantee.

Reproduce the native edit regression with `npm run test:host -- --vsix artifacts/kernel-lens-0.3.2.vsix --scenario reduction --live --env-file ~/.env --capture`. Omit `--live --env-file ~/.env` for controlled Gateway responses. Use `npm run eval:live -- --env-file ~/.env --filter '^cuda_reduction_' --output artifacts/reduction-after.json` for the direct four-case check.

## Original 0.3.0 run, September 20, 2026

- All **30 core regression tests** passed, including direct execution of both test files. Strict TypeScript and bundling passed.
- The **packaged VSIX** passed the mocked-Gateway native editor suite, including opt-in automatic reviews and discarding a review that completes after the code changes.
- The **same packaged extension with live Jev** automatically flagged the missing-reset training loop, changed it to supported after `zero_grad()` was added, and passed exclusion/pause/resume checks. This live host test did not request an Astra review. The screenshot is `artifacts/kernel-lens-live-profile.png`; the captured first assessment took 3,147 ms.
- **21/21 selected expected labels matched across 15 live Jev cases.** Twenty matched above the default 0.75 display threshold; the inference-mode result was tentative at 0.57. The full probability profiles are retained in `artifacts/jev-profile-final.json`.
- Median classification round trip: **267 ms**. Median rubric assessment round trip: **323 ms**. One assessment took **6,724 ms**, and two transient server errors needed retries. These timings exclude the editor typing pause and context preparation.
- Local context preparation for a synthetic 25 KB, 250-function Python file: **2.7 ms median / 4.1 ms P95** over 80 edits. This is specific to the test machine, not an end-to-end latency promise.
- Browser checks passed at 260 px and 380 px. The VSIX excludes credentials, test code, preview fixtures and runtime dependency installation; the bundled local diagnosis engine is absent. Packaged size is approximately 132 KB.

## Reproducible checks

- `npm run check`: strict TypeScript, transport/context/assessment/advisor regression tests, production bundle.
- `npm run test:host`: real VS Code with a mocked Gateway. Tests automatic assessments after completed edits, no requests without permission, route/profile caching, corrections, delayed stale responses, incomplete edits, auth errors, rate-limit recovery, review/handoff behavior, excluded paths and pause/resume. No user source or real key is used.
- `npm run test:package`: the same host flow against the actual unpacked VSIX, not a checkout bundle.
- `npm run test:ui`: actual sidebar assets in Playwright. Tests classification rows, probabilities, model review controls, context, navigation, loading, stale gating, literal rendering of hostile HTML-like strings, and 260px/380px layouts.
- `npm run eval:live -- --env-file ~/.env --output artifacts/jev-profile-final.json`: live Jev classifications for the synthetic paired/contextual corpus. Questions, top choices, probabilities, latency, mismatches and expected labels are retained.
- `npm run test:host -- --vsix artifacts/kernel-lens-0.3.0.vsix --live --env-file ~/.env --capture`: live Jev in a native temporary editor, assessing a missing reset then its correction, with an actual sidebar screenshot.

## Model-quality evidence

The small corpus covers missing and correct gradient resets, helper-mediated resets, intentional accumulation, a caller-dependent unknown, CPU-model/CUDA-input mismatch and its correction, restarting a loader iterator, shared tile reuse and cooperative loading, local versus global accumulation, floating-point values versus floating-point indices, Triton max-padding identity, and explicit inference mode.

The first pass misclassified duplicate cooperative loads as supported. A clarified question explicitly distinguishes a serial loop index from a thread coordinate. CUDA paired fixtures now supply their 16×16 launch contract. Subsequent targeted live calls selected the expected outcomes for both the buggy and corrected versions. The missing reset/helper/device/Triton cases were distinguished by the model itself, without local diagnostic overrides.

Native Jev probabilities still vary between runs. Some expected answers remain tentative at the default threshold. The tests validate selected dimensions, not every prediction in every profile, and the corpus is too small and synthetic to establish general code-review accuracy. Rubric revisions based on these fixtures mean they are regression examples, not a held-out benchmark. Runtime performance was not measured.

Gateway returned a transient 503 during an initial run and a 429 during an unpaced run. Live evaluation now paces requests; the extension separately retries rate limits and one transient server failure. Partial evaluation results are saved before a later failure.

## Astra access

The Astra review path is tested through the real editor with a mocked structured response, including exact-source anchoring, stale cancellation, token limits, response validation and clipboard handoff. A real request to `openai/gpt-6-astra` was attempted with the existing Gateway key. Gateway returned HTTP 403 `no_providers_available`; a fixed tiny diagnostic request confirmed: **the free-tier account cannot access this model until it adds paid credits**.

No live Astra completion was obtained on that account. The implementation does not silently switch to a different model. Jev assessments remain available, and the handoff can be pasted into an existing coding-agent session. Public Gateway and OpenAI documentation were checked for the model ID, Chat Completions support, strict JSON output and supported low reasoning effort.

## Limits

The extension performs no GPU profiling, compilation or user-code execution. CUDA resource/latency dimensions are qualitative candidates with unknowns for missing hardware/runtime data. The separate 0.5 offline benchmark above is not a runtime feature. A high model probability is not a measured correctness rate. Source locations are model selections among actual lines, not verified fault tokens; uncertain locations remain block-level. Generated-review links require matching visible source.

Remote SSH, Cursor, notebook cells and public Marketplace distribution have not been independently validated. The package is tested in VS Code 1.98.2 on Linux. Source/key exclusions and redaction are covered by automated tests, but redaction is not a universal secret detector.

See the machine-readable artifacts for the actual final run results and local preparation timings. Prior 0.2.1 source and VSIX artifacts are retained separately for rollback; their local-rule tests do not count toward the 0.3 validation.
