> Historical record through 0.5. See [current coverage](SYSTEMS-COVERAGE.md) for the 0.6 systems refactor.

# Workflow coverage

Kernel Lens sends the completed block and bounded context to Jev. Jev chooses the workload and classifies every question in that profile. A source parser supplies boundaries and context; it does not supply semantic verdicts. The labels below are hypotheses from source, not measured throughput or proof of correctness.

`test/fixtures/workflows.ts` adds paired problematic/corrected examples and valid exceptions. Along with `test/fixtures/predictions.ts`, the corpus contains a concern and a sound control for every training, inference and common tensor question. Tests assert this coverage so adding a question without a corresponding pair fails locally. The live evaluator saves full model responses and verifies selected expected labels; two clean-code controls additionally reject any unexpected strong concern anywhere in their profile.

## Training

| Question | Problem case | Sound control or context distinction |
| --- | --- | --- |
| Gradient reset | No reset, or clearing after backward and before step | Reset at update boundaries; helper-mediated reset; intentional accumulation |
| Gradient connectivity | Detaching the optimized loss before backward | Connected student loss with a separate detached teacher/logging branch |
| Optimizer lifetime | Recreating AdamW inside the update loop | One optimizer reused across updates |
| Loss contract | Softmax predictions passed to cross entropy | Raw logits; probability targets in the target argument are valid |
| Update order | Step before backward | Backward followed by step |
| Scheduler order | Scheduler advances before the optimizer | Scheduler follows updates at the stated cadence |
| Accumulation boundaries | Every microbatch clears an intended four-step window | Reset and step only at update boundaries |
| Accumulation scaling | Summing when the explicit contract requires a mean | Loss scaled for equal-sized microbatches |
| Device placement | CPU parameters with CUDA input | Matching visible placement; caller-dependent placement stays unknown |
| Training behavior | No-grad forward in a trainable update | Training forward retains gradients |
| Validation transition | Validation leaves later updates in eval mode | Isolated eval/no-grad followed by restoration of train mode |
| AMP and clipping | Clip still-scaled gradients | Unscale before clipping, then scaler step/update |
| Iterator lifecycle | Repeated `next(iter(loader))` | A persistent iterator or normal loader loop |
| Host synchronization | Reading a loss scalar each update | Device-side aggregation and deferred readback |
| Graph retention | Appending graph-connected losses | Detached logging values; this control only addresses graph retention, not arbitrary unbounded GPU storage |
| Distributed participation | Only one rank executes synchronized backward | Matching backward participation |
| Accumulation communication | Every DDP microbatch synchronizes | Intermediate forward and backward both under `no_sync`, final backward synchronized |

## Inference

| Question | Problem case | Sound control or context distinction |
| --- | --- | --- |
| Autograd overhead | Prediction with gradient recording enabled | Enclosing `inference_mode`/`no_grad` preserved even when focusing an inner loop |
| Evaluation behavior | Dropout model explicitly left in train mode | Visible eval setup |
| Device placement | CPU model with CUDA input | Matching setup, including setup below the function |
| Host transfers | GPU → CPU/NumPy → GPU in a repeated path | Intermediates remain on device |
| Storage growth | Repeated concatenation of growing outputs | Indexed writes into preallocated output |
| Decode reuse | Full growing prefix recomputed for each token | Prefill followed by new-token input and reused key/value states |
| Attention implementation | Explicit scores, softmax and value product | SDPA expressed; actual fused backend remains runtime dependent |
| Causal contract | Explicit causal prefill without a causal mask | Causal prefill mask; bidirectional attention is a different contract |
| Padded-batch mask | Tokenizer pads but the model receives only IDs | Mask forwarded, including through `**batch` |
| Functional dropout | Positive SDPA dropout during inference | Explicit zero dropout; `eval()` alone does not control SDPA's argument |
| Serving setup | Model construction/loading on each request | Startup or class-owned model reused by the handler |
| Serving memory | Persistent unbounded output history | Bounded deque; detaching alone still retains tensor storage |

## Common PyTorch and input pipelines

| Question | Problem case | Sound control or context distinction |
| --- | --- | --- |
| Host-to-device submission | Blocking CUDA copies per batch | Pinned source and nonblocking submission; overlap and speedup are unmeasured |
| Tensor batching | Independent row operations called separately in Python | Batched operation; sequential token dependencies are legitimate |
| Allocator reuse | `empty_cache()` in every hot iteration | No per-iteration flush; one release outside the loop is distinct |
| Compiled graph continuity | Tensor-dependent Python branch under `torch.compile` | Tensor control or `torch.where`; ordinary eager code is not flagged as compiled |
| Normalization stability | Unbounded exponential normalization without stabilization | Stable primitive or max subtraction |
| Layout compatibility | Transpose then incompatible flattening `view` | Explicit layout handling with `reshape` or `contiguous` |
| CPU preprocessing | Serial tokenization on the GPU submission path | Preprocessing supplied by loader workers/prefetch |

Common questions are included in training, inference, input-pipeline and generic PyTorch profiles. Questions with no applicable operation can be classified as not applicable. These are one batched assessment request, not one network request per question.

## Context and editor checks

Local tests cover decorator preservation, setup initialized below a function, same-class initialization, decorated callers/helpers, adjacent function boundaries, context budgets, and invalidating a workload route when its enclosing code changes. Used workspace imports are tested without a language server, including unused imports ahead of the relevant definition. Stable scope identifiers survive blank-line shifts, and the bounded previous edit is labeled historical rather than current evidence. Additional live cases rename common aliases and include frozen teachers and probability targets so a valid exception is not treated as a problem.

The native workflow test edits the same Python file through six problematic/corrected pairs: disconnected loss, recreated optimizer, decode cache, padding mask, functional dropout and compiled control. With the sidebar closed it checks CodeLens, hover content, model-selected source locations, stale annotation removal and specific correction feedback. No Analyze command, generated review or source rewrite is used. The CUDA edit test separately covers shuffle → serial → shuffle and completion of an inner block inside an unfinished kernel. A delayed-location test proves that stale anchors cannot resurrect an old finding and that an interrupted selection can resume from a cached primary report.

`scripts/prepare-external-audit.py` prepares a separate reproducible audit from pinned official PyTorch MNIST, language-generation and Triton softmax examples, plus controlled mutations and an offline CUDA reduction benchmark. It freezes source hashes and expected labels before evaluation. The 0.5 run includes a missed Triton load mask; its full result is preserved. Neither its questions nor expectations were revised after that run. See [validation](VALIDATION.md) for the measured limits.

Run the full live corpus with `npm run eval:live -- --env-file ~/.env --output artifacts/workflow-audit-final.json`. See [validation](VALIDATION.md) for recorded results and limitations. This is a synthetic regression corpus used to refine the questions; it is not an independent accuracy benchmark. Jev can still miss problems, and not every outcome in every profile has an asserted expected label.

The questions draw on the [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html), [functional SDPA contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html), and [Transformers cache documentation](https://huggingface.co/docs/transformers/main/en/cache_explanation). Each question also carries a relevant reference link in the extension.
