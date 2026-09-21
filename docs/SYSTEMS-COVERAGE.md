# Initial systems coverage · 0.6.1

Direct observations use semantic rules. Fixed Jev dimension questions also review code without a local issue match, using source anchors and the selected hardware. Coverage is bounded: listing a dimension does not imply complete framework understanding.

| Lens | Implemented observations and investigations |
|---|---|
| CUDA | Narrow-branch serial reductions; shuffle reduction positives; block barriers in halving reductions; repeated global addresses; loop-invariant global RMW accumulation; large local arrays; adjacent thread-x addresses; guarded-access and power-of-two execution assumptions. |
| Triton | Recognized dot/reduction primitives, load/store operations and JIT boundary mapping. |
| PyTorch / JAX | Known host reads/waits in loops with device uncertainty; new compiled callables in loops; decorated compilation boundaries; tensor shape/value control; scoped no-host-read positives; explicit materialization; growing concatenation; recognized producer-consumer chains; fixed-shape CUDA graph capture and host reads during capture. |
| Input pipeline | Explicit zero-worker DataLoader policy, persistent workers and repeated next(iter(loader)) on a known DataLoader. These are source policies, not claims of an input bottleneck. |
| Distributed | AllReduce, AllGather, ReduceScatter, AllToAll, broadcast, send/recv, JAX sharding constraints, direct gathered-value materialization, A→B→A constraints and possible communication/compute overlap. |
| vLLM / SGLang | Recognized Python engine/configuration constructors; explicit TP and concurrency limits; chunked-prefill policy; conditional long-prefill/decode contention; graph policy; prefix-cache policy; explicit PD mode. |

Tensor shapes and dtype must be explicit or supplied before computing bytes. Default-group rank count must be explicit before estimating equal-sized per-rank gathered/scattered buffers or theoretical ring AllReduce traffic. Custom groups remain unknown. These estimates are not communication timings.

Conventional logical KV sizing requires all dimensions passed through a vLLM `hf_overrides` literal (`num_hidden_layers`, `num_key_value_heads`, `head_dim`), an explicit byte-width-resolvable cache/model dtype and explicit sequence/concurrency limits. Unused similarly named globals do not establish model dimensions. The formula assumes conventional dense K/V storage in every layer; latent attention, sliding-window layouts, allocator overhead, cache replication and physical per-rank placement require additional analysis.

Serving support currently covers explicit Python calls, not arbitrary YAML, CLI strings or scheduler implementation recovery. Missing flags are not interpreted as framework defaults. TP, DP, EP and PP keyword values are preserved in IR, but their presence alone does not justify a performance issue. Complete CUDA indexing verification, tensor correctness, optimizer correctness, automatic kernel counts and full sharding/dataflow reconstruction are not implemented. FLOP/roofline estimates are limited to known compatible two-dimensional GPU FP32 matmul/mm/dot inputs. Other operations retain qualitative classifications and requirements.

## Semantic references

- [Jev evaluation interface](https://vercel.com/kb/guide/typesafe-jev-and-ai-sdk): bounded decisions and distributions; application-owned response policy.
- [PyTorch distributed APIs](https://docs.pytorch.org/docs/stable/distributed): collective signatures, output/input relationships and asynchronous operation policy.
- [JAX device_get](https://docs.jax.dev/en/latest/_autosummary/jax.device_get.html): host-transfer API.
- [JAX sharding constraints](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html): compiler constraints, not unconditional proof of a physical transfer.
- [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/): scheduling, chunked prefill and parallelism tradeoffs.
- [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments): explicit engine scheduling, graph, cache and disaggregation configuration.

Unit controls cover comments/strings, unrelated methods, shadowed and monkey-patched APIs, reassignment, distinct dataflow, unknown calls, incomplete context and literal dead branches. These controls test specific failure modes; they do not establish universal false-positive freedom.

## Broader Jev review

CUDA dimensions include explicit operand reuse, scalar versus matrix instructions, thread participation, coalescing, resource pressure, latency hiding, shared staging and normalization arithmetic. PyTorch dimensions cover training iterator use, host reads, retained graphs, DDP accumulation, inference transfers/caches/setup, batching and materialization. JAX adds fixed graph/compilation dimensions; distributed and serving code adds collective-placement, layout and execution-policy dimensions. These findings use medium-confidence inference, selected source evidence and an unmeasured runtime label. Missing context remains unknown; no numeric GPU utilization is invented.
