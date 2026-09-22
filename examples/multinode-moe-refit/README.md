# MoE weight export: gathering every shard before exporting any

This is a retrospective example from [verl PR #6612](https://github.com/verl-project/verl/pull/6612), authored by `wuxibin89`. Both implementations are upstream code. JEV assessed the original source without receiving the patch or PR explanation.

| Open | What to inspect |
| --- | --- |
| [Before](before.py#L667) | Lines 667–676: allocate a full-group tensor, AllGather, then export. |
| [After](after.py#L666) | Lines 666–672: broadcast and export one rank's shard at a time. |
| [Before helpers](before_utils.py#L111) / [after helpers](after_utils.py#L111) | Expert naming and parameter mapping. |
| [JEV findings](captures/before.report.json) | Source evidence, confidence and next checks. |

## What JEV caught

One recorded assessment of the original code returned in **876 ms**:

| Finding | Assessment | Structural confidence |
| --- | --- | --- |
| AllGather inside the parameter loop | Possible issue | High |
| Gathered intermediate before its consumer | Possible issue | Medium |

The useful investigation is whether export requires every shard to be materialized simultaneously. Both findings request shapes for stronger claims. Runtime impact is unmeasured.

## The upstream change

Before:

```python
output_shape = list(unsharded_tensor.shape)
output_shape[0] *= ps.extra_parallel_sizes["ep"]
stacked_tensor = torch.empty(output_shape, dtype=unsharded_tensor.dtype, device=device)
torch.distributed.all_gather_into_tensor(stacked_tensor, unsharded_tensor, group=ps.ep_group)
yield from process_func(name, stacked_tensor)
del stacked_tensor
```

After:

```python
ep_rank, ep_size = ps.ep_rank, ps.ep_size
buffer = torch.empty_like(unsharded_tensor)
for src_ep_rank in range(ep_size):
    tensor = unsharded_tensor if src_ep_rank == ep_rank else buffer
    torch.distributed.broadcast(tensor, group_src=src_ep_rank, group=ps.ep_group)
    yield from process_func(name, tensor, ep_rank=src_ep_rank)
```

The updated helper uses the source rank to preserve global expert indices. Consumers must finish using or copying yielded buffer views before reuse.

The supplied deployment scenario was **two nodes × eight H100 SXM GPUs, EP=16, BF16 parameters**. Shapes and interconnect bandwidth were unspecified; this was not an upstream benchmark configuration.

For a local shard of `L` bytes, the explicit extra buffer changes from **`EP × L` to `L`**. At EP=16, that buffer is 16× smaller. This is a source-derived allocation comparison, not a JEV numerical prediction or a measured total-memory reduction. Communication remains, and sequential broadcasts have different overhead. No training or performance benchmark was run.

## Try it

1. Open `before.py` at `get_per_tensor_param` (line 646).
2. Select **Auto** and **H100** in Kernel Lens. For the recorded setup, copy the deployment text from [provenance.json](captures/provenance.json) into `kernelLens.hardware`.
3. Select the helper and mapping from line 111 onward in `before_utils.py` and run **Kernel Lens: Attach Selection as Context**. Return to `before.py` and choose **Analyze**. Leave task intent empty; Astra is not needed for classification.
4. Inspect the findings around lines 670–676, then compare with `after.py`, lines 668–672.

```sh
code --diff examples/multinode-moe-refit/before.py examples/multinode-moe-refit/after.py
```

With repository dependencies installed, verify the frozen requests and replay the captured original response offline:

```sh
node --import tsx examples/multinode-moe-refit/replay.cjs
```

The successful capture covers **before.py**; **after.py** is the upstream reference implementation. [Capture details](captures/README.md) include exact requests, revisions and capture limitations. A fresh editor request can differ with available context.

These are framework source snapshots, not standalone training scripts. Original copyright notices and the [Apache 2.0 license](LICENSE-APACHE-2.0.txt) are preserved.
