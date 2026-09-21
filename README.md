# Kernel Lens

Live performance insights for GPU kernels in VS Code, powered by JEV. Pause while coding to see what looks good, what deserves attention, and what to inspect next.

## How it works

Kernel Lens sends your function, relevant helpers, and selected GPU context to [JEV](https://vercel.com/i/what-is-jev) through Vercel AI Gateway. Kernel Lens defines the performance questions and allowed answers; JEV returns classifications with probabilities.

The extension pairs those answers with source evidence and a predefined next check, shown beside your code and in the sidebar.

## What JEV classifies

The kernel lens has **24 classification dimensions** informed by Nsight Compute metric families. Examples:

| Performance target | Example classification buckets | What the insight tells you |
| --- | --- | --- |
| Compute / tensor pipeline | Scalar contraction, warp MMA, WGMMA, Blackwell `tcgen05` | Which compute path the kernel expresses. |
| Lane participation / reductions | Serial tail, cooperative tree, warp shuffle, elected-thread MMA issue | How work is distributed across lanes. |
| Global memory / reuse | Contiguous or strided access, shared tiles, repeated global reads | Where access patterns or operand reuse deserve inspection. |
| Shared memory | Swizzled layout, padding, broadcast, extra staging | Which layout and staging decisions to examine. |
| Copy / compute overlap | Overlap expressed, immediate wait, single or multiple buffers | Whether the source exposes useful work between issue and wait. |
| Synchronization / MMA cadence | Per-tile barrier, per-tile MMA wait, multiple in-flight groups | Where synchronization constrains progress. |
| Registers / shared-memory footprint | Per-thread array pressure, multiple shared buffers | Which resource tradeoffs need compiler evidence. |
| Accumulators / epilogue | Register or TMEM accumulation, repeated global updates, TMEM read waits | Where accumulation and output handling may add work. |

Each insight includes **evidence, confidence, and one next check**. Missing evidence stays unknown. These are static classifications; runtime impact is unmeasured unless measurements are supplied.

<details>
<summary>Example JSON request sent to JEV</summary>

Sent to `POST https://ai-gateway.vercel.sh/v1/evaluate`. This excerpt shows one question; additional source metadata, helper/operation context, hardware details, and other questions are omitted.

```json
{
  "model": "typesafe-ai/jev",
  "state": {
    "current": {
      "file": "sum32.cu",
      "code": "L1: __global__ void sum32(float* x) {\nL2:   if (threadIdx.x == 0) {\nL3:     float sum = 0;\nL4:     for (int i = 0; i < 32; ++i) sum += x[i];\nL5:     x[0] = sum;\nL6:   }\nL7: }"
    },
    "targetHardware": "Assume B200."
  },
  "questions": {
    "review.cuda-reduction": {
      "type": "choice",
      "instructions": "Classify Reduction organization independently. Classify reduction organization, including warp shuffle, shared tree and narrow serial tail. Matrix compute is not automatically a scalar reduction loop. Use the full function and resolved helper instructions. Select unknown for missing relationships/configuration; not_applicable for absent work. Relevant lines: 4.",
      "criteria": {
        "unknown": "The distinguishing source relationship or required configuration is unavailable.",
        "not_applicable": "This execution path does not contain work relevant to this axis.",
        "tree": "A halving/shared or recognized block reduction distributes the reduction across threads.",
        "serial": "A single lane or narrow branch sums multiple elements serially."
      }
    }
  }
}
```

`state` supplies the evidence. `questions` defines what to classify, and `criteria` defines the allowed answers. JEV returns a selected `choice` and `probabilities` for each question; Kernel Lens maps the result to an insight and next check.

</details>

[All classification criteria](src/core/systems/kernel-review.ts) · [Framework coverage](docs/SYSTEMS-COVERAGE.md)

## Build and install locally

Requires **Node.js 22+** and **VS Code 1.98+**. Open a terminal in this repository:

```sh
npm ci
npm run package
code --install-extension artifacts/kernel-lens-0.7.0.vsix --force
```

`npm run package` builds the extension and creates the VSIX. You can also install that file through **Extensions → ⋯ → Install from VSIX**. After upgrading, run **Developer: Reload Window**.

## Start using it

1. Run **Kernel Lens: Set AI Gateway API Key** from the Command Palette and enter your key.
2. Run **Kernel Lens: Enable Live Insights for This Workspace** in a trusted workspace.
3. Open a CUDA or Triton file and keep coding. Insights update after a short typing pause.

The assumed GPU defaults to **B200**. Change it in the sidebar to H100, A100, or Unspecified/custom.

Selected source context is sent to AI Gateway after you enable the workspace. Your key is stored in VS Code secret storage; unchanged context is cached.

## Development

`npm run build` builds without packaging. Open this repository in VS Code and press **F5** to launch the extension in a development window. Run `npm run check` for type checking, tests, and a build.

Share the VSIX with others; each user supplies their own Gateway key. [Architecture](docs/SYSTEMS-ARCHITECTURE.md) · [Release notes](docs/RELEASE-0.7.0.md)
