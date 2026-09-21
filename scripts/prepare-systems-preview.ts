import { writeFileSync } from 'node:fs';
import { BlockReader } from '../src/core/blocks';
import { analyzeSystems } from '../src/core/systems/analyze';
import { systemsReport } from '../src/core/systems/evaluate';
import { insightPresentation } from '../src/core/presentation';
import manifest from '../package.json';
const source = `__global__ void reduce(float* out, const float* input) {
    __shared__ float tile[256];
    float value = input[threadIdx.x];
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < 32; i++) sum += tile[i];
        out[0] = sum;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) tile[threadIdx.x] += tile[threadIdx.x + stride];
        __syncthreads();
    }
}`;
const context = new BlockReader().read({ file: 'kernels/reduce.cu', language: 'cpp', source, cursorLine: 0 });
const report = systemsReport(context, analyzeSystems(context));
const state = { version: manifest.version, enabled: true, configured: true, consented: true, phase: 'ready', message: 'Insights updated. Keep writing.', domain: 'auto', intent: '', file: context.file, unit: context.unit, stale: false,
  context: { characters: context.characters, truncated: false, references: [] }, report, ...insightPresentation(report),
  advisor: { mode: 'onDemand', model: 'openai/gpt-6-astra', phase: 'idle', message: 'Ask for a deeper review when you need it.' },
  references: [], requests: 1, advisorRequests: 0, totalTokens: 0, totalCost: 0 };
writeFileSync('artifacts/systems-preview.json', JSON.stringify(state));
console.log('Prepared sidebar preview from the production analysis and finding renderer.');
