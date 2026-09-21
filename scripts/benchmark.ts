import { readFileSync, writeFileSync } from 'node:fs';
import { BlockReader } from '../src/core/blocks';
import { systemsPayload } from '../src/core/systems/evaluate';
import { analyzeSystems } from '../src/core/systems/analyze';

const cuda = readFileSync('test/fixtures/shared-matmul.cu', 'utf8');
const python = 'import torch\ndef run():\n    x = torch.randn((1,),device="cuda")\n    for step in range(32):\n        y = x.item()\n';
const large = 'import torch\n' + Array.from({ length: 250 }, (_, i) => `def compute_${i}(x):\n    a = torch.randn(32, 64)\n    b = torch.randn(64, 16)\n    c = torch.matmul(a,b)\n    return c\n`).join('\n');
const results = [];
for (const [name, source, language] of [['cuda-fragment', cuda, 'cuda-cpp'], ['tensor-dataflow', python, 'python'], ['250-functions', large, 'python']]) {
  const engine = new BlockReader();
  const start = performance.now();
  analyzeSystems(engine.read({file: name!, language: language!, source: source!, assessmentScope: 'function', cursorLine: 3}));
  const first = performance.now() - start;
  const timings = [];
  for (let i = 0; i < 80; i++) {
    const start = performance.now();
    const context=engine.read({file: name!, language: language!, source: source + (language === 'python' ? `\n# edit ${i}` : `\n// edit ${i}`), assessmentScope: 'function', cursorLine: 3});
    systemsPayload(context,analyzeSystems(context));
    timings.push(Math.round((performance.now() - start) * 10) / 10);
  }
  timings.sort((a, b) => a - b);
  results.push({ case: name, characters: source!.length, initialMs: first, medianEditMs: timings[40], p95EditMs: timings[76], maxEditMs: timings.at(-1) });
}
const report = { environment: `${process.platform} / Node ${process.version}`, timestamp: new Date().toISOString(), note: 'Local block/context preparation plus Systems IR and all static lenses and model payload preparation, repeated without report-cache reuse. Excludes default 700 ms typing pause, UI rendering and model/network work. Synthetic fixtures, not a cross-device guarantee or user-code performance measurement.', results };
writeFileSync('artifacts/systems-local-latency-0.6.1.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
