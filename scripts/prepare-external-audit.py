"""Freeze expectations before calling Jev. Downloads are evaluation inputs only.

Use --local-dir /tmp to reuse kernel-lens-{mnist,generate,softmax}.py files.
The full pinned sources and controlled mutations go into artifacts, not the VSIX.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument('--local-dir', type=Path)
parser.add_argument('--output', type=Path, default=Path('artifacts/external-cases-0.5.json'))
args = parser.parse_args()
urls = {
    'mnist': 'https://raw.githubusercontent.com/pytorch/examples/acc295dc7b90714f1bf47f06004fc19a7fe235c4/mnist/main.py',
    'generate': 'https://raw.githubusercontent.com/pytorch/examples/acc295dc7b90714f1bf47f06004fc19a7fe235c4/word_language_model/generate.py',
    'softmax': 'https://raw.githubusercontent.com/triton-lang/triton/84fa223cb8fe44f671a0fcd543236f44df975dda/python/tutorials/02-fused-softmax.py',
}
sources = {name: (args.local_dir / f'kernel-lens-{name}.py').read_text() if args.local_dir else urlopen(url, timeout=30).read().decode() for name, url in urls.items()}
cases = []

def add(name, source, focus, pack, expected, origin, mutation='none', file='example.py'):
    cases.append(dict(name=name, file=file, source=source, cursorLine=next(i for i, l in enumerate(source.splitlines()) if focus in l),
                      pack=pack, expect={k: [v] for k, v in expected.items()}, origin=origin, mutation=mutation))

mnist = sources['mnist']
add('external_mnist_train_original', mnist, 'def train(', 'training', {'train-grad-reset': 'supported', 'train-graph': 'supported', 'train-step-order': 'supported', 'train-loss': 'supported'}, urls['mnist'])
add('external_mnist_reset_removed', mnist.replace('        optimizer.zero_grad()\n', ''), 'def train(', 'training', {'train-grad-reset': 'concern'}, urls['mnist'], 'Remove zero_grad from the training loop.')
add('external_mnist_update_before_backward', mnist.replace('        loss.backward()\n        optimizer.step()', '        optimizer.step()\n        loss.backward()'), 'def train(', 'training', {'train-step-order': 'concern'}, urls['mnist'], 'Swap step and backward.')
generate = sources['generate']
add('external_language_generate_original', generate, 'for i in range(args.words):', 'inference', {'infer-grad': 'supported', 'infer-mode': 'supported'}, urls['generate'])
add('external_language_generate_grad_enabled', generate.replace('with torch.no_grad():', 'with torch.enable_grad():'), 'for i in range(args.words):', 'inference', {'infer-grad': 'concern'}, urls['generate'], 'Enable autograd during generation.')
softmax = sources['softmax']
add('external_triton_softmax_original', softmax, 'def softmax_kernel(', 'triton', {'triton-bounds': 'supported', 'triton-reduction': 'supported'}, urls['softmax'])
add('external_triton_softmax_zero_padding', softmax.replace("other=-float('inf')", 'other=0.0'), 'def softmax_kernel(', 'triton', {'triton-reduction': 'concern'}, urls['softmax'], 'Use zero for padded reduction lanes.')
add('external_triton_softmax_unmasked_load', softmax.replace("tl.load(input_ptrs, mask=mask, other=-float('inf'))", 'tl.load(input_ptrs)'), 'def softmax_kernel(', 'triton', {'triton-bounds': 'concern'}, urls['softmax'], 'Remove the load mask; the original example uses irregular widths.')
benchmark = Path('test/performance/reduction.cu').read_text()
for variant, outcome in [('serial_reduce', 'concern'), ('warp_reduce', 'supported')]:
    add(f'offline_benchmark_{variant}', benchmark, f'void {variant}(', 'cuda', {'cuda-thread-work': outcome}, 'test/performance/reduction.cu', file='reduction.cu')
result = {
    'note': 'Frozen expected labels on three pinned public examples, five controlled mutations, and two variants of one offline CUDA benchmark. These examples were not used to tune rubric 5.0.0. Selected labels only; not a general accuracy measurement. Full predictions and misses are retained.',
    'rubricSha256': hashlib.sha256(Path('src/core/rubrics.ts').read_bytes()).hexdigest(),
    'sources': {k: dict(url=urls[k], sha256=hashlib.sha256(v.encode()).hexdigest()) for k, v in sources.items()},
    'licenses': ['https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/LICENSE', 'https://github.com/triton-lang/triton/blob/84fa223cb8fe44f671a0fcd543236f44df975dda/LICENSE'],
    'cases': cases,
}
args.output.parent.mkdir(exist_ok=True, parents=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(f'Froze {len(cases)} cases with {sum(len(c["expect"]) for c in cases)} expected labels in {args.output}')
