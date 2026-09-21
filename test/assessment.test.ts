import test from 'node:test';
import assert from 'node:assert/strict';
import { BlockReader, contextHash } from '../src/core/blocks';
import { assessmentPayload, routingPayload, classify, assess, reportFromAnswers, changesSince, routeAnswers } from '../src/core/assessment';
import { parseAnswers, GatewayError } from '../src/core/gateway';
import { metrics, metricsFor } from '../src/core/rubrics';
import { inlineInsight } from '../src/core/presentation';
import { parseReview, review, reviewPayload, handoff } from '../src/core/advisor';
import { predictionCases } from './fixtures/predictions';
import { reductionKernel, serialReduction } from './fixtures/reduction';
import type { CodeContext, Route } from '../src/live-types';

const reader = new BlockReader();
function context(name = 'training_missing_reset') {
  const item = predictionCases.find(c => c.name === name)!;
  return reader.read({ ...item, language: item.file.endsWith('.py') ? 'python' : 'cpp' });
}
const route: Route = { technology: 'pytorch', activity: 'training', packs: ['training'], uncertain: false, probabilities: { technology: .99, activity: .99 } };
function answers(payload: { questions: Record<string, { criteria: Record<string, string> }> }, selected: Record<string, string> = {}, p = .96) {
  return { answers: Object.fromEntries(Object.entries(payload.questions).map(([id, question]) => {
    const options = Object.keys(question.criteria), choice = selected[id] ?? (options.includes('unknown') ? 'unknown' : options[0]!);
    return [id, { type: 'choice', choice, probabilities: Object.fromEntries(options.map(k => [k, k === choice ? p : (1 - p) / (options.length - 1)])) }];
  })) };
}
function report(ctx = context(), choice = 'concern', p = .96) {
  const payload = assessmentPayload(ctx, route);
  return reportFromAnswers(ctx, route, parseAnswers(answers(payload, { 'train-grad-reset': choice }, p), payload), { latencyMs: 280 });
}
test('completed training loop includes model/optimizer setup and enclosing reset boundaries', () => {
  const c = context('training_intentional_accumulation');
  assert.equal(c.unit.ready, true); assert.equal(c.unit.kind, 'ForStatement');
  assert.match(c.enclosing, /zero_grad/);
  assert.match(context().preamble, /Linear.*cuda/);
  assert.match(context('training_cpu_model_cuda_input').preamble, /Linear\(8, 2\)$/m);
});
test('called helpers and known callers are supplied, including reset in another function', () => {
  assert.ok(context('training_reset_in_helper').references.some(r => r.name === 'clear' && r.code.includes('zero_grad')));
  const c = reader.read({ file: 'x.py', language: 'python', source: 'def step(model, x):\n    return model(x)\ndef caller(model, x):\n    model.to("cuda")\n    return step(model, x)\n', cursorLine: 0 });
  assert.ok(c.references.some(r => r.name === 'caller' && r.code.includes('model.to')));
});
test('decorators and CUDA attributes survive context extraction', () => {
  assert.match(context('inference_explicit_mode').code, /@torch.inference_mode/);
  assert.match(context('triton_max_wrong_padding').code, /@triton.jit/);
  assert.match(context('cuda_float_index_invalid').code, /__global__/);
  assert.match(context('cuda_tile_reuse_race').preamble, /__shared__/);
});
test('unfinished blocks wait; completed blocks remain assessable with an unfinished sibling', () => {
  for (const source of ['def f(x):\n', 'def f(x)\n    return x', 'def f(x):\n    y = (']) {
    assert.equal(reader.read({ file: 'x.py', language: 'python', source, cursorLine: 0 }).unit.ready, false, source);
  }
  assert.equal(reader.read({ file: 'x.cu', language: 'cpp', source: '__global__ void f() {\n int i = 1;', cursorLine: 0 }).unit.ready, false);
  const c = reader.read({ file: 'x.py', language: 'python', source: 'def first():\n    return 1\n\ndef second():\n    x = (', cursorLine: 0 });
  assert.equal(c.unit.ready, true); assert.equal(c.unit.name, 'first');
});
test('moving immediately after a function keeps the completed function', () => {
  const c = reader.read({ file: 'x.py', language: 'python', source: 'def f(x):\n    return x\n\n', cursorLine: 3 });
  assert.equal(c.unit.name, 'f'); assert.equal(c.unit.ready, true);
});
test('finished CUDA blocks are assessed after a newline or trailing barrier inside an unfinished kernel', () => {
  for (const suffix of ['\n\n', '\n    __syncthreads();\n\n']) {
    const body = serialReduction.slice(0, serialReduction.lastIndexOf('\n'));
    const source = `__global__ void reduce(float* arr_cpy) {\n${body}${suffix}`;
    const c = reader.read({ file: 'editing.cu', language: 'cpp', source, cursorLine: source.split('\n').length - 1 });
    assert.equal(c.unit.kind, 'IfStatement'); assert.equal(c.unit.ready, true);
    assert.match(c.code, /threadIdx.x == 0/); assert.match(c.code, /sum \+= arr_cpy\[i\]/);
    if (suffix.includes('__syncthreads')) assert.match(c.code, /__syncthreads/);
  }
  const source = reductionKernel(serialReduction).replace('    __syncthreads();\n    if (threadIdx.x == 0) output', '    if (threadIdx.x > 0) {\n        float x = (\n    if (threadIdx.x == 0) output');
  const c = reader.read({ file: 'editing.cu', language: 'cpp', source, cursorLine: 12 });
  assert.equal(c.unit.ready, false, 'an unfinished new block never silently assesses the earlier one');
});
test('CUDA thread participation is a model question alongside launch-level parallelism', () => {
  const c = reader.read({ file: 'reduce.cu', language: 'cpp', source: reductionKernel(serialReduction), cursorLine: 7 });
  assert.match(c.enclosing, /threadIdx.x == 0/);
  const p = assessmentPayload(c, { ...route, technology: 'cuda', activity: 'kernel', packs: ['cuda'] });
  assert.ok(p.questions['cuda-thread-work']); assert.ok(p.questions['cuda-parallelism']);
  assert.match(p.questions['cuda-thread-work']!.criteria.supported!, /final result/);
  assert.equal(Object.hasOwn(p.state, 'sourceFacts'), false);
});
test('long CUDA functions retain the branch enclosing a late inner loop within the context budget', () => {
  const source = reductionKernel(serialReduction, '    arr_cpy[threadIdx.x % 32] += 1.0f;\n'.repeat(160));
  const c = reader.read({ file: 'long.cu', language: 'cpp', source,
    cursorLine: source.split('\n').findIndex(line => line.includes('sum += arr_cpy')) });
  assert.equal(c.unit.kind, 'ForStatement'); assert.equal(c.unit.ready, true);
  assert.equal(c.truncated, true); assert.ok(c.characters <= 16000);
  assert.match(c.enclosing, /threadIdx.x == 0/);
  assert.match(c.enclosing, /__shared__ float arr_cpy/);
});
test('inner Python loops inherit decorators and relevant globals initialized after the function', () => {
  const c = context('audit_setup_below_function');
  assert.equal(c.unit.kind, 'ForStatement');
  assert.match(c.enclosing, /@torch.inference_mode/);
  assert.match(c.preamble, /Linear\(8, 2\).*cuda\(\).*eval\(\)/);
});
test('same-class initialization and decorated helpers survive automatic context gathering', () => {
  const c = context('audit_serving_retention_supported');
  assert.ok(c.references.some(r => r.reason === 'Class initialization' && r.code.includes('deque(maxlen=32)')));
  const source = 'import torch\n@torch.no_grad()\ndef frozen(model, x):\n    return model(x)\ndef predict(model, x):\n    return frozen(model, x)\n';
  const helper = reader.read({ file: 'helpers.py', language: 'python', source, cursorLine: 4 });
  assert.equal(helper.unit.name, 'predict', 'an adjacent definition does not select the previous function at its boundary');
  assert.ok(helper.references.some(r => r.name === 'frozen' && r.code.includes('@torch.no_grad')));
});
test('routing cache changes when a focused block keeps its text but its enclosing workflow changes', () => {
  const prefix = 'import torch\ndef run(model, batches):\n    for x in batches:\n        if x.numel():\n            result = model(x)\n';
  const a = reader.read({ file: 'role.py', language: 'python', source: prefix + '        result.sum().backward()\n        optimizer.step()\n', cursorLine: 4 });
  const b = reader.read({ file: 'role.py', language: 'python', source: prefix + '        yield result\n', cursorLine: 4 });
  assert.equal(a.code, b.code); assert.notEqual(a.routingKey, b.routingKey);
  const decorated = reader.read({ file: 'role.py', language: 'python', source: 'import torch\n@torch.inference_mode()\ndef run(model, batches):\n    for x in batches:\n        if x.numel():\n            result = model(x)\n        yield result\n', cursorLine: 5 });
  assert.notEqual(b.routingKey, decorated.routingKey);
});
test('generic PyTorch helpers receive tensor checks even without a training/inference role', () => {
  const p = routingPayload(context('audit_normalization_supported'));
  const r = routeAnswers(parseAnswers(answers(p, { technology: 'pytorch', activity: 'general' }), p));
  assert.deepEqual(r.packs, ['pytorch']);
  const ids = metricsFor(r).map(m => m.id);
  for (const id of ['tensor-numerics', 'tensor-batching', 'tensor-compile', 'tensor-layout']) assert.ok(ids.includes(id));
  const manual = routeAnswers(parseAnswers(answers(p, { technology: 'pytorch', activity: 'general' }), p), 'inference');
  assert.deepEqual(manual.packs, ['inference', 'pytorch']);
});
test('context budgets and key redaction apply to every supplied context field', () => {
  const secret = 'test-exact-secret-that-must-not-escape';
  const c = reader.read({ file: 'x.py', language: 'python', source: `def f():\n    x = "${secret}"\n` + '    x += 1\n'.repeat(600), cursorLine: 300,
    maxCharacters: 4000, secret, intent: secret.repeat(1000), hardware: secret, references: [{ name: secret, startLine: 1, code: secret.repeat(200), reason: secret }] });
  assert.ok(c.characters <= 4000); assert.ok(c.truncated); assert.ok(!JSON.stringify(c).includes(secret));
  assert.match(c.code, /L301:/);
});
test('cache keys reflect helper/hardware/intent changes and ignore character bookkeeping', () => {
  const a = context(); const b = structuredClone(a); b.characters++;
  assert.equal(contextHash(a), contextHash(b));
  b.hardware = 'Target H100, block 256'; assert.notEqual(contextHash(a), contextHash(b));
  b.hardware = a.hardware; b.references.push({ name: 'reset', startLine: 1, code: 'def reset(): pass', reason: 'helper' });
  assert.notEqual(contextHash(a), contextHash(b));
});
test('Jev routing selects workload packs without local source-pattern diagnoses', () => {
  const p = routingPayload(context());
  const result = routeAnswers(parseAnswers(answers(p, { technology: 'pytorch', activity: 'training' }), p));
  assert.deepEqual(result.packs, ['training', 'pytorch']);
  const mixed = routeAnswers(parseAnswers(answers(p, { technology: 'triton', activity: 'training' }), p));
  assert.deepEqual(mixed.packs, ['triton', 'training', 'pytorch']);
  const uncertain = routeAnswers(parseAnswers(answers(p), p));
  assert.deepEqual(uncertain.packs, ['general']); assert.equal(uncertain.uncertain, true);
  assert.deepEqual(routeAnswers(parseAnswers(answers(p), p), 'cuda').packs, ['cuda']);
});
test('all rubric dimensions are evaluated even when their operation is missing', () => {
  const c = reader.read({ file: 'x.py', language: 'python', source: 'def train():\n    pass', cursorLine: 0 });
  const payload = assessmentPayload(c, route);
  assert.equal(Object.keys(payload.questions).length, metricsFor(route).length);
  assert.ok(payload.questions['train-grad-reset']); assert.ok(payload.questions['train-device']);
  assert.equal(Object.hasOwn(payload.state, 'sourceFacts'), false);
  assert.equal(new Set(metrics.map(m => m.id)).size, metrics.length);
  assert.ok(metrics.every(m => m.outcomes.unknown && m.outcomes.not_applicable && !Object.hasOwn(m, 'trigger')));
});
test('every training, inference and common tensor question has a concern case and a sound control in the live corpus', () => {
  for (const m of metrics.filter(m => ['training', 'inference', 'pytorch'].includes(m.pack) || m.id === 'data-preprocess')) {
    const cases = predictionCases.filter(c => m.id in c.expect);
    assert.ok(cases.some(c => c.expect[m.id]!.includes('concern')), `${m.id} needs a positive case`);
    assert.ok(cases.some(c => c.expect[m.id]!.includes('supported')), `${m.id} needs a counterexample`);
  }
});
test('unknowns and low-probability concerns stay visible without strong warnings', () => {
  const r = report(context(), 'concern', .65);
  const concern = r.assessments.find(a => a.id === 'train-grad-reset')!;
  assert.equal(concern.signal, 'concern'); assert.equal(concern.tentative, true);
  assert.ok(r.assessments.some(a => a.signal === 'unknown'));
  const resolved = report(context(), 'supported');
  assert.equal(resolved.assessments.find(a => a.id === concern.id)?.signal, 'supported');
  assert.match(inlineInsight(r), /Possible: Gradient reset/);
  assert.equal(inlineInsight(resolved), '◉ Code insights');
  assert.match(inlineInsight(report()), /^◉ Code insights · Gradient reset/);
});
test('assessment changes compare model outcomes in the same scope, never call unknown a fix', () => {
  const bad = report(), good = report(context(), 'supported');
  assert.match(changesSince(bad, good)[0]!, /Gradient reset misplaced or missing → Reset per update/);
  assert.match(changesSince(bad, report(context(), 'unknown'))[0]!, /Needs context/);
  assert.equal(changesSince(bad, report(context(), 'supported', .6)).length, 0);
  good.scope.startLine++; assert.equal(changesSince(bad, good).length, 1, 'line movement preserves scope identity');
  good.scope.identity = 'another-function'; assert.equal(changesSince(bad, good).length, 0);
});
test('real routing and assessment transport contracts use typed independent questions', async () => {
  const fetcher: typeof fetch = async (url, options) => {
    assert.equal(url, 'https://ai-gateway.vercel.sh/v1/evaluate');
    const p = JSON.parse(String(options?.body));
    assert.equal(p.model, 'typesafe-ai/jev'); assert.ok(p.state.current.code);
    return Response.json({ ...answers(p, p.questions.technology ? { technology: 'pytorch', activity: 'training' } : { 'train-grad-reset': 'concern' }), usage: { inputTokens: 1200 }, providerMetadata: { gateway: { cost: '0' } } });
  };
  const c = context(), first = await classify(c, 'synthetic-key', new AbortController().signal, 'auto', fetcher);
  const r = await assess(c, first.route, 'synthetic-key', new AbortController().signal, .75, fetcher);
  assert.equal(r.assessments.find(a => a.id === 'train-grad-reset')?.outcome, 'concern'); assert.equal(r.cost, 0);
});
function reviewEnvelope(ctx: CodeContext, patch: Record<string, unknown> = {}) {
  const line = ctx.code.split('\n').find(l => l.includes('loss.backward()'))!;
  return { choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({ summary: 'Gradients accumulate across independent updates.', items: [{ metricId: 'train-grad-reset', verdict: 'confirmed', title: 'Reset gradients between updates', explanation: 'Each update reuses accumulated gradients.', action: 'Call optimizer.zero_grad at the start of each independent update.', line: Number(line.match(/^L(\d+)/)![1]), evidence: 'loss.backward()', ...patch }] }) } }] };
}
test('review requests use Astra, a strict schema and independently verify Jev hypotheses', () => {
  const p = reviewPayload(context(), report(), undefined, { zeroDataRetention: true });
  assert.equal(p.model, 'openai/gpt-6-astra'); assert.equal(p.reasoning_effort, 'low'); assert.equal(p.response_format.json_schema.strict, true);
  assert.equal(p.providerOptions?.gateway.zeroDataRetention, true);
  assert.match(p.messages[0]!.content, /Jev classifications can be wrong/);
  assert.equal(Object.hasOwn(p, 'temperature'), false);
});
test('review source references are anchored only to exact visible code', () => {
  const c = context(), r = report(c);
  const good = parseReview(reviewEnvelope(c), c, r); assert.ok(good.items[0]!.line);
  const bad = parseReview(reviewEnvelope(c, { line: 999, evidence: 'made up code' }), c, r);
  assert.equal(bad.items[0]!.line, null); assert.equal(bad.items[0]!.evidence, '');
});
test('truncated, refused and malformed reviews never become recommendations', () => {
  const c = context(), r = report(c), truncated = reviewEnvelope(c); truncated.choices[0]!.finish_reason = 'length';
  assert.throws(() => parseReview(truncated, c, r), GatewayError);
  assert.throws(() => parseReview(reviewEnvelope(c, { metricId: 'fake-metric' }), c, r), GatewayError);
  assert.throws(() => parseReview({ choices: [{ finish_reason: 'stop', message: { content: 'not-json' } }] }, c, r), GatewayError);
});
test('generated review and handoff preserve context and never execute or rewrite source', async () => {
  const c = context(), r = report(c), before = JSON.stringify(c);
  const result = await review(c, r, 'secret-test-key', new AbortController().signal, undefined, async (url, options) => {
    assert.equal(url, 'https://ai-gateway.vercel.sh/v1/chat/completions'); assert.equal(options?.redirect, 'error');
    return Response.json({ ...reviewEnvelope(c), usage: { prompt_tokens: 900, completion_tokens: 200 } });
  });
  assert.equal(result.inputTokens, 900); assert.equal(result.outputTokens, 200);
  assert.match(handoff(c, r, result), /hypotheses, not validated facts/);
  assert.equal(JSON.stringify(c), before);
});
