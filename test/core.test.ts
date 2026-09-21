import test from 'node:test';
import assert from 'node:assert/strict';
import { isExcluded, redact } from '../src/core/context';
import { routingPayload as buildPayload, classify } from '../src/core/assessment';
import { GatewayError, parseAnswers, requestEvaluation, type GatewayOptions } from '../src/core/gateway';
import { BlockReader } from '../src/core/blocks';
import { ReportCache, RequestBudget, RevisionGuard } from '../src/core/scheduling';
import { testGatewayConnection } from '../src/core/connection';
import type { CodeContext } from '../src/live-types';
const fixture = () => new BlockReader().read({ file: 'x.cu', language: 'cpp', source: '__global__ void f(float* x) { int i = threadIdx.x; x[i] *= 2; }', cursorLine: 0 });
const evaluate = (snapshot: CodeContext, key: string, signal: AbortSignal, fetcher: typeof fetch = fetch, options: GatewayOptions = {}) => classify(snapshot, key, signal, 'auto', fetcher, options);
function envelope(snapshot = fixture()) {
  const payload = buildPayload(snapshot);
  const answers = Object.fromEntries(Object.entries(payload.questions).map(([id, q]) => {
    const choice = id === 'technology' ? 'cuda' : 'kernel', options = Object.keys(q.criteria);
    return [id, { type: 'choice', choice, probabilities: Object.fromEntries(options.map(key => [key, key === choice ? .97 : .03 / (options.length - 1)])) }];
  }));
  return { answers, usage: { inputTokens: 321 }, providerMetadata: { gateway: { cost: '0.0000123' } } };
}

test('sensitive and dependency paths are excluded on Windows and Unix', () => {
  for (const name of ['.env', '.env.local', 'a/.env.production', 'C:\\repo\\.ssh\\config', 'a/node_modules/x.py', 'cert.key']) assert.ok(isExcluded(name), name);
  assert.ok(isExcluded('/repo/proprietary/kernel.cu', ['proprietary']));
  assert.equal(isExcluded('/repo/kernels/reduce.cu'), false);
});
test('common secret literals are redacted and code remains intact', () => {
  const value = 'api_key = "vck_abcdefghijklmnop"\npassword="sensitive"\nx = tl.load(ptr)';
  assert.equal(redact(value), 'api_key = "[REDACTED]"\npassword="[REDACTED]"\nx = tl.load(ptr)');
});
test('rounded distributions are accepted, missing and invalid probabilities rejected', () => {
  const snapshot = fixture(); const payload = buildPayload(snapshot);
  const raw = envelope(snapshot);
  assert.ok(parseAnswers(raw, payload)['technology']);
  const broken = structuredClone(raw);
  broken.answers['technology']!.probabilities.cuda = NaN;
  assert.throws(() => parseAnswers(broken, payload), GatewayError);
  delete broken.answers['technology'];
  assert.throws(() => parseAnswers(broken, payload), GatewayError);
});
test('gateway sends the real request contract and extracts usage without exposing credentials', async () => {
  const snapshot = fixture();
  const fetcher: typeof fetch = async (url, options) => {
    assert.equal(url, 'https://ai-gateway.vercel.sh/v1/evaluate');
    assert.equal(new Headers(options?.headers).get('Authorization'), 'Bearer test-key-only');
    assert.equal(JSON.parse(String(options?.body)).model, 'typesafe-ai/jev');
    assert.equal(JSON.parse(String(options?.body)).providerOptions, undefined, 'standard requests work on Hobby without a paid routing option');
    assert.equal(options?.redirect, 'error');
    return Response.json(envelope(snapshot));
  };
  const result = await evaluate(snapshot, 'test-key-only', new AbortController().signal, fetcher);
  assert.equal(result.inputTokens, 321); assert.equal(result.cost, 0.0000123);
});
test('explicit ZDR requests preserve the option and never retry a plan denial without it', async () => {
  let calls = 0;
  const fetcher: typeof fetch = async (_url, options) => {
    calls++;
    assert.equal(JSON.parse(String(options?.body)).providerOptions.gateway.zeroDataRetention, true);
    return Response.json({ error: { type: 'permission_denied', message: 'Zero Data Retention (ZDR) is only available for Pro and Enterprise plans. Current plan: hobby.' } }, { status: 403 });
  };
  await assert.rejects(evaluate(fixture(), 'test-key-only', new AbortController().signal, fetcher, { zeroDataRetention: true }), (error: unknown) => {
    assert.ok(error instanceof GatewayError);
    assert.equal(error.status, 403);
    assert.match(error.message, /Turn off Kernel Lens: Zero Data Retention/);
    assert.match(error.message, /permission_denied/);
    return true;
  });
  assert.equal(calls, 1);
});
test('403 errors preserve status, type and request ID without echoing source or key', async () => {
  await assert.rejects(evaluate(fixture(), 'test-key-only', new AbortController().signal,
    async () => Response.json({ error: { type: 'access_denied', message: 'private source code test-key-only' } },
      { status: 403, headers: { 'x-vercel-id': 'sfo1::test-key-only::example' } })), (error: unknown) => {
    assert.ok(error instanceof GatewayError);
    assert.match(error.message, /HTTP 403/); assert.match(error.message, /access_denied/);
    assert.match(error.message, /request sfo1::\[REDACTED\]::example/);
    assert.ok(!error.message.includes('private source')); assert.ok(!error.message.includes('test-key-only'));
    return true;
  });
});
test('connection check tests the curl shape and real Kernel Lens shape with sample data', async () => {
  const bodies: any[] = [];
  const results = await testGatewayConnection('test-key-only', new AbortController().signal, async (_url, options) => {
    assert.equal(new Headers(options?.headers).get('Authorization'), 'Bearer test-key-only');
    const body = JSON.parse(String(options?.body)); bodies.push(body);
    assert.equal(body.providerOptions, undefined);
    if (body.questions.wantsRefund) return Response.json({ answers: { wantsRefund: { type: 'boolean', probability: 0.99 } } });
    const answers = Object.fromEntries(Object.entries(body.questions).map(([id, value]) => {
      const criteria = (value as { criteria: Record<string, string> }).criteria;
      return [id, { type: 'choice', choice: Object.keys(criteria)[0], probabilities: Object.fromEntries(Object.keys(criteria).map((key, index) => [key, index === 0 ? 1 : 0])) }];
    }));
    return Response.json({ answers });
  });
  assert.equal(results.length, 2); assert.ok(results.every(result => result.ok));
  assert.equal(bodies[0].state, 'Please cancel my subscription and refund my payment.');
  assert.equal(bodies[1].state.current.file, 'connection-test.cu');
  assert.deepEqual(bodies[1].state.references, []);
});
test('connection check isolates a ZDR plan denial and redacts provider diagnostics', async () => {
  let calls = 0;
  const results = await testGatewayConnection('test-key-only', new AbortController().signal, async (_url, options) => {
    const body = JSON.parse(String(options?.body)); calls++;
    if (!body.providerOptions) return Response.json({ answers: { wantsRefund: { type: 'boolean', probability: 0.99 } } });
    return Response.json({ error: { type: 'permission_denied', message: 'ZDR unavailable for hobby. Authorization: Bearer test-key-only\n' } }, { status: 403 });
  }, () => {}, { zeroDataRetention: true });
  assert.equal(calls, 2); assert.equal(results.length, 2);
  assert.equal(results[0]?.ok, true); assert.equal(results[1]?.ok, false);
  assert.match(results[1]!.detail, /ZDR unavailable for hobby/);
  assert.match(results[1]!.detail, /\[REDACTED\]/);
  assert.ok(!results[1]!.detail.includes('test-key-only'));
});
test('connection check identifies HTML denials and cancellation stops further requests', async () => {
  const failed = await testGatewayConnection('test-key-only', new AbortController().signal,
    async () => new Response('<html>Access denied: test-key-only</html>', { status: 403, headers: { 'content-type': 'text/html' } }));
  assert.equal(failed.length, 1); assert.equal(failed[0]?.ok, false);
  assert.match(failed[0]!.detail, /HTML instead of a Gateway JSON error/);
  assert.ok(!failed[0]!.detail.includes('test-key-only'));
  const controller = new AbortController(); let calls = 0;
  await assert.rejects(testGatewayConnection('test-key-only', controller.signal, async () => {
    calls++; controller.abort();
    return Response.json({ answers: { wantsRefund: { type: 'boolean', probability: 0.99 } } });
  }), /aborted/i);
  assert.equal(calls, 1);
});
test('401 responses become actionable errors and never reflect the raw body', async () => {
  await assert.rejects(evaluate(fixture(), 'test-key-only', new AbortController().signal,
    async () => new Response('secret-source-code', { status: 401 })), (err: unknown) => {
    assert.ok(err instanceof GatewayError); assert.equal(err.status, 401); assert.ok(!err.message.includes('secret-source')); return true;
  });
});
test('429 retry-after becomes a bounded cooldown', async () => {
  await assert.rejects(evaluate(fixture(), 'test-key-only', new AbortController().signal,
    async () => new Response('', { status: 429, headers: { 'retry-after': '17' } })), (err: unknown) => {
    assert.ok(err instanceof GatewayError); assert.equal(err.retryAfterMs, 17000); return true;
  });
});
test('an edit cancels in-flight requests and disqualifies their stale results', () => {
  const guard = new RevisionGuard();
  const old = guard.invalidate(), signal = guard.signal(old);
  const next = guard.invalidate();
  assert.equal(signal.aborted, true); assert.equal(guard.current(old), false); assert.equal(guard.current(next), true);
});
test('transport failures use the bounded retry path without exposing request details', async () => {
  const key = 'private-test-key';
  await assert.rejects(requestEvaluation({}, key, new AbortController().signal, async () => { throw new TypeError(`fetch failed: ${key}`); }),
    (error: unknown) => error instanceof GatewayError && error.status === 503 && !error.message.includes(key));
  const controller = new AbortController(); controller.abort();
  const canceled = new DOMException('Canceled', 'AbortError');
  await assert.rejects(requestEvaluation({}, key, controller.signal, async () => { throw canceled; }), error => error === canceled);
});
test('rolling request budget handles spacing, capacity, and server cooldown', () => {
  const budget = new RequestBudget();
  assert.equal(budget.delay(2, 0), 0);
  budget.record(0); assert.equal(budget.delay(2, 500), 1500);
  budget.record(2000); assert.equal(budget.delay(2, 3000), 57000);
  assert.equal(budget.delay(2, 60001), 0);
  budget.cooldown(30000, 60001); assert.equal(budget.delay(2, 61001), 29000);
});
test('cache expires and uses bounded storage', () => {
  const cache = new ReportCache<number>(); cache.set('a', 1, 0);
  assert.equal(cache.get('a', 100), 1); assert.equal(cache.get('a', 120001), undefined);
  for (let i = 0; i < 30; i++) cache.set(String(i), i, 0);
  assert.equal(cache.get('0', 1), undefined); assert.equal(cache.get('29', 1), 29);
});
