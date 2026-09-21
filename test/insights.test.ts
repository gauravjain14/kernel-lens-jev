import test from 'node:test';
import assert from 'node:assert/strict';
import { BlockReader, contextHash, withRecentEdit } from '../src/core/blocks';
import { sourceCandidates, anchorPayload, locateFindings } from '../src/core/anchors';
import { insightPresentation, selectedConcerns } from '../src/core/presentation';
import { guidance } from '../src/core/guidance';
import { metrics } from '../src/core/rubrics';
import type { Assessment, AssessmentReport } from '../src/live-types';
const source = `import torch\ndef train(loader):\n    for x in loader:\n        optimizer = torch.optim.AdamW(model.parameters())\n        loss = model(x).sum()\n        loss.backward()\n        optimizer.step()\n`;
const context = (s = source, line = 1) => new BlockReader().read({ file: 'train.py', language: 'python', source: s, cursorLine: line });
const assessment = (id = 'train-optimizer', outcome = 'concern', probability = .98): Assessment => {
  const metric = metrics.find(m => m.id === id)!, bucket = metric.outcomes[outcome]!;
  return { id, pack: metric.pack, label: metric.label, category: metric.category, outcome, bucket: bucket.label, signal: bucket.signal, probability, tentative: probability < .75, probabilities: { [outcome]: probability }, reference: metric.reference };
};
const report = (...assessments: Assessment[]): AssessmentReport => ({ route: { technology: 'pytorch', activity: 'training', packs: ['training', 'pytorch'], uncertain: false, probabilities: { technology: 1, activity: 1 } }, assessments, scope: { name: 'train', startLine: 2, endLine: 8 }, fingerprint: 'test', changes: [], latencyMs: 1 });
test('every model risk has a consequence, next check and specific evidence gap', () => {
  for (const m of metrics) assert.ok(guidance[m.id]?.every(s => s.length > 15), m.id);
});
test('risks are ranked, overlapping loader signals merge, and distinct dropout issues remain separate', () => {
  const r = report(assessment('tensor-transfer'), assessment('train-graph', 'concern', .6), assessment('train-optimizer'), assessment('infer-mode'), assessment('infer-attention-dropout'), assessment('train-loader'), assessment('data-iterator'));
  const ids = selectedConcerns(r).map(a => a.id);
  assert.equal(ids.length, 6); assert.equal(ids.at(-1), 'train-graph');
  assert.ok(ids.includes('infer-mode') && ids.includes('infer-attention-dropout'));
  const p = insightPresentation(r); assert.equal(p.insights.at(-1)?.kind, 'tentative');
  assert.ok(p.insights.some(i => i.relatedIds.length === 2)); assert.equal(r.assessments[1]?.probability, .6);
});
test('only a strong supported outcome yields an improvement; uncertainty never clears a risk', () => {
  const bad = report(assessment());
  assert.equal(insightPresentation(bad).insights[0]?.change, 'new');
  assert.equal(insightPresentation(bad, bad).insights[0]?.change, 'ongoing');
  assert.equal(insightPresentation(report(assessment('train-optimizer', 'supported')), bad).improvements[0]?.title, 'Optimizer reused');
  for (const next of [assessment('train-optimizer', 'unknown'), assessment('train-optimizer', 'not_applicable'), assessment('train-optimizer', 'supported', .6)]) {
    assert.equal(insightPresentation(report(next), bad).improvements.length, 0);
  }
});
test('overlapping eval hypotheses merge only after Jev anchors them to the same statement', () => {
  const r = report(assessment('infer-mode'), assessment('infer-attention-dropout'));
  r.anchors = { 'infer-mode': { startLine: 4, endLine: 4, code: 'dropout(x)', probability: .98 }, 'infer-attention-dropout': { startLine: 5, endLine: 5, code: 'dropout(y)', probability: .98 } };
  assert.equal(insightPresentation(r).insights.length, 2);
  r.anchors['infer-mode'] = r.anchors['infer-attention-dropout']!;
  const p = insightPresentation(r); assert.equal(p.insights.length, 1); assert.equal(p.insights[0]?.relatedIds.length, 2);
});
test('scope identity survives blank lines and distinguishes neighboring loops', () => {
  const a = context(), b = context('\n\n' + source, 3);
  assert.equal(a.unit.identity, b.unit.identity);
  const repeated = source + '    for y in loader:\n        consume(y)\n';
  assert.notEqual(context(repeated, 2).unit.identity, context(repeated, 8).unit.identity);
});
test('bounded edit evidence keeps removed operations separate and leaves current-context caching intact', () => {
  const a = context(), b = context(source.replace('        optimizer = torch.optim.AdamW(model.parameters())\n', ''));
  const edit = withRecentEdit(a, b, 16000);
  assert.match(edit.recentEdit!.before, /AdamW/); assert.equal(edit.recentEdit!.after, '');
  assert.equal(contextHash(edit), contextHash(b)); assert.ok(edit.characters <= 16000);
  assert.equal(withRecentEdit(a, b, b.characters).recentEdit, undefined);
});
test('source candidates are real current lines and fit a bounded request', () => {
  const c = context(); const candidates = sourceCandidates(c);
  assert.ok(candidates.some(s => s.startLine === 4 && s.code.includes('AdamW')));
  assert.ok(candidates.every(s => c.code.includes(`L${s.startLine}: ${s.code}`)));
  assert.ok(candidates.length <= 48);
  assert.equal(Object.keys(anchorPayload(c, report(assessment())).questions).length, 1);
});
test('Jev chooses an existing source line; low-confidence or scope choices never create precise evidence', async () => {
  for (const [choice, p] of [['L4', .98], ['L4', .55], ['scope', .99]] as const) {
    const fetcher: typeof fetch = async (_url, init) => {
      const payload = JSON.parse(String(init?.body));
      return Response.json({ answers: Object.fromEntries(Object.entries(payload.questions).map(([id, q]) => {
        const keys = Object.keys((q as { criteria: Record<string, string> }).criteria);
        return [id, { type: 'choice', choice, probabilities: Object.fromEntries(keys.map(k => [k, k === choice ? p : (1 - p) / (keys.length - 1)])) }];
      })) });
    };
    const r = await locateFindings(context(), report(assessment()), 'test', new AbortController().signal, fetcher);
    assert.equal(!!r.anchors['train-optimizer'], choice === 'L4' && p >= .75);
    if (r.anchors['train-optimizer']) assert.match(r.anchors['train-optimizer'].code, /AdamW/);
  }
});
test('presentation uses precise evidence only when the model supplied a valid anchor', () => {
  const r = report(assessment()); assert.equal(insightPresentation(r).insights[0]?.anchored, false);
  r.anchors = { 'train-optimizer': { startLine: 4, endLine: 4, code: 'optimizer = torch.optim.AdamW(model.parameters())', probability: .98 } };
  const finding = insightPresentation(r).insights[0]!;
  assert.equal(finding.startLine, 4); assert.equal(finding.anchored, true); assert.match(finding.consequence, /adaptive state/);
});
