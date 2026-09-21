import { parseArgs } from 'node:util';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { BlockReader } from '../src/core/blocks';
import { assess, classify } from '../src/core/assessment';
import { predictionCases, type PredictionCase } from '../test/fixtures/predictions';
import { GatewayError } from '../src/core/gateway';
import { review } from '../src/core/advisor';

async function main() {
  const { values } = parseArgs({ options: { 'env-file': { type: 'string' }, 'cases-file': { type: 'string' }, filter: { type: 'string' }, output: { type: 'string' }, review: { type: 'boolean', default: false } } });
  if (values['env-file']) process.loadEnvFile(values['env-file']);
  const key = process.env.AI_GATEWAY_API_KEY;
  if (!key) throw new Error('Set AI_GATEWAY_API_KEY or pass --env-file. No requests sent.');
  const external = values['cases-file'] ? readFileSync(values['cases-file'], 'utf8') : undefined;
  const supplied = external ? JSON.parse(external) as { cases: PredictionCase[]; note?: string } : undefined;
  const note = supplied?.note ?? 'Small synthetic smoke corpus. Not a general accuracy measurement. All predictions and failures are preserved.';
  const corpusSha256 = external ? createHash('sha256').update(external).digest('hex') : undefined;
  const reader = new BlockReader(), results: Record<string, unknown>[] = [];
  let checks = 0, matches = 0, surfaced = 0;
  let lastRequest = 0;
  const save = () => {
    if (values.output) writeFileSync(values.output, JSON.stringify({ date: new Date().toISOString(), cases: results.length, matches, checks, nonTentativeMatches: surfaced,
      note, corpusSha256, results }, null, 2) + '\n');
  };
  const retry = async <T>(operation: () => Promise<T>): Promise<T> => {
    for (let attempt = 0; ; attempt++) {
      const delay = Math.max(0, lastRequest + 2200 - Date.now());
      if (delay) await new Promise(resolve => setTimeout(resolve, delay));
      lastRequest = Date.now();
      try { return await operation(); } catch (error) {
        if (!(error instanceof GatewayError) || !(error.status >= 500 || error.status === 429) || attempt >= 2) throw error;
        const wait = error.status === 429 ? error.retryAfterMs : 1500;
        console.log(JSON.stringify({ retry: attempt + 1, status: error.status, waitMs: wait }));
        await new Promise(resolve => setTimeout(resolve, wait));
      }
    }
  };
  const corpus: PredictionCase[] = supplied?.cases ?? predictionCases;
  const selected = corpus.filter(c => !values.filter || new RegExp(values.filter).test(c.name));
  if (!selected.length) throw new Error('No prediction cases matched.');
  for (const item of selected) {
    const context = reader.read({ file: item.file, language: item.file.endsWith('.cu') ? 'cpp' : 'python', source: item.source, cursorLine: item.cursorLine, intent: item.intent, secret: key });
    const { route, latencyMs: routeMs } = await retry(() => classify(context, key, new AbortController().signal));
    const report = await retry(() => assess(context, route, key, new AbortController().signal));
    const outcomes = Object.entries(item.expect).map(([id, expected]) => {
      const answer = report.assessments.find(a => a.id === id);
      const match = !!answer && expected.includes(answer.outcome);
      checks++; if (match) matches++; if (match && !answer?.tentative) surfaced++;
      return { id, expected, got: answer?.outcome ?? 'NOT_ROUTED', p: answer?.probability, match, tentative: answer?.tentative };
    });
    if (item.noStrongConcerns) {
      const unexpected = report.assessments.filter(a => a.signal === 'concern' && !a.tentative);
      const match = unexpected.length === 0; checks++; if (match) { matches++; surfaced++; }
      outcomes.push({ id: 'NO_UNEXPECTED_STRONG_CONCERNS', expected: ['none'], got: unexpected.map(a => a.id).join(',') || 'none', p: undefined, match, tentative: false });
    }
    const result = { name: item.name, route, routeMs, latencyMs: report.latencyMs, inputTokens: report.inputTokens, cost: report.cost, outcomes,
      context: { unit: context.unit, characters: context.characters, truncated: context.truncated, references: context.references.map(r => ({ name: r.name, reason: r.reason })) },
      profile: report.assessments.map(a => ({ id: a.id, choice: a.outcome, p: a.probability })) };
    results.push(result);
    save();
    console.log(JSON.stringify({ name: item.name, packs: route.packs, routeMs, latencyMs: report.latencyMs, outcomes }));
    if (values.review && results.length === 1) {
      const generated = await review(context, report, key, new AbortController().signal);
      results.push({ name: `${item.name}_astra_review`, review: generated }); save();
      console.log(JSON.stringify({ advisorModel: generated.model, latencyMs: generated.latencyMs, inputTokens: generated.inputTokens, outputTokens: generated.outputTokens, items: generated.items.length }));
    }
  }
  const summary = { date: new Date().toISOString(), cases: results.length, matches, checks, nonTentativeMatches: surfaced,
    note, corpusSha256, results };
  if (values.output) writeFileSync(values.output, JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ matches, checks, nonTentativeMatches: surfaced }));
  if (matches < checks) process.exitCode = 1;
}
void main().catch(error => { console.error(error instanceof Error ? error.message : 'Prediction evaluation failed'); process.exitCode = 1; });
