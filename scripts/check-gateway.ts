import { testGatewayConnection } from '../src/core/connection';
import { parseArgs } from 'node:util';
import { requestEvaluation } from '../src/core/gateway';

async function main(): Promise<void> {
  const { values, positionals } = parseArgs({ options: { zdr: { type: 'boolean', default: false }, model: { type: 'string' } }, allowPositionals: true });
  const envFile = positionals[0];
  if (envFile) {
    try { process.loadEnvFile(envFile); }
    catch { console.error('Could not load the specified environment file. No requests were made.'); process.exitCode = 1; return; }
  }
  const key = process.env.AI_GATEWAY_API_KEY;
  if (!key) { console.error('Set AI_GATEWAY_API_KEY, or pass an environment-file path. No requests were made.'); process.exitCode = 1; return; }
  console.log(`Gateway connection test · Node ${process.version} · fixed sample data only`);
  if (values.model) {
    try {
      await requestEvaluation({ model: values.model, max_completion_tokens: 100, reasoning_effort: 'low', messages: [{ role: 'user', content: 'Reply with OK.' }] }, key, new AbortController().signal, fetch, true, 'chat/completions');
      console.log('PASS · selected model accepted the fixed sample request.');
    } catch (error) { console.error(error instanceof Error ? error.message : 'Model test failed.'); process.exitCode = 1; }
    return;
  }
  const results = await testGatewayConnection(key, new AbortController().signal, fetch,
    result => console.log(`${result.ok ? 'PASS' : 'FAIL'} · ${result.name}\n${result.detail}`), { zeroDataRetention: values.zdr });
  if (results.some(result => !result.ok)) process.exitCode = 1;
}
void main().catch(() => { console.error('Connection test could not complete.'); process.exitCode = 1; });
