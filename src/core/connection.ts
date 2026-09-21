import { BlockReader } from './blocks';
import { routingPayload } from './assessment';
import { GatewayError, parseAnswers, requestEvaluation, type GatewayOptions } from './gateway';

export interface ConnectionCheck { name: string; ok: boolean; detail: string }

// Fixed sample data only. Never accept a workspace snapshot here: the basic
// request intentionally has no ZDR override and error messages are displayed.
export async function testGatewayConnection(
  apiKey: string, signal: AbortSignal, fetcher: typeof fetch = fetch,
  onResult: (result: ConnectionCheck) => void = () => {},
  options: GatewayOptions = {},
): Promise<ConnectionCheck[]> {
  if (!apiKey.trim()) throw new GatewayError('Set an AI Gateway key before testing the connection.', 401);
  const basic = {
    model: 'typesafe-ai/jev',
    state: 'Please cancel my subscription and refund my payment.',
    questions: { wantsRefund: { type: 'boolean', instructions: 'Is the customer requesting a refund?' } },
  };
  const sample = routingPayload(new BlockReader().read({
    file: 'connection-test.cu', language: 'cuda-cpp',
    source: '__global__ void scale(float* x, int n) {\n  int i = blockIdx.x * blockDim.x + threadIdx.x;\n  if (i < n) x[i] *= 2.0f;\n}',
    cursorLine: 0, intent: 'Scale each of n elements by two.', references: [], maxCharacters: 4000,
  }), options);
  const booleanAnswer = (raw: unknown) => {
    const answer = (raw as { answers?: { wantsRefund?: { type?: string; probability?: number } } } | null)?.answers?.wantsRefund;
    if (answer?.type !== 'boolean' || typeof answer.probability !== 'number'
      || !Number.isFinite(answer.probability) || answer.probability < 0 || answer.probability > 1) {
      throw new GatewayError('Jev returned an incompatible answer to the connection test.', 502);
    }
  };
  const checks = [
    { name: 'Basic Jev request (same as curl)', payload: basic, validate: booleanAnswer },
    ...(options.zeroDataRetention ? [{ name: 'Same request with zero data retention', payload: { ...basic, providerOptions: sample.providerOptions }, validate: booleanAnswer }] : []),
    { name: 'Kernel Lens request with sample CUDA code', payload: sample, validate: (raw: unknown) => { parseAnswers(raw, sample); } },
  ];
  const results: ConnectionCheck[] = [];
  for (const check of checks) {
    signal.throwIfAborted();
    let result: ConnectionCheck;
    try {
      const raw = await requestEvaluation(check.payload, apiKey, signal, fetcher, true);
      check.validate(raw);
      result = { name: check.name, ok: true, detail: 'Gateway accepted the request and returned a valid Jev answer.' };
    } catch (error) {
      signal.throwIfAborted();
      result = { name: check.name, ok: false, detail: error instanceof GatewayError ? error.message
        : 'No HTTP response was received. The request timed out or could not connect from the extension host.' };
    }
    results.push(result); onResult(result);
    // Each check adds one difference. Stop at the first failure to isolate it.
    if (!result.ok) break;
  }
  return results;
}
