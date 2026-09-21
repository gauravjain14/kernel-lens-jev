import { redact } from './context';
import type { Choice } from '../live-types';
export interface GatewayOptions { zeroDataRetention?: boolean }

export class GatewayError extends Error {
  constructor(message: string, public readonly status: number, public readonly retryAfterMs = 0) { super(message); }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

export function parseAnswers(raw: unknown, payload: { questions: Record<string, { criteria: Record<string, string> }> }): Record<string, Choice> {
  if (!isRecord(raw) || !isRecord(raw.answers)) throw new GatewayError('Jev returned an invalid answer envelope. Try Analyze again.', 502);
  const answers: Record<string, Choice> = {};
  for (const [id, question] of Object.entries(payload.questions)) {
    const answer = raw.answers[id];
    if (!isRecord(answer) || answer.type !== 'choice' || typeof answer.choice !== 'string'
      || !Object.hasOwn(question.criteria, answer.choice) || !isRecord(answer.probabilities)) {
      throw new GatewayError('Jev returned an incomplete or incompatible decision. Try Analyze again.', 502);
    }
    const probabilities: Record<string, number> = {};
    for (const key of Object.keys(question.criteria)) {
      const p = answer.probabilities[key];
      if (typeof p !== 'number' || !Number.isFinite(p) || p < 0 || p > 1) throw new GatewayError('Jev returned an invalid probability distribution.', 502);
      probabilities[key] = p;
    }
    const sum = Object.values(probabilities).reduce((a, b) => a + b, 0);
    const selected = probabilities[answer.choice] ?? 0;
    if (Math.abs(sum - 1) > 0.06 || Object.values(probabilities).some(p => p > selected + 0.011)) {
      throw new GatewayError('Jev returned an inconsistent probability distribution.', 502);
    }
    answers[id] = { type: 'choice', choice: answer.choice, probabilities };
  }
  return answers;
}

function safeDetail(value: unknown, apiKey: string, limit = 800): string {
  if (typeof value !== 'string') return '';
  let text = value;
  for (const secret of new Set([apiKey.trim(), encodeURIComponent(apiKey.trim())])) {
    if (secret) text = text.replaceAll(secret, '[REDACTED]');
  }
  return redact(text).replace(/[\u0000-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g, ' ').slice(0, limit);
}

export async function requestEvaluation(
  payload: unknown, apiKey: string, signal: AbortSignal,
  fetcher: typeof fetch = fetch, includeErrorMessage = false, api: 'evaluate' | 'chat/completions' = 'evaluate',
): Promise<unknown> {
  if (!apiKey.trim()) throw new GatewayError('Set an AI Gateway key to enable live insights.', 401);
  const timeout = AbortSignal.timeout(api === 'evaluate' ? 15_000 : 45_000);
  let response: Response;
  try {
    response = await fetcher(`https://ai-gateway.vercel.sh/v1/${api}`, {
      method: 'POST', signal: AbortSignal.any([signal, timeout]), redirect: 'error',
      headers: { Authorization: `Bearer ${apiKey.trim()}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    if (signal.aborted) throw error;
    // Use the controller's bounded transient retry without exposing a transport
    // exception that could contain request headers or source text.
    throw new GatewayError(timeout.aborted ? 'AI Gateway timed out. Retry Analyze if the connection does not recover.'
      : 'Could not reach AI Gateway. Retry Analyze if the connection does not recover.', timeout.aborted ? 504 : 503);
  }
  if (!response.ok) {
    let raw: unknown;
    try { raw = await response.json(); } catch { /* Non-JSON proxy or edge response. */ }
    const envelope = isRecord(raw) ? raw : {};
    const details = isRecord(envelope.error) ? envelope.error : envelope;
    const kind = typeof details.type === 'string' && /^[a-z][a-z0-9_]{0,63}$/.test(details.type)
      ? safeDetail(details.type, apiKey, 64) : '';
    const requestId = safeDetail(response.headers.get('x-vercel-id') ?? response.headers.get('x-request-id'), apiKey, 160);
    const providerMessage = typeof envelope.error === 'string' ? envelope.error : details.message;
    const retry = response.headers.get('retry-after');
    const retryMs = retry ? (/^\d+(?:\.\d+)?$/.test(retry) ? Number(retry) * 1000 : Date.parse(retry) - Date.now()) : 30_000;
    const messages: Record<number, string> = {
      401: 'AI Gateway rejected the credential. Update your AI Gateway key, then Analyze again.',
      402: 'AI Gateway needs available credits. Check the team’s Gateway balance.',
      403: 'AI Gateway denied this request. Run Kernel Lens: Test AI Gateway Connection for details.',
      404: 'The Jev evaluation endpoint is unavailable. Check Vercel’s Gateway status.',
      429: 'AI Gateway rate limited this request. Live checks will resume after a short cooldown.',
    };
    if (response.status === 403 && kind === 'permission_denied' && typeof providerMessage === 'string'
      && /Zero Data Retention.*only available for Pro and Enterprise/i.test(providerMessage)) {
      messages[403] = 'This Gateway plan does not support zero data retention. Turn off Kernel Lens: Zero Data Retention in Settings to use standard Gateway routing.';
    }
    if (api === 'chat/completions' && kind === 'no_providers_available') {
      messages[403] = 'No AI Gateway provider is available for the selected review model on this account. Check model access and credits, choose another review model in Settings, or use Copy for coding agent. Jev assessments remain available.';
    }
    if (api === 'chat/completions') messages[404] = 'The selected review model is unavailable. Check Kernel Lens: Advisor Model in Settings.';
    // Only the connection test opts into provider messages, with fixed sample data.
    // Normal analysis can contain source code echoed by the provider.
    const message = includeErrorMessage ? safeDetail(providerMessage, apiKey) : '';
    const context = [`HTTP ${response.status}`, kind, requestId ? `request ${requestId}` : ''].filter(Boolean).join(' · ');
    const format = /text\/html/i.test(response.headers.get('content-type') ?? '') ? ' The server returned HTML instead of a Gateway JSON error.' : '';
    throw new GatewayError(`${message || messages[response.status] || 'AI Gateway could not complete the request.'} [${context}]${format}`, response.status,
      response.status === 429 ? Math.max(5000, Math.min(120_000, Number.isFinite(retryMs) ? retryMs : 30_000)) : 0);
  }
  try { return await response.json(); }
  catch { throw new GatewayError('AI Gateway returned a response that is not valid JSON.', 502); }
}
