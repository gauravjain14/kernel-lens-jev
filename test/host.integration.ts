import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import type { KernelLensApi } from '../src/extension';
import manifest from '../package.json';

const bad = `import torch\nmodel = torch.nn.Linear(8, 2).cuda()\noptimizer = torch.optim.AdamW(model.parameters())\ndef train(loader):\n    model.train()\n    for x, y in loader:\n        x, y = x.cuda(), y.cuda()\n        loss = torch.nn.functional.cross_entropy(model(x), y)\n        loss.backward()\n        optimizer.step()\n`;
const good = bad.replace('        loss =', '        optimizer.zero_grad(set_to_none=True)\n        loss =');
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
export async function run(): Promise<void> {
  const extension = vscode.extensions.getExtension<KernelLensApi>(`${manifest.publisher}.${manifest.name}`)!;
  assert.ok(extension); const api = await extension.activate(); assert.ok(api.testing);
  assert.equal(api.getState().version, manifest.version);
  const root = process.env.KERNEL_LENS_TEST_ROOT ?? extension.extensionPath;
  const config = vscode.workspace.getConfiguration('kernelLens');
  await config.update('debounceMs', 300, vscode.ConfigurationTarget.Workspace);
  await config.update('maxRequestsPerMinute', 120, vscode.ConfigurationTarget.Workspace);
  const waitFor = async (predicate: () => boolean, label: string, timeout = 15000) => {
    const start = Date.now(); while (!predicate() && Date.now() - start < timeout) await delay(30);
    assert.ok(predicate(), `${label}: ${JSON.stringify(api.getState())}`);
  };
  const uri = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri, 'training_step.py');
  await vscode.workspace.fs.writeFile(uri, Buffer.from(bad));
  const doc = await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(uri), 'python');
  let editor = await vscode.window.showTextDocument(doc);
  editor.selection = new vscode.Selection(5, 0, 5, 0);
  await waitFor(() => api.getState().phase === 'disabled', 'no key produces setup state, never fabricated semantic results');
  assert.equal(api.getState().requests, 0); assert.equal(api.getState().report, undefined);
  const replace = async (source: string) => {
    editor = await vscode.window.showTextDocument(doc);
    editor.selection = new vscode.Selection(0, 0, 0, 0);
    await editor.edit(b => b.replace(new vscode.Range(new vscode.Position(0, 0), doc.lineAt(doc.lineCount - 1).range.end), source));
    editor.selection = new vscode.Selection(Math.min(5, doc.lineCount - 1), 0, Math.min(5, doc.lineCount - 1), 0);
  };
  const live = process.env.KERNEL_LENS_LIVE === '1';
  const calls: { api: string; code: string }[] = [];
  let fail: number | undefined, hold: (() => void) | undefined, delayed = false;
  let delayReview = false, reviewHeld: (() => void) | undefined;
  let delayAnchors = false, anchorsHeld: (() => void) | undefined;
  const fetcher: typeof fetch = async (url, options) => {
    const p = JSON.parse(String(options?.body));
    calls.push({ api: String(url), code: p.state?.current?.code ?? '' });
    if (delayAnchors && Object.keys(p.questions ?? {}).some(id => id.startsWith('anchor_'))) {
      delayAnchors = false; await new Promise<void>(resolve => { anchorsHeld = resolve; });
    }
    if (delayReview && String(url).endsWith('/chat/completions')) { delayReview = false; await new Promise<void>(resolve => { reviewHeld = resolve; }); }
    if (delayed) { delayed = false; await new Promise<void>(resolve => { hold = resolve; }); }
    if (fail) { const status = fail; fail = undefined; if (status === -1) throw new TypeError('fetch failed'); return Response.json({ error: { type: 'test_error' } }, { status, headers: { 'retry-after': '0' } }); }
    if (String(url).endsWith('/chat/completions')) return Response.json({ choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({ summary: 'Reset gradients between independent updates.', items: [{ metricId: 'train-grad-reset', verdict: 'confirmed', title: 'Gradients persist across updates', explanation: 'This loop calls backward and step without resetting gradients.', action: 'Clear gradients at the independent update boundary.', line: 9, evidence: 'loss.backward()' }] }) } }], usage: { prompt_tokens: 500, completion_tokens: 80 } });
    const answers = Object.fromEntries(Object.entries(p.questions).map(([id, question]) => {
      const q = question as { criteria: Record<string, string> }, keys = Object.keys(q.criteria);
      const currentSource = [p.state?.current?.code, p.state?.enclosingCode, p.state?.importsAndSetup].join('\n');
      const choice = id.startsWith('anchor_') ? keys.find(k => k !== 'scope' && q.criteria[k]?.includes('loss.backward()')) ?? 'scope'
        : id === 'technology' ? 'pytorch' : id === 'activity' ? 'training' : id === 'train-grad-reset' ? (currentSource.includes('zero_grad') ? 'supported' : 'concern') : 'unknown';
      return [id, { type: 'choice', choice, probabilities: Object.fromEntries(keys.map(k => [k, k === choice ? .98 : .02 / (keys.length - 1)])) }];
    }));
    return Response.json({ answers, usage: { inputTokens: 500 }, providerMetadata: { gateway: { cost: '0' } } });
  };
  if (live) {
    const key = process.env.AI_GATEWAY_API_KEY; assert.ok(key, 'live test key is provided without printing it');
    await api.testing!.configure(key, true);
  } else {
    await api.testing!.configure('synthetic-host-key', false, fetcher);
    await delay(600); assert.equal(calls.length, 0, 'a key alone never sends code without workspace permission');
    await api.testing!.configure('synthetic-host-key', true, fetcher);
  }
  await waitFor(() => api.getState().phase === 'ready' && api.getState().report?.assessments.some(a => a.id === 'train-grad-reset' && a.signal === 'concern') === true, 'typing/opening complete code automatically produces a Jev profile', live ? 30000 : 15000);
  const first = api.getState();
  assert.equal(first.report?.route.activity, 'training');
  for (const id of ['train-grad-reset', 'train-graph', 'train-optimizer', 'train-ddp-accum', 'tensor-batching', 'tensor-compile']) {
    assert.ok(first.report?.assessments.some(a => a.id === id), `training profile includes ${id}`);
  }
  assert.equal(first.advisorRequests, 0, 'Astra is not invoked by default');
  assert.ok(vscode.languages.getDiagnostics(uri).some(d => d.source === 'Kernel Lens · Jev'));
  await vscode.commands.executeCommand('kernelLens.open'); await delay(500);
  assert.equal(api.getState().phase, 'ready', 'webview focus preserves the profile');
  if (process.env.KERNEL_LENS_CAPTURE === '1') execFileSync('python', ['-c', 'from PIL import ImageGrab; import sys; ImageGrab.grab().save(sys.argv[1])', path.join(root, 'artifacts/kernel-lens-live-profile.png')]);
  const proof = { version: first.version, extensionPath: extension.extensionPath, liveGateway: live, report: first.report, requests: first.requests };
  if (!live) {
    const before = calls.length;
    await vscode.commands.executeCommand('kernelLens.analyze'); await waitFor(() => !api.getState().stale && api.getState().phase === 'ready', 'unchanged context reuses cached results');
    assert.equal(calls.length, before);
    await vscode.commands.executeCommand('kernelLens.review');
    assert.equal(api.getState().advisor.phase, 'ready'); assert.equal(api.getState().advisorRequests, 1);
    assert.ok(api.getState().advisor.result?.items[0]?.line);
    await vscode.commands.executeCommand('kernelLens.copyHandoff'); assert.match(await vscode.env.clipboard.readText(), /train-grad-reset/);
    assert.equal(doc.getText(), bad, 'reviews and handoffs do not rewrite code');
  }
  await replace(good);
  await waitFor(() => !api.getState().stale && api.getState().report?.assessments.some(a => a.id === 'train-grad-reset' && a.signal === 'supported') === true, 'fixing reset updates the live classification', live ? 30000 : 15000);
  assert.equal(api.getState().advisor.result, undefined, 'old recommendations do not survive an edit');
  assert.ok(api.getState().report?.changes.some(c => c.includes('Reset per update')));
  if (!live) {
    const locationSource = bad.replace('optimizer.step()', 'optimizer.step()  # delayed location selection');
    delayAnchors = true; await replace(locationSource);
    await waitFor(() => !!anchorsHeld, 'primary findings show while source location is still in flight');
    assert.equal(api.getState().phase, 'ready');
    await replace(good); anchorsHeld!(); anchorsHeld = undefined;
    await waitFor(() => api.getState().phase === 'ready' && !api.getState().stale, 'correction supersedes old location request');
    assert.ok(!api.getState().insights?.some(i => i.id === 'train-grad-reset'), 'a late location cannot resurrect a corrected finding');
    const beforeResume = calls.length;
    await replace(locationSource);
    await waitFor(() => !!api.getState().report?.anchors?.['train-grad-reset'] && !api.getState().stale, 'cached primary result resumes interrupted source selection');
    assert.equal(calls.length, beforeResume + 1, 'resuming location selection does not repeat routing or assessment');
    await replace(good); await waitFor(() => api.getState().phase === 'ready' && !api.getState().stale, 'cached corrected revision restored');
    // Deliberately let an obsolete response arrive after a newer document edit.
    await api.testing!.configure('synthetic-host-key', true, fetcher);
    await waitFor(() => api.getState().phase === 'ready', 'fresh mocked route ready');
    delayed = true; await replace(bad + '# obsolete revision\n');
    await waitFor(() => !!hold, 'old model request is in flight');
    await replace(good + '# latest revision\n'); hold!(); hold = undefined;
    await waitFor(() => !api.getState().stale && api.getState().report?.assessments.some(a => a.id === 'train-grad-reset' && a.signal === 'supported') === true, 'late stale response is rejected');
    await replace('def unfinished():\n    x = (');
    await waitFor(() => api.getState().unit?.ready === false, 'incomplete block defers model calls');
    const incompleteCalls = calls.length; await delay(700); assert.equal(calls.length, incompleteCalls);
    assert.ok(vscode.languages.getDiagnostics(uri).some(d => d.code === 'syntax'));
    // An authentication failure stays explicit and never becomes an empty success profile.
    fail = 401; await replace(bad + '# denied\n');
    await waitFor(() => api.getState().phase === 'error', 'Gateway errors are visible');
    assert.match(api.getState().message, /credential/);
    await vscode.commands.executeCommand('kernelLens.analyze');
    await waitFor(() => api.getState().phase === 'ready', 'manual retry clears auth block');
    // Retry after a rate limit without a keystroke or another Analyze click.
    fail = 429; await replace(bad.replace('optimizer.step()', 'optimizer.step(closure=None)'));
    await waitFor(() => api.getState().phase === 'limited', 'rate limit is surfaced');
    await waitFor(() => api.getState().phase === 'ready', 'rate limited work resumes automatically');
    fail = -1; await replace(bad.replace('optimizer.step()', 'optimizer.step(closure = None)'));
    await waitFor(() => api.getState().phase === 'limited', 'temporary connection failure is surfaced');
    await waitFor(() => api.getState().phase === 'ready', 'transport failure recovers without Analyze');
    // Automatic reviews run only after explicit configuration, and a late review
    // for old code cannot reappear after the author fixes the code.
    delayReview = true;
    await config.update('advisor.mode', 'auto', vscode.ConfigurationTarget.Workspace);
    await waitFor(() => !!reviewHeld, 'automatic review respects its separate budget and eventually starts', 35000);
    await replace(good); reviewHeld!(); reviewHeld = undefined;
    await waitFor(() => !api.getState().stale && api.getState().report?.assessments.some(a => a.id === 'train-grad-reset' && a.signal === 'supported') === true, 'fixed revision replaces the reviewed revision');
    assert.equal(api.getState().advisor.result, undefined, 'late advice is discarded');
    assert.equal(api.getState().advisorRequests, 2);
    await config.update('advisor.mode', 'onDemand', vscode.ConfigurationTarget.Workspace);
    const setupUri = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri, 'setup_model.py');
    await vscode.workspace.fs.writeFile(setupUri, Buffer.from('import torch\nmodel = torch.nn.Linear(8, 2).cuda()\n'));
    await replace('from unused_a import NeverCalledA\nfrom unused_b import NeverCalledB\nfrom unused_c import NeverCalledC\n' + bad.replace('model = torch.nn.Linear(8, 2).cuda()', 'from setup_model import model'));
    await waitFor(() => api.getState().phase === 'ready' && !api.getState().stale && api.getState().context?.references.some(r => r.name === 'setup_model.py') === true,
      'a used workspace import supplies model setup without a language server or manual attachment');
    assert.equal(api.getState().context?.references.some(r => r.name.startsWith('unused_')), false);
  }
  const excluded = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri, '.env.py');
  await vscode.workspace.fs.writeFile(excluded, Buffer.from('private = 1'));
  await vscode.window.showTextDocument(await vscode.workspace.openTextDocument(excluded));
  await waitFor(() => api.getState().phase === 'unsupported', 'excluded files never reach the model');
  await vscode.commands.executeCommand('kernelLens.pause');
  const pausedCalls = api.getState().requests;
  await replace(bad); await delay(700); assert.equal(api.getState().requests, pausedCalls);
  assert.equal(vscode.languages.getDiagnostics(uri).filter(d => d.source?.includes('Kernel Lens')).length, 0);
  await vscode.commands.executeCommand('kernelLens.enable');
  await waitFor(() => api.getState().phase === 'ready', 'existing key and permission resume without setup');
  await fs.writeFile(path.join(root, live ? 'artifacts/live-host-verification-0.5.json' : 'artifacts/package-verification-0.5.json'), JSON.stringify(proof, null, 2) + '\n');
  console.log(live ? 'Kernel Lens LIVE Jev editor test passed: automatic assessment, corrected-code update, excluded files and pause/resume. No Astra review requested.'
    : 'Kernel Lens mocked Gateway editor integration passed: automatic assessments, fixes, caching, permission, manual/automatic review, late response cancellation, exclusions and pause/resume.');
}
