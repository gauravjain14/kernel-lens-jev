import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import type { KernelLensApi } from '../src/extension';
import manifest from '../package.json';
import { reductionKernel, serialReduction, shuffleReduction } from './fixtures/reduction';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
export async function run(): Promise<void> {
  const extension = vscode.extensions.getExtension<KernelLensApi>(`${manifest.publisher}.${manifest.name}`)!;
  const api = await extension.activate(); assert.ok(api.testing);
  const live = process.env.KERNEL_LENS_LIVE === '1';
  const config = vscode.workspace.getConfiguration('kernelLens');
  await config.update('debounceMs', 300, vscode.ConfigurationTarget.Workspace);
  await config.update('maxRequestsPerMinute', 120, vscode.ConfigurationTarget.Workspace);
  const waitFor = async (predicate: () => boolean, label: string, timeout = live ? 40000 : 15000) => {
    const start = Date.now();
    while (!predicate() && Date.now() - start < timeout) await delay(30);
    assert.ok(predicate(), `${label}: ${JSON.stringify(api.getState())}`);
  };
  const fetcher: typeof fetch = async (_url, options) => {
    const p = JSON.parse(String(options?.body));
    const state = JSON.stringify({ current: p.state?.current, enclosing: p.state?.enclosingCode });
    return Response.json({ answers: Object.fromEntries(Object.entries(p.questions).map(([id, q]) => {
      const keys = Object.keys((q as { criteria: Record<string, string> }).criteria);
      const choice = id.startsWith('anchor_') ? 'scope' : id === 'technology' ? 'cuda' : id === 'activity' ? 'kernel'
        : id === 'cuda-thread-work' ? state.includes('__shfl_down_sync') ? 'supported' : 'concern' : 'unknown';
      return [id, { type: 'choice', choice, probabilities: Object.fromEntries(keys.map(k => [k, k === choice ? .98 : .02 / (keys.length - 1)])) }];
    })), usage: { inputTokens: 900 }, providerMetadata: { gateway: { cost: '0' } } });
  };
  // Put the function header above the visible viewport, as in a real kernel.
  const initial = reductionKernel(shuffleReduction, '    // Other kernel setup\n'.repeat(45));
  const uri = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri, 'reduction.cu');
  await vscode.workspace.fs.writeFile(uri, Buffer.from(initial));
  const doc = await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(uri), 'cpp');
  const editor = await vscode.window.showTextDocument(doc);
  const select = (offset: number) => {
    const p = doc.positionAt(offset); editor.selection = new vscode.Selection(p, p);
    editor.revealRange(new vscode.Range(p, p), vscode.TextEditorRevealType.InCenter);
  };
  const start = initial.indexOf(shuffleReduction);
  select(start + shuffleReduction.length);
  const key = live ? process.env.AI_GATEWAY_API_KEY : 'synthetic-reduction-key'; assert.ok(key);
  await api.testing!.configure(key, true, live ? undefined : fetcher);
  const outcome = (value: string) => api.getState().phase === 'ready' && !api.getState().stale
    && api.getState().report?.assessments.some(a => a.id === 'cuda-thread-work' && a.outcome === value && !a.tentative) === true;
  await waitFor(() => outcome('supported'), 'shuffle reduction gets an automatic supported assessment');
  const before = api.getState();

  // Delete the old block, start typing the replacement, and pause with incomplete
  // syntax. No Analyze command is used anywhere in this regression.
  await editor.edit(b => b.delete(new vscode.Range(doc.positionAt(start), doc.positionAt(start + shuffleReduction.length))));
  select(start);
  const partial = '    if (threadIdx.x == 0) {\n        float sum = (';
  await editor.edit(b => b.insert(doc.positionAt(start), partial)); select(start + partial.length);
  await waitFor(() => api.getState().unit?.ready === false, 'unfinished expression waits for completion');
  const requestsWhileTyping = api.getState().requests;
  await delay(700); assert.equal(api.getState().requests, requestsWhileTyping, 'incomplete typing does not send an assessment');
  await editor.edit(b => b.replace(new vscode.Range(doc.positionAt(start), doc.positionAt(start + partial.length)), serialReduction));
  select(start + serialReduction.length);
  await waitFor(() => outcome('concern'), 'typing the completed serial replacement automatically flags thread participation');
  const after = api.getState();
  assert.ok(after.requests > before.requests); assert.equal(after.advisorRequests, 0);
  assert.deepEqual(after.report!.route.packs, ['cuda']);
  assert.equal(after.unit!.kind, 'IfStatement');
  assert.ok(after.report!.changes.some(c => c.startsWith('Thread participation:')));
  assert.equal(doc.getText(), initial.replace(shuffleReduction, serialReduction), 'Lens never rewrites the edited code');
  assert.ok(vscode.languages.getDiagnostics(uri).some(d => d.message.includes('Serial work in a narrow branch')));
  await vscode.commands.executeCommand('workbench.action.closeSidebar'); await delay(600);
  const root = process.env.KERNEL_LENS_TEST_ROOT!;
  if (process.env.KERNEL_LENS_CAPTURE === '1') execFileSync('python', ['-c',
    'from PIL import ImageGrab; import sys; ImageGrab.grab().save(sys.argv[1])',
    path.join(root, `artifacts/reduction-${live ? 'live' : 'mocked'}-editor-0.5.png`)]);

  await vscode.window.showTextDocument(doc);
  await editor.edit(b => b.replace(new vscode.Range(doc.positionAt(start), doc.positionAt(start + serialReduction.length)), shuffleReduction));
  select(start + shuffleReduction.length);
  await waitFor(() => outcome('supported'), 'restoring the shuffle reduction clears the concern automatically');
  const restored = api.getState();
  assert.ok(restored.improvements?.some(i => i.id === 'cuda-thread-work'), 'the corrected reduction has a specific improvement');

  // Complete the same inner block while the outer kernel is still being written.
  const unfinishedKernel = initial.slice(0, start) + serialReduction + '\n\n';
  await editor.edit(b => b.replace(new vscode.Range(new vscode.Position(0, 0), doc.lineAt(doc.lineCount - 1).range.end), unfinishedKernel));
  select(unfinishedKernel.length);
  await waitFor(() => outcome('concern') && api.getState().unit?.ready === true, 'finished inner block is assessed inside an unfinished kernel');
  assert.equal(api.getState().unit!.kind, 'IfStatement');
  await fs.writeFile(path.join(root, `artifacts/reduction-${live ? 'live' : 'mocked'}-host-0.5.json`), JSON.stringify({
    version: api.getState().version, extensionPath: extension.extensionPath, liveGateway: live,
    scenario: 'Delete shuffle, type unfinished expression, finish serial replacement, restore shuffle, finish inner block in unfinished kernel. No Analyze command.',
    before: before.report, after: after.report, restored: restored.report, unfinishedKernel: api.getState().report,
    requests: api.getState().requests, advisorRequests: api.getState().advisorRequests,
  }, null, 2) + '\n');
  console.log(`Kernel Lens ${live ? 'LIVE Jev' : 'mocked Gateway'} reduction edit regression passed.`);
}
