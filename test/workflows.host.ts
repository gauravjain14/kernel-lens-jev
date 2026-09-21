import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import type { KernelLensApi } from '../src/extension';
import { workflowCases } from './fixtures/workflows';
import manifest from '../package.json';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));
export async function run(): Promise<void> {
  const extension = vscode.extensions.getExtension<KernelLensApi>(`${manifest.publisher}.${manifest.name}`)!;
  const api = await extension.activate(); assert.ok(api.testing);
  const live = process.env.KERNEL_LENS_LIVE === '1';
  const root = process.env.KERNEL_LENS_TEST_ROOT!;
  const config = vscode.workspace.getConfiguration('kernelLens');
  await config.update('debounceMs', 700, vscode.ConfigurationTarget.Workspace);
  await config.update('maxRequestsPerMinute', live ? 30 : 120, vscode.ConfigurationTarget.Workspace);
  const filter = process.env.KERNEL_LENS_WORKFLOW_FILTER;
  const names = ['detached_loss', 'optimizer_lifetime', 'decode_cache', 'padding_mask', 'attention_dropout', 'compiled_control'].filter(name => !filter || name === filter);
  assert.ok(names.length, 'workflow filter names a supported fixture pair');
  const evidence: Record<string, string> = {
    'train-graph': 'loss.detach()', 'train-optimizer': 'optimizer = torch.optim.AdamW',
    'infer-kv-cache': 'use_cache=False', 'infer-padding': 'return model(input_ids=',
    'infer-attention-dropout': 'dropout_p=0.2', 'tensor-compile': '.item()',
  };
  let active = workflowCases.find(c => c.name === `audit_${names[0]}_concern`)!;
  const fetcher: typeof fetch = async (url, options) => {
    assert.ok(String(url).endsWith('/evaluate'), 'no generated review is needed for these live insights');
    const p = JSON.parse(String(options?.body));
    // The same clean source is reused by several pairs. Its mocked assessment
    // must be consistent when the extension reuses an unchanged-context report.
    const expected = Object.assign({}, ...workflowCases.filter(c => c.source === active.source).map(c => c.expect));
    const answers = Object.fromEntries(Object.entries(p.questions).map(([id, question]) => {
      const criteria = (question as { criteria: Record<string, string> }).criteria, keys = Object.keys(criteria);
      const choice = id.startsWith('anchor_') ? keys.find(k => k !== 'scope' && criteria[k]!.includes(evidence[id.slice(7)] ?? 'NO_TEST_ANCHOR')) ?? 'scope'
        : id === 'technology' ? 'pytorch' : id === 'activity' ? active.pack : expected[id]?.[0] ?? 'unknown';
      return [id, { type: 'choice', choice, probabilities: Object.fromEntries(keys.map(k => [k, k === choice ? .98 : .02 / (keys.length - 1)])) }];
    }));
    return Response.json({ answers, usage: { inputTokens: 2000 }, providerMetadata: { gateway: { cost: '0' } } });
  };
  const uri = vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri, 'workflow.py');
  await vscode.workspace.fs.writeFile(uri, Buffer.from(active.source));
  const doc = await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(uri), 'python');
  let editor = await vscode.window.showTextDocument(doc);
  editor.selection = new vscode.Selection(active.cursorLine, 0, active.cursorLine, 0);
  const key = live ? process.env.AI_GATEWAY_API_KEY : 'synthetic-workflow-key'; assert.ok(key);
  await api.testing!.configure(key, true, live ? undefined : fetcher);
  await vscode.commands.executeCommand('workbench.action.closeSidebar');
  const observations: unknown[] = [];
  for (const name of names) {
    for (const state of ['concern', 'supported']) {
      active = workflowCases.find(c => c.name === `audit_${name}_${state}`)!;
      editor = await vscode.window.showTextDocument(doc);
      const changed = doc.getText() !== active.source;
      const priorFingerprint = api.getState().report?.fingerprint;
      const started = Date.now();
      if (changed) await editor.edit(b => b.replace(new vscode.Range(new vscode.Position(0, 0), doc.lineAt(doc.lineCount - 1).range.end), active.source));
      editor.selection = new vscode.Selection(active.cursorLine, 0, active.cursorLine, 0);
      editor.revealRange(editor.selection, vscode.TextEditorRevealType.AtTop);
      if (changed) {
        await delay(30);
        assert.ok(api.getState().stale || !api.getState().report, 'old source annotations become stale immediately');
        const staleLenses = await vscode.commands.executeCommand<vscode.CodeLens[]>('vscode.executeCodeLensProvider', uri);
        assert.equal(staleLenses?.filter(l => l.command?.command.startsWith('kernelLens.')).length ?? 0, 0);
      }
      const id = Object.keys(active.expect)[0]!;
      const matched = () => {
        const current = api.getState();
        return current.phase === 'ready' && !current.stale && (!changed || current.report?.fingerprint !== priorFingerprint)
          && current.report?.route.packs.includes(active.pack)
          && current.report.assessments.some(a => a.id === id && active.expect[id]!.includes(a.outcome));
      };
      const start = Date.now(); while (!matched() && api.getState().phase !== 'error' && Date.now() - start < (live ? 150000 : 12000)) await delay(40);
      if (!matched()) await fs.writeFile(path.join(root, `artifacts/workflow-failure-${Date.now()}.json`), JSON.stringify({
        name: active.name, liveGateway: live, extensionPath: extension.extensionPath, state: api.getState(), completed: observations,
      }, null, 2) + '\n');
      assert.ok(matched(), `${active.name}: ${JSON.stringify(api.getState())}`);
      const firstResultMs = Date.now() - started;
      while (api.getState().report?.detailStatus === 'locating' && Date.now() - start < (live ? 150000 : 12000)) await delay(40);
      assert.equal(doc.getText(), active.source); assert.equal(api.getState().advisorRequests, 0);
      const snapshot = api.getState();
      const lenses = (await vscode.commands.executeCommand<vscode.CodeLens[]>('vscode.executeCodeLensProvider', uri))?.filter(l => l.command?.command.startsWith('kernelLens.')) ?? [];
      if (state === 'concern') {
        const finding = snapshot.insights?.find(i => i.relatedIds.includes(id));
        assert.ok(finding, 'a model concern appears as an editor finding');
        assert.ok(lenses.length > 0, 'a clickable finding is available with the sidebar closed');
        if (!live) assert.ok(finding.anchored, `mock source selection is applied for ${id}`);
        const hovers = await vscode.commands.executeCommand<vscode.Hover[]>('vscode.executeHoverProvider', uri, new vscode.Position(finding.startLine - 1, doc.lineAt(finding.startLine - 1).firstNonWhitespaceCharacterIndex));
        const hoverText = hovers?.flatMap(h => h.contents).map(c => typeof c === 'string' ? c : 'value' in c ? c.value : '').join('\n').replaceAll('&nbsp;', ' ');
        if (snapshot.insights?.slice(0, 2).includes(finding)) assert.ok(hoverText?.includes('Next check'), `hover explains consequence and next check: ${hoverText}`);
        if (finding.anchored) assert.equal(doc.lineAt(finding.startLine - 1).text, finding.code);
      } else if (!live) assert.ok(snapshot.improvements?.some(i => i.id === id), `specific improvement is visible for ${id}`);
      observations.push({ name: active.name, report: snapshot.report, insights: snapshot.insights, improvements: snapshot.improvements,
        firstResultMs, sourceDetailMs: Date.now() - started, codeLenses: lenses.map(l => ({ line: l.range.start.line + 1, title: l.command?.title })), requests: snapshot.requests });
      await fs.writeFile(path.join(root, `artifacts/workflows-${live ? 'live' : 'mocked'}-host-0.5${filter ? '-focused' : ''}.json`), JSON.stringify({
        version: snapshot.version, liveGateway: live, extensionPath: extension.extensionPath,
        scenario: 'Edit the same file with sidebar closed. Verify CodeLens, hover, source locations, stale gating and specific improvements. No Analyze or generated review.', observations,
      }, null, 2) + '\n');
      if (state === 'concern' && ['optimizer_lifetime', 'decode_cache'].includes(name) && process.env.KERNEL_LENS_CAPTURE === '1') {
        await delay(450);
        execFileSync('python', ['-c', 'from PIL import ImageGrab; import sys; ImageGrab.grab().save(sys.argv[1])',
          path.join(root, `artifacts/${name}-${live ? 'live' : 'mocked'}-editor-0.5.png`)]);
        const finding = snapshot.insights?.find(i => i.relatedIds.includes(id));
        if (finding) {
          if (name === 'optimizer_lifetime') {
            const p = new vscode.Position(finding.startLine - 1, doc.lineAt(finding.startLine - 1).firstNonWhitespaceCharacterIndex);
            editor.selection = new vscode.Selection(p, p); await vscode.commands.executeCommand('editor.action.showHover'); await delay(450);
            execFileSync('python', ['-c', 'from PIL import ImageGrab; import sys; ImageGrab.grab().save(sys.argv[1])', path.join(root, `artifacts/optimizer-${live ? 'live' : 'mocked'}-hover-0.5.png`)]);
            await vscode.commands.executeCommand('editor.action.hideHover');
          }
          await vscode.commands.executeCommand('kernelLens.showFinding', finding.id, snapshot.report!.fingerprint); await delay(450);
          assert.equal(api.getState().selectedFinding, finding.id, 'clicking a finding selects its sidebar card');
          execFileSync('python', ['-c', 'from PIL import ImageGrab; import sys; ImageGrab.grab().save(sys.argv[1])', path.join(root, `artifacts/${name}-${live ? 'live' : 'mocked'}-detail-0.5.png`)]);
          await vscode.commands.executeCommand('workbench.action.closeSidebar');
        }
      }
    }
  }
  console.log(`Kernel Lens ${live ? 'LIVE Jev' : 'mocked Gateway'} workflow editor tests passed: ${observations.length} automatic training/inference profiles, corrections and workload changes.`);
}
