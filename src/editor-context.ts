import * as vscode from 'vscode';
import { hardwareId, type HardwareId } from './core/systems/hardware';
import { isExcluded } from './core/context';
import { BlockReader } from './core/blocks';
import { defaultAdvisorModel } from './core/advisor';
import type { CodeContext, PackSetting, Reference } from './live-types';

export interface AttachedReference { id: string; uri: string; name: string; startLine: number; endLine: number }
export interface LensSettings {
  domain: PackSetting; debounceMs: number; maxRequestsPerMinute: number; maxContextCharacters: number;
  minimumProbability: number; showInline: boolean; showDiagnostics: boolean;
  zeroDataRetention: boolean; useJev: boolean; excludedPaths: string[]; hardware: string; hardwareProfile: HardwareId;
  advisorMode: 'off' | 'onDemand' | 'auto'; advisorModel: string; advisorRequestsPerMinute: number;
}
export function settings(resource?: vscode.Uri): LensSettings {
  const config = vscode.workspace.getConfiguration('kernelLens', resource);
  const number = (name: string, fallback: number, min: number, max: number) => {
    const value = config.get<number>(name, fallback);
    return Number.isFinite(value) ? Math.max(min, Math.min(max, value)) : fallback;
  };
  const domain = config.get<PackSetting>('domain', 'auto');
  const excluded = config.get<unknown>('excludedPaths', []);
  const mode = config.get<string>('advisor.mode', 'onDemand');
  return {
    domain: ['auto', 'cuda', 'triton', 'pytorch', 'jax', 'distributed', 'serving', 'training', 'inference', 'data', 'general'].includes(domain) ? domain : 'auto',
    debounceMs: number('debounceMs', 700, 300, 10000),
    maxRequestsPerMinute: Math.floor(number('maxRequestsPerMinute', 30, 2, 120)),
    maxContextCharacters: Math.floor(number('maxContextCharacters', 16000, 4000, 40000)),
    minimumProbability: number('minimumProbability', .75, .6, .99),
    showInline: config.get('showInline', true), showDiagnostics: config.get('showDiagnostics', true),
    zeroDataRetention: config.get<boolean>('zeroDataRetention', false) === true,
    useJev: config.get<boolean>('useJev', true) === true,
    excludedPaths: Array.isArray(excluded) ? excluded.filter((item): item is string => typeof item === 'string') : [],
    hardwareProfile: hardwareId(config.get('hardwareProfile', 'b200')),
    hardware: config.get<string>('hardware', '').slice(0, 600),
    advisorMode: mode === 'auto' || mode === 'off' ? mode : 'onDemand',
    advisorModel: config.get<string>('advisor.model', defaultAdvisorModel).trim() || defaultAdvisorModel,
    advisorRequestsPerMinute: Math.floor(number('advisor.maxRequestsPerMinute', 2, 1, 6)),
  };
}
export function eligible(document: vscode.TextDocument, excluded: string[]): boolean {
  return ['file', 'vscode-remote', 'untitled'].includes(document.uri.scheme) && !isExcluded(document.uri.path, excluded)
    && (['cuda-cpp', 'cuda', 'cpp', 'c', 'python'].includes(document.languageId) || /\.(?:cu|cuh|cpp|cc|c|h|hpp|py|pyi)$/.test(document.fileName));
}
export function readBlock(reader: BlockReader, editor: vscode.TextEditor, config: LensSettings, intent: string, key: string, references: Reference[] = []): CodeContext {
  return reader.read({ file: vscode.workspace.asRelativePath(editor.document.uri, false), language: editor.document.languageId,
    source: editor.document.getText(), cursorLine: editor.selection.active.line, intent, hardware: config.hardware, hardwareProfile: config.hardwareProfile,
    references, maxCharacters: config.maxContextCharacters, secret: key, assessmentScope: 'function' });
}
export async function collectReferences(editor: vscode.TextEditor, context: CodeContext, attached: AttachedReference[], config: LensSettings): Promise<Reference[]> {
  const references: Reference[] = [];
  const read = async (uri: vscode.Uri, start: number, end: number, reason: string) => {
    if (!['file', 'vscode-remote'].includes(uri.scheme) || !vscode.workspace.getWorkspaceFolder(uri) || isExcluded(uri.path, config.excludedPaths)) return;
    try {
      const doc = await vscode.workspace.openTextDocument(uri);
      if (start >= doc.lineCount || doc.getText().length > 200_000) return;
      const last = Math.min(end, doc.lineCount - 1, start + 65);
      references.push({ name: vscode.workspace.asRelativePath(uri, false), startLine: start + 1, reason,
        code: doc.getText(new vscode.Range(start, 0, last, doc.lineAt(last).text.length)).split('\n').map((line, i) => `L${start + i + 1}: ${line}`).join('\n') });
    } catch { /* Missing references stay unknown, never block the current code. */ }
  };
  for (const ref of attached.slice(0, 3)) await read(vscode.Uri.parse(ref.uri), ref.startLine, ref.endLine, 'Attached context');
  // Reuse an installed language service if available. No compiler/server is installed or run by Lens.
  const source = editor.document.getText();
  const visibleLines = new Set([context.code, context.enclosing, context.preamble].join('\n').split('\n').flatMap(l => {
    const n = /^L(\d+):/.exec(l); return n ? [Number(n[1]) - 1] : [];
  }));
  const calls = [...source.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)]
    .filter(m => visibleLines.has(editor.document.positionAt(m.index!).line) && !['if', 'for', 'while', 'range', 'print', 'len'].includes(m[1]!))
    .sort((a, b) => context.unit.kind === 'FunctionDefinition' ? a.index! - b.index! : Math.abs(editor.document.positionAt(a.index!).line - editor.selection.active.line) - Math.abs(editor.document.positionAt(b.index!).line - editor.selection.active.line))
    .filter((m, index, all) => all.findIndex(n => n[1] === m[1]) === index).slice(0, 6);
  const lookedUp = await Promise.allSettled(calls.map(async match => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        Promise.resolve(vscode.commands.executeCommand<(vscode.Location | vscode.LocationLink)[]>('vscode.executeDefinitionProvider', editor.document.uri,
          editor.document.positionAt(match.index!))),
        new Promise<undefined>(resolve => { timer = setTimeout(() => resolve(undefined), 150); }),
      ]);
    } finally { if (timer) clearTimeout(timer); }
  }));
  for (const result of lookedUp) {
    if (result.status !== 'fulfilled') continue;
    for (const location of result.value ?? []) {
      if (references.length >= 3) break;
      const uri = 'uri' in location ? location.uri : location.targetUri;
      const range = 'range' in location ? location.range : location.targetRange;
      if (uri.toString() === editor.document.uri.toString() || !vscode.workspace.getWorkspaceFolder(uri)
        || references.some(ref => ref.name === vscode.workspace.asRelativePath(uri, false))) continue;
      await read(uri, range.start.line, Math.max(range.end.line, range.start.line + 40), 'Referenced definition');
    }
  }
  // Resolve explicit workspace Python imports even without a language server.
  // No repository crawl or execution: at most three named modules, bounded reads.
  if (editor.document.languageId === 'python' && references.length < 3) {
    const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
    if (!folder) return references;
    const visible = [context.code, context.enclosing, context.preamble].join('\n').split('\n')
      .filter(line => !/^L\d+:\s*(?:from\s|import\s)/.test(line)).join('\n');
    const imports = [...source.matchAll(/^\s*from\s+([.\w]+)\s+import\s+([\w]+)(?:\s+as\s+(\w+))?/gm)]
      .filter(m => new RegExp(`\\b${m[3] ?? m[2]}\\b`).test(visible)).slice(0, 3);
    for (const match of imports) {
      if (references.length >= 3) break;
      const module = match[1]!, symbol = match[2]!;
      const dots = /^\.+/.exec(module)?.[0].length ?? 0;
      const relative = module.slice(dots).replaceAll('.', '/');
      const parent = vscode.Uri.joinPath(editor.document.uri, '..', ...Array(Math.max(0, dots - 1)).fill('..'));
      const bases = dots ? [parent] : [vscode.Uri.joinPath(editor.document.uri, '..'), folder.uri];
      let found = false;
      for (const base of bases) {
        for (const suffix of ['.py', '/__init__.py']) {
          const uri = vscode.Uri.joinPath(base, relative + suffix);
          if (!vscode.workspace.getWorkspaceFolder(uri) || isExcluded(uri.path, config.excludedPaths) || references.some(r => r.name === vscode.workspace.asRelativePath(uri, false))) continue;
          try {
            const stat = await vscode.workspace.fs.stat(uri); if (stat.size > 200_000) continue;
            const doc = await vscode.workspace.openTextDocument(uri);
            const lines = doc.getText().split('\n');
            const start = lines.findIndex(l => new RegExp(`^\\s*(?:(?:async\\s+)?def\\s+|class\\s+)?${symbol}\\s*(?:[=(:]|$)`).test(l));
            if (start < 0) continue;
            await read(uri, Math.max(0, start - 2), start + 60, `Imported definition · ${symbol}`); found = true; break;
          } catch { /* Unavailable local modules remain explicit context gaps. */ }
        }
        if (found) break;
      }
    }
  }
  return references;
}
