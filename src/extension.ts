import * as vscode from 'vscode';
import { hardwareId } from './core/systems/hardware';
import { RevisionGuard, RequestBudget, ReportCache } from './core/scheduling';
import { GatewayError } from './core/gateway';
import { testGatewayConnection } from './core/connection';
import { isExcluded } from './core/context';
import { BlockReader, contextHash, withRecentEdit } from './core/blocks';
import { changesSince, concernSignature } from './core/assessment';
import { analyzeSystems, systemsVersion } from './core/systems/analyze';
import { assessSystems, systemsPayload, systemsReport } from './core/systems/evaluate';
import { parseEnrichment } from './core/systems/enrichment';
import { sourceRevision } from './core/systems/source';
import type { Enrichment } from './core/systems/types';
import { review, handoff } from './core/advisor';
import { metrics } from './core/rubrics';
import { insightPresentation, insightIcon, insightLabel, performanceAssessments, kernelSummary } from './core/presentation';
import { guidance } from './core/guidance';
import { collectReferences, eligible, readBlock, settings, type AttachedReference } from './editor-context';
import { LensView } from './webview';
import type { Advice, AssessmentReport, CodeContext, Insight, LiveState, Usage } from './live-types';

const secretName = 'kernelLens.aiGatewayKey';
export interface KernelLensApi {
  getState(): LiveState;
  testing?: { configure(key: string, consent: boolean, fetcher?: typeof fetch): Promise<void> };
}
export async function activate(context: vscode.ExtensionContext): Promise<KernelLensApi> {
  const controller = new LensController(context);
  await controller.initialize(); context.subscriptions.push(controller);
  return { getState: () => structuredClone(controller.state),
    ...(context.extensionMode === vscode.ExtensionMode.Test ? { testing: {
      configure: async (key: string, consent: boolean, fetcher?: typeof fetch) => controller.configureTest(key, consent, fetcher),
    } } : {}) };
}

class LensController implements vscode.Disposable {
  state: LiveState;
  private view: LensView;
  private key = '';
  private refs: AttachedReference[];
  private reader = new BlockReader();
  private guard = new RevisionGuard();
  private budget = new RequestBudget(200);
  private advisorBudget = new RequestBudget(15000);
  private reports = new ReportCache<AssessmentReport>();
  private advice = new ReportCache<Advice>();
  private priorReports = new Map<string, AssessmentReport>();
  private readyContexts = new Map<string, CodeContext>();
  private enrichment = new Map<string, { revision: string; facts: Enrichment }>();
  private improvementTimer?: ReturnType<typeof setTimeout>;
  private scheduledAt = Date.now();
  private codeLenses: vscode.CodeLens[] = [];
  private findingHovers: { range: vscode.Range; content: vscode.MarkdownString }[] = [];
  private lensEvents = new vscode.EventEmitter<void>();
  private findingDecorations = new Map<Insight['kind'], vscode.TextEditorDecorationType>();
  private advised = new Map<string, string>();
  private timer?: ReturnType<typeof setTimeout>;
  private editor?: vscode.TextEditor;
  private lastContext?: CodeContext;
  private lastVersion = -1;
  private authBlocked = false;
  private advisorController?: AbortController;
  private advisorTimer?: ReturnType<typeof setTimeout>;
  private fetcher: typeof fetch = fetch;
  private connectionTest?: AbortController;
  private output = vscode.window.createOutputChannel('Kernel Lens · Gateway');
  private diagnostics = vscode.languages.createDiagnosticCollection('Kernel Lens');
  private decoration = vscode.window.createTextEditorDecorationType({
    after: { color: new vscode.ThemeColor('descriptionForeground'), margin: '0 0 0 2em' },
    rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed,
  });
  private status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 10);
  private subscriptions: vscode.Disposable[] = [];
  constructor(private context: vscode.ExtensionContext) {
    for (const [kind, color] of Object.entries({ correctness: 'editorWarning.foreground', performance: 'editorInfo.foreground', tentative: 'editorHint.foreground', improved: 'testing.iconPassed' })) {
      this.findingDecorations.set(kind as Insight['kind'], vscode.window.createTextEditorDecorationType({
        gutterIconPath: vscode.Uri.joinPath(context.extensionUri, 'media', `gutter-${kind}.svg`), gutterIconSize: 'contain',
        overviewRulerColor: new vscode.ThemeColor(color), overviewRulerLane: vscode.OverviewRulerLane.Right,
        after: { color: new vscode.ThemeColor(color), margin: '0 0 0 1.5em', fontWeight: '500' },
        rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed,
      }));
    }
    this.refs = context.workspaceState.get<AttachedReference[]>('references', []).slice(0, 3);
    const config = settings();
    this.state = { version: String(context.extension.packageJSON.version), phase: 'disabled', message: 'Connect Jev to see live assessments.',
      enabled: false, configured: false, consented: context.workspaceState.get('consented', false), domain: config.domain,
      intent: context.workspaceState.get('intent', ''), references: [], stale: false, requests: 0, advisorRequests: 0, totalTokens: 0, totalCost: 0,
      advisor: { mode: config.advisorMode, model: config.advisorModel, phase: 'idle', message: 'Reviews run when requested.' } };
    this.view = new LensView(context, () => this.state, message => this.handleMessage(message));
  }
  async initialize(): Promise<void> {
    this.key = (await this.context.secrets.get(secretName) ?? process.env.AI_GATEWAY_API_KEY ?? '').trim();
    this.state.configured = !!this.key;
    this.state.enabled = vscode.workspace.isTrusted && this.context.workspaceState.get('enabled', true);
    this.state.phase = this.state.enabled ? 'waiting' : 'paused';
    this.syncRefs(); this.status.command = 'kernelLens.open'; this.status.show();
    const command = (name: string, action: () => unknown) => vscode.commands.registerCommand(`kernelLens.${name}`, action);
    this.subscriptions.push(
      vscode.window.registerWebviewViewProvider('kernelLens.insights', this.view),
      command('open', () => vscode.commands.executeCommand('kernelLens.insights.focus')),
      command('enable', () => this.enable()), command('enableJev', () => this.enable()), command('pause', () => this.pause()),
      command('disableJev', () => this.pause()),
      command('analyze', () => { this.authBlocked = false; if (!this.state.enabled || !this.key || !this.state.consented) return this.enable(); this.schedule(true); }),
      command('setKey', () => this.setKey()), command('clearKey', () => this.clearKey()), command('testConnection', () => this.testConnection()),
      command('setIntent', () => this.setIntent()), command('attach', () => this.attach()),
      command('review', () => this.runReview(false)), command('copyHandoff', () => this.copyHandoff()),
      command('importEvidence', () => this.importEvidence()),
      vscode.commands.registerCommand('kernelLens.showFinding', async (id: string, fingerprint?: string) => {
        if (this.state.stale || (fingerprint && fingerprint !== this.state.report?.fingerprint) || !this.state.insights?.some(i => i.id === id)) return;
        this.state.selectedFinding = id; await vscode.commands.executeCommand('kernelLens.insights.focus'); this.view.update();
      }),
      vscode.languages.registerCodeLensProvider(['python', 'cpp', 'c', 'cuda', 'cuda-cpp'].map(language => ({ language })), {
        onDidChangeCodeLenses: this.lensEvents.event,
        provideCodeLenses: doc => this.editor?.document === doc && !this.state.stale && this.state.enabled && settings(doc.uri).showInline ? this.codeLenses : [],
      }),
      vscode.languages.registerHoverProvider(['python', 'cpp', 'c', 'cuda', 'cuda-cpp'].map(language => ({ language })), {
        provideHover: (doc, position) => {
          if (this.editor?.document !== doc || this.state.stale || !this.state.enabled) return;
          const matched = this.findingHovers.filter(h => h.range.contains(position));
          if (matched.length) return new vscode.Hover(matched.map(h => h.content), matched[0]!.range);
        },
      }),
      vscode.window.onDidChangeActiveTextEditor(editor => { if (editor) this.schedule(); }),
      vscode.workspace.onDidChangeTextDocument(event => {
        if (event.contentChanges.length && (event.document === this.editor?.document || vscode.workspace.getWorkspaceFolder(event.document.uri))) this.schedule(false, true);
      }),
      vscode.workspace.onDidCloseTextDocument(document => {
        this.reader.close(vscode.workspace.asRelativePath(document.uri, false)); this.diagnostics.delete(document.uri);
        if (this.editor?.document === document) { this.editor = undefined; this.lastContext = undefined; this.invalidate(true); this.schedule(); }
      }),
      vscode.window.onDidChangeTextEditorSelection(event => { if (event.textEditor === vscode.window.activeTextEditor) this.schedule(); }),
      vscode.workspace.onDidChangeConfiguration(event => {
        if (event.affectsConfiguration('kernelLens')) { this.authBlocked = false; this.clearCaches(); this.schedule(false, true); }
      }),
      this.context.secrets.onDidChange(async event => {
        if (event.key !== secretName) return;
        this.key = (await this.context.secrets.get(secretName) ?? process.env.AI_GATEWAY_API_KEY ?? '').trim();
        this.state.configured = !!this.key; this.authBlocked = false; this.clearCaches(); this.schedule(false, true);
      }),
    );
    this.publish(); this.schedule();
  }
  async configureTest(key: string, consent: boolean, fetcher?: typeof fetch): Promise<void> {
    this.key = key; this.state.configured = !!key; this.state.consented = consent; this.fetcher = fetcher ?? fetch;
    this.authBlocked = false; this.clearCaches(); this.schedule(true, true);
  }
  private clearCaches(): void { this.reports.clear(); this.advice.clear(); this.advised.clear(); }
  private publish(): void {
    const config = settings(this.editor?.document.uri);
    this.state.hardwareProfile = config.hardwareProfile; this.state.domain = config.domain; this.state.advisor.mode = config.advisorMode; this.state.advisor.model = config.advisorModel;
    const busy = ['classifying', 'assessing'].includes(this.state.phase);
    const concerns = this.state.stale ? 0 : this.state.insights?.filter(i => i.kind !== 'tentative').length ?? 0;
    this.status.text = `$(${busy ? 'loading~spin' : this.state.phase === 'error' ? 'warning' : this.state.enabled ? 'pulse' : 'debug-pause'}) Code insights${busy ? ' · Checking' : this.state.phase === 'error' ? ' · Offline' : this.state.stale ? ' · Updating' : concerns ? ` · ${concerns}` : this.state.improvements?.length ? ' · Improved' : ''}`;
    this.status.tooltip = `Kernel Lens — ${this.state.message}`;
    this.showActivity(); this.view.update();
  }
  private showActivity(): void {
    const editor = this.editor;
    if (!editor || editor.document.isClosed || !this.state.enabled || !settings(editor.document.uri).showInline
      || !eligible(editor.document, settings(editor.document.uri).excludedPaths)
      || this.state.phase === 'ready' || this.state.phase === 'paused' || this.state.phase === 'unsupported') return;
    const label = this.state.phase === 'classifying' || this.state.phase === 'assessing' ? 'Checking…'
      : this.state.phase === 'waiting' ? this.state.unit?.ready === false ? 'Finish this block' : 'Waiting for typing to pause…'
      : this.state.phase === 'limited' ? this.state.message
      : this.state.phase === 'error' ? 'Could not update — open Kernel Lens'
      : 'Connect Jev to start';
    const line = Math.max(0, Math.min(editor.selection.active.line, editor.document.lineCount - 1));
    editor.setDecorations(this.decoration, [{ range: editor.document.lineAt(line).range,
      renderOptions: { after: { contentText: `◉ Code insights · ${label}` } },
      hoverMessage: new vscode.MarkdownString().appendText(this.state.message) }]);
  }
  private clearAnnotations(): void {
    this.diagnostics.clear(); this.codeLenses = []; this.findingHovers = []; this.lensEvents.fire();
    for (const editor of vscode.window.visibleTextEditors) {
      editor.setDecorations(this.decoration, []);
      for (const decoration of this.findingDecorations.values()) editor.setDecorations(decoration, []);
    }
  }
  private invalidate(clear: boolean): number {
    if (this.timer) clearTimeout(this.timer); this.timer = undefined;
    if (this.advisorTimer) clearTimeout(this.advisorTimer); this.advisorTimer = undefined;
    if (this.improvementTimer) clearTimeout(this.improvementTimer); this.improvementTimer = undefined;
    this.advisorController?.abort(); this.advisorController = undefined;
    this.clearAnnotations();
    this.state.advisor.result = undefined; this.state.advisor.phase = 'idle';
    this.state.improvements = []; this.state.selectedFinding = undefined;
    if (clear) { this.state.report = undefined; this.state.context = undefined; this.state.unit = undefined; this.state.insights = []; }
    this.state.stale = !clear && !!this.state.report;
    return this.guard.invalidate();
  }
  private schedule(manual = false, force = false): void {
    const editor = vscode.window.activeTextEditor ?? this.editor;
    if (!manual && !force && editor === this.editor && editor?.document.version === this.lastVersion && this.lastContext && !this.lastContext.truncated
      && editor.selection.active.line + 1 >= this.lastContext.unit.startLine && editor.selection.active.line + 1 <= this.lastContext.unit.endLine) return;
    const changedFile = editor?.document.uri.toString() !== this.editor?.document.uri.toString();
    this.scheduledAt = Date.now();
    const ticket = this.invalidate(changedFile); this.editor = editor;
    if (changedFile) this.lastContext = undefined;
    if (!this.state.enabled || !vscode.workspace.isTrusted) { this.publish(); return; }
    if (!editor || editor.document.isClosed || !eligible(editor.document, settings(editor.document.uri).excludedPaths)) {
      this.state.phase = 'unsupported'; this.state.message = 'Open a CUDA, Triton, or Python source file.';
      this.state.file = undefined; this.state.report = undefined; this.publish(); return;
    }
    this.state.file = vscode.workspace.asRelativePath(editor.document.uri, false);
    this.state.phase = 'waiting'; this.state.message = 'Waiting for a complete block and a pause in typing.'; this.publish();
    this.timer = setTimeout(() => { void this.prepare(editor, ticket, manual).catch(error => {
      if (this.guard.current(ticket)) { this.state.phase = 'error'; this.state.message = error instanceof Error ? error.message : 'Could not read this block.'; this.publish(); }
    }); }, manual ? 0 : 140);
  }
  private async prepare(editor: vscode.TextEditor, ticket: number, manual: boolean): Promise<void> {
    if (!this.guard.current(ticket)) return;
    const config = settings(editor.document.uri), version = editor.document.version;
    const current = () => this.guard.current(ticket) && this.state.enabled && vscode.workspace.isTrusted && !editor.document.isClosed
      && editor.document.version === version && this.editor?.document.uri.toString() === editor.document.uri.toString();
    let block = readBlock(this.reader, editor, config, this.state.intent, this.key);
    this.state.unit = block.unit; this.lastContext = block; this.lastVersion = version;
    this.showSyntax(editor, block);
    if (!block.unit.ready) { this.state.message = 'Finish this block to get its Jev assessment.'; this.publish(); return; }
    if (!this.key || !this.state.consented || !config.useJev) {
      this.state.phase = 'disabled'; this.state.message = !this.key ? 'Set your AI Gateway key to start live Jev assessments.'
        : !this.state.consented ? 'Enable Jev for this workspace to start live assessments.' : 'Jev is paused in settings.';
      this.state.report = undefined; this.state.stale = false; this.publish(); return;
    }
    if (this.authBlocked) { this.state.phase = 'error'; this.state.message = 'Gateway access needs attention. Test the connection or retry Analyze.'; this.publish(); return; }
    const refs = await collectReferences(editor, block, this.refs, config);
    if (!current()) return;
    if (refs.length) block = readBlock(this.reader, editor, config, this.state.intent, this.key, refs);
    const supplied = this.enrichment.get(this.scopeKey(block));
    if (supplied && supplied.revision === sourceRevision(block)) block = { ...block, enrichment: supplied.facts };
    else if (supplied) this.enrichment.delete(this.scopeKey(block));
    block = withRecentEdit(this.readyContexts.get(this.scopeKey(block)), block, config.maxContextCharacters);
    this.lastContext = block;
    this.state.context = { characters: block.characters, truncated: block.truncated, references: block.references.map(r => ({ name: r.name, reason: r.reason })) };
    const cacheKey = `systems-${systemsVersion}:${config.domain}:${config.minimumProbability}:${config.zeroDataRetention}:${contextHash(block)}`;
    const cached = this.reports.get(cacheKey);
    if (cached) {
      this.accept(cached, block, editor, true);
      return;
    }
    this.timer = setTimeout(() => { void this.runAssessment(block, cacheKey, editor, ticket, current, 0); }, manual ? 0 : Math.max(0, config.debounceMs - 140));
    this.publish();
  }
  private async runAssessment(block: CodeContext, cacheKey: string, editor: vscode.TextEditor, ticket: number, current: () => boolean, retry: number): Promise<void> {
    if (!current()) return;
    const config = settings(editor.document.uri);
    let analysis: ReturnType<typeof analyzeSystems>;
    try { analysis = analyzeSystems(block, config.domain); } catch {
      this.state.phase = 'error'; this.state.message = 'The static analysis could not interpret this block. Edit or retry Analyze.'; this.publish(); return;
    }
    const callsJev = Object.keys(systemsPayload(block, analysis, config).questions).length > 0;
    const delay = callsJev ? this.budget.delay(config.maxRequestsPerMinute) : 0;
    if (delay > 0) {
      this.state.phase = 'limited'; this.state.message = `Next Jev assessment in ${Math.ceil(delay / 1000)}s.`; this.publish();
      this.timer = setTimeout(() => { void this.runAssessment(block, cacheKey, editor, ticket, current, retry); }, delay + 10); return;
    }
    const signal = this.guard.signal(ticket);
    try {
      this.state.phase = 'assessing'; this.state.message = callsJev ? 'Jev is assessing the observed execution structures.' : 'Checking the visible execution structures.'; this.publish();
      if (callsJev) { this.budget.record(); this.state.requests++; }
      const report = await assessSystems(block, this.key, signal, config.minimumProbability, this.fetcher, config, config.domain, analysis);
      this.usage(report); if (!current()) return;
      this.reports.set(cacheKey, report); this.accept(report, block, editor, false);
      if (config.advisorMode === 'auto' && concernSignature(report)) void this.runReview(true);
    } catch (error) {
      if (!current() || signal.aborted) return;
      if (error instanceof GatewayError && (error.status === 429 || error.status >= 500 && retry < 1)) {
        const backoff = error.status === 429 ? error.retryAfterMs : 1500;
        this.budget.cooldown(backoff); this.state.phase = 'limited'; this.state.message = `Gateway is busy; retrying in ${Math.ceil(backoff / 1000)}s.`;
        this.timer = setTimeout(() => { void this.runAssessment(block, cacheKey, editor, ticket, current, retry + 1); }, backoff + 20);
      } else {
        this.authBlocked = error instanceof GatewayError && [401, 402, 403].includes(error.status);
        this.state.phase = 'error'; this.state.message = error instanceof GatewayError ? error.message : 'Jev could not reach AI Gateway. Retry Analyze.';
        if(error instanceof GatewayError && error.status>=500) {
          // Keep direct observations useful during provider outages. Do not
          // cache a failed prediction, compare it as an improvement, or present
          // unassessed model hypotheses as if Jev endorsed them.
          const direct={...analysis,candidates:analysis.candidates.filter(c=>!c.modelMayAssess)};
          const report=systemsReport(block,direct,{}, {latencyMs:0},config.minimumProbability,config.domain);
          this.state.report=report;this.state.stale=false;
          Object.assign(this.state,insightPresentation(report));this.state.improvements=[];
          this.state.message+=' Showing source observations only; Analyze retries Jev predictions.';
          this.annotate(editor,block,report);
        }
      }
      this.publish();
    }
  }
  private usage(usage: Usage): void { this.state.totalTokens += (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0); this.state.totalCost += usage.cost ?? 0; }
  private scopeKey(block: CodeContext): string { return `${block.file}:${block.unit.identity ?? `${block.unit.name}:${block.unit.startLine}`}`; }
  private accept(report: AssessmentReport, block: CodeContext, editor: vscode.TextEditor, cached: boolean): void {
    const key = this.scopeKey(block), prior = this.priorReports.get(key);
    this.state.report = { ...report, changes: changesSince(prior, report), visibleAfterMs: Date.now() - this.scheduledAt };
    Object.assign(this.state, insightPresentation(this.state.report, prior, block));
    this.priorReports.set(key, report); this.readyContexts.set(key, block);
    if (!concernSignature(report)) this.advised.delete(key);
    while (this.priorReports.size > 30) this.priorReports.delete(this.priorReports.keys().next().value!);
    while (this.readyContexts.size > 30) this.readyContexts.delete(this.readyContexts.keys().next().value!);
    this.state.phase = 'ready'; this.state.stale = false; this.state.message = cached ? 'Insights are current. Keep writing.' : 'Insights updated. Keep writing.';
    const config = settings(editor.document.uri);
    const cachedAdvice = this.advice.get(`${report.fingerprint}:${config.advisorModel}`);
    if (cachedAdvice) this.state.advisor = { ...this.state.advisor, phase: 'ready', result: cachedAdvice, message: 'Review reused for unchanged context.' };
    else this.state.advisor.message = config.advisorMode === 'auto' ? 'New strong concerns can trigger a review.' : `Review this block with ${config.advisorModel.includes('astra') ? 'Astra' : config.advisorModel} when you need a concrete next step.`;
    this.annotate(editor, block, report); this.publish();
    if (this.state.improvements?.length) this.improvementTimer = setTimeout(() => {
      if (this.state.report?.fingerprint !== report.fingerprint || this.state.stale) return;
      this.state.improvements = []; this.annotate(editor, block, this.state.report); this.publish();
    }, 8000);
  }
  private showSyntax(editor: vscode.TextEditor, block: CodeContext): void {
    if (!settings(editor.document.uri).showDiagnostics) return;
    this.diagnostics.set(editor.document.uri, block.unit.syntax.map(issue => {
      const d = new vscode.Diagnostic(new vscode.Range(editor.document.positionAt(issue.from), editor.document.positionAt(issue.to)), issue.message, vscode.DiagnosticSeverity.Information);
      d.source = 'Kernel Lens · syntax readiness'; d.code = 'syntax'; return d;
    }));
  }
  private annotate(editor: vscode.TextEditor, block: CodeContext, report: AssessmentReport): void {
    this.clearAnnotations(); this.showSyntax(editor, block);
    const config = settings(editor.document.uri);
    const findings = this.state.insights ?? [], shown = [...findings.slice(0, 8), ...(this.state.improvements ?? []).slice(0, 1)];
    const kernelLabel=kernelSummary(report);
    if(kernelLabel&&config.showInline) {
      const line=Math.max(0,Math.min(editor.document.lineCount-1,block.unit.startLine-1));
      this.codeLenses.push(new vscode.CodeLens(new vscode.Range(line,0,line,0),{title:`◉ Code insights · ${kernelLabel}`,command:'kernelLens.open'}));
    }
    const diagnostics = [...this.diagnostics.get(editor.document.uri) ?? []];
    const decorations = new Map<Insight['kind'], vscode.DecorationOptions[]>();
    const decoratedLines = new Set<number>();
    for (const finding of shown) {
      const line = Math.max(0, Math.min(editor.document.lineCount - 1, finding.startLine - 1));
      const sourceLine = editor.document.lineAt(line);
      const range = new vscode.Range(line, sourceLine.firstNonWhitespaceCharacterIndex, line, sourceLine.text.length);
      const hover = new vscode.MarkdownString();
      const f = finding.finding;
      hover.appendText(`${insightLabel(finding)} · ${finding.title}\n\n${finding.consequence}${finding.nextCheck ? `\n\nNext check: ${finding.nextCheck}` : ''}\n\n${f ? `Structural confidence: ${f.confidence}. Evidence: level ${f.evidence_level} (${f.basis}).\nRequires: ${f.requires.join(', ')}.\n${f.runtime_impact_measured ? 'Supplied runtime measurements.' : 'Runtime impact is unmeasured.'}` : `${finding.anchored ? 'Source location selected by Jev.' : 'Block-level prediction; the precise source location is not isolated.'} Runtime impact is unmeasured.`}`);
      hover.value = hover.value.replaceAll('&nbsp;', ' ');
      if (finding.kind !== 'improved') {
        const args = encodeURIComponent(JSON.stringify([finding.id, report.fingerprint]));
        hover.appendMarkdown(`\n\n[Open finding](command:kernelLens.showFinding?${args}) · [Ask Astra](command:kernelLens.review)`);
        hover.isTrusted = { enabledCommands: ['kernelLens.showFinding', 'kernelLens.review'] };
      }
      this.findingHovers.push({ range, content: hover });
      if (config.showInline && !decoratedLines.has(line)) {
        decoratedLines.add(line);
        const others = shown.filter(i => i !== finding && i.startLine === finding.startLine).length;
        const suffix = others ? ` +${others}` : '';
        const list = decorations.get(finding.kind) ?? [];
        list.push({ range, renderOptions: { after: { contentText: `${insightIcon(finding)} ${finding.kind === 'tentative' ? 'Possible: ' : ''}${finding.title.length > 44 ? finding.title.slice(0, 43) + '…' : finding.title}${suffix}` } } });
        decorations.set(finding.kind, list);
        this.codeLenses.push(new vscode.CodeLens(range, { title: `${insightIcon(finding)} Code insights · ${finding.kind === 'improved' ? 'Improved: ' : finding.change === 'new' ? 'New: ' : ''}${finding.title}${suffix}`, command: finding.kind === 'improved' ? 'kernelLens.open' : 'kernelLens.showFinding', arguments: [finding.id, report.fingerprint] }));
      }
      if (config.showDiagnostics && finding.kind !== 'tentative' && finding.kind !== 'improved') {
        const d = new vscode.Diagnostic(range, `${insightLabel(finding)}: ${finding.title}. ${finding.nextCheck}${f && !f.runtime_impact_measured ? ' Runtime impact is unmeasured.' : ''}`, finding.kind === 'correctness' ? vscode.DiagnosticSeverity.Warning : vscode.DiagnosticSeverity.Information);
        d.source = f && f.model_choice === undefined ? 'Kernel Lens · static evidence' : 'Kernel Lens · Jev'; d.code = finding.id; diagnostics.push(d);
      }
    }
    for (const [kind, list] of decorations) editor.setDecorations(this.findingDecorations.get(kind)!, list);
    if (config.showDiagnostics) this.diagnostics.set(editor.document.uri, diagnostics);
    if (!shown.length && config.showInline) {
      const performance = performanceAssessments(report);
      const supported = performance.find(a => a.signal === 'supported' && !a.tentative);
      const unknown = performance.find(a => a.signal === 'unknown');
      const line = Math.max(0, Math.min(editor.selection.active.line, editor.document.lineCount - 1));
      const label = kernelLabel ?? (supported ? supported.bucket : unknown ? 'More evidence needed' : 'No supported structural finding');
      const f = report.findings?.find(f => f.id === (supported ?? unknown)?.id);
      const hover = new vscode.MarkdownString().appendText(f ? `${f.evidence.explanation}\n\nNext check: ${f.next_check}\nRequires: ${f.requires.join(', ')}\n${f.runtime_impact_measured ? 'Runtime evidence supplied for this revision.' : 'Runtime impact is unmeasured.'}` : supported ? `Observed practice: ${supported.bucket}.` : unknown ? guidance[unknown.id]?.[2] ?? 'Available evidence is insufficient.' : 'No supported structural finding in the recognized operations. Unsupported calls and omitted code remain unassessed.');
      editor.setDecorations(this.decoration, [{ range: editor.document.lineAt(line).range, renderOptions: { after: { contentText: `◉ Code insights · ${label}` } }, hoverMessage: hover }]);
    }
    this.lensEvents.fire();
  }
  private async runReview(automatic: boolean): Promise<void> {
    const block = this.lastContext, report = this.state.report, editor = this.editor;
    const config = settings(editor?.document.uri);
    if (!block || !report || !editor || !this.state.enabled || this.state.stale || this.state.phase !== 'ready' || !this.key || !this.state.consented
      || config.advisorMode === 'off' || this.state.advisor.phase === 'running') return;
    const signature = concernSignature(report), scope = this.scopeKey(block);
    if (automatic && (!signature || this.advised.get(scope) === signature)) return;
    const cacheKey = `${report.fingerprint}:${config.advisorModel}`, cached = this.advice.get(cacheKey);
    if (cached) { this.state.advisor = { ...this.state.advisor, phase: 'ready', result: cached, message: 'Review reused for unchanged context.' }; this.publish(); return; }
    const delay = this.advisorBudget.delay(config.advisorRequestsPerMinute);
    if (delay > 0) {
      this.state.advisor.message = `Review budget: available in ${Math.ceil(delay / 1000)}s. Jev continues independently.`;
      if (automatic) {
        if (this.advisorTimer) clearTimeout(this.advisorTimer);
        this.advisorTimer = setTimeout(() => { if (this.state.report?.fingerprint === report.fingerprint && !this.state.stale) void this.runReview(true); }, delay + 10);
      }
      this.publish(); return;
    }
    const controller = new AbortController(), fingerprint = report.fingerprint;
    this.advisorController = controller; this.advisorBudget.record(); this.state.advisorRequests++;
    this.state.advisor.phase = 'running'; this.state.advisor.message = `${config.advisorModel.includes('astra') ? 'Astra' : config.advisorModel} is checking the evidence and preparing a short review.`; this.publish();
    try {
      const result = await review(block, report, this.key, controller.signal, config.advisorModel, this.fetcher, config);
      this.usage(result);
      if (controller.signal.aborted || this.state.stale || this.state.report?.fingerprint !== fingerprint) return;
      this.advice.set(cacheKey, result); this.advised.set(scope, signature);
      while (this.advised.size > 30) this.advised.delete(this.advised.keys().next().value!);
      this.state.advisor = { ...this.state.advisor, phase: 'ready', result, message: `Review completed in ${(result.latencyMs / 1000).toFixed(1)}s.` };
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof GatewayError && error.status === 429) this.advisorBudget.cooldown(error.retryAfterMs);
      this.state.advisor.phase = 'error'; this.state.advisor.message = error instanceof GatewayError ? error.message : 'The review could not reach AI Gateway. Jev assessments remain available.';
    } finally { if (this.advisorController === controller) this.advisorController = undefined; this.publish(); }
  }
  private async copyHandoff(): Promise<void> {
    if (!this.lastContext || !this.state.report || this.state.stale) return;
    await vscode.env.clipboard.writeText(handoff(this.lastContext, this.state.report, this.state.advisor.result));
    void vscode.window.showInformationMessage('Code, context and Jev assessments copied. Paste into Astra or your coding agent.');
  }
  private async importEvidence(): Promise<void> {
    const block = this.lastContext;
    if (!block || this.state.stale || !block.unit.ready) { void vscode.window.showInformationMessage('Open a complete source block before attaching performance evidence.'); return; }
    const revision = sourceRevision(block), scope = this.scopeKey(block);
    const files = await vscode.window.showOpenDialog({ title: 'Attach shapes, compiler, hardware or runtime evidence to this source revision', canSelectMany: false, filters: { 'Performance evidence': ['json'] } });
    if (!files?.[0]) return;
    try {
      const stat = await vscode.workspace.fs.stat(files[0]);
      if (stat.size > 32_000) throw new Error('The evidence file must be smaller than 32 KB.');
      const facts = parseEnrichment(JSON.parse(Buffer.from(await vscode.workspace.fs.readFile(files[0])).toString('utf8')));
      if (this.state.stale || !this.lastContext || sourceRevision(this.lastContext) !== revision) throw new Error('The source changed while selecting evidence. Attach it to the current revision again.');
      this.enrichment.set(scope, { revision, facts });
      while (this.enrichment.size > 16) this.enrichment.delete(this.enrichment.keys().next().value!);
      this.clearCaches(); this.schedule(true, true);
      void vscode.window.showInformationMessage('Evidence attached to this block. It expires when the source context changes.');
    } catch (error) { void vscode.window.showErrorMessage(error instanceof SyntaxError ? 'The evidence file is not valid JSON.' : error instanceof Error ? error.message : 'Could not read the evidence file.'); }
  }
  private async enable(): Promise<void> {
    if (!vscode.workspace.isTrusted) return;
    if (!this.state.consented) {
      const answer = await vscode.window.showInformationMessage('Kernel Lens sends focused code, relevant callers/helpers, workspace definitions and optional context to Jev through AI Gateway. Astra reviews use the same context when requested or when automatic reviews are enabled. Enable for this workspace?', { modal: true }, 'Enable for This Workspace');
      if (answer !== 'Enable for This Workspace') return;
      this.state.consented = true; await this.context.workspaceState.update('consented', true);
    }
    if (!this.key && !(await this.setKey())) return;
    if (!settings().useJev) await vscode.workspace.getConfiguration('kernelLens').update('useJev', true, this.settingTarget());
    this.state.enabled = true; this.authBlocked = false; await this.context.workspaceState.update('enabled', true); this.schedule(true, true);
  }
  private async pause(): Promise<void> {
    this.invalidate(true); this.state.enabled = false; this.state.phase = 'paused'; this.state.message = 'Paused. No code is being sent.';
    await this.context.workspaceState.update('enabled', false); this.publish();
  }
  private async setKey(): Promise<boolean> {
    const value = await vscode.window.showInputBox({ title: 'Kernel Lens · AI Gateway key', password: true, ignoreFocusOut: true,
      prompt: 'Saved in VS Code secret storage. Your existing Jev Gateway key works here.',
      validateInput: text => text.trim().length < 12 || /\s/.test(text.trim()) ? 'Enter the full key without spaces.' : undefined });
    if (!value?.trim()) return false;
    this.key = value.trim(); this.state.configured = true; this.authBlocked = false; this.clearCaches();
    await this.context.secrets.store(secretName, this.key); this.schedule(true, true); return true;
  }
  private async clearKey(): Promise<void> {
    await this.pause(); await this.context.secrets.delete(secretName);
    this.key = (process.env.AI_GATEWAY_API_KEY ?? '').trim(); this.state.configured = !!this.key;
    this.clearCaches(); this.state.message = this.key ? 'Stored key removed; an editor-environment key remains available.' : 'Stored key removed.'; this.publish();
  }
  private async testConnection(): Promise<void> {
    if (!this.key) { void vscode.window.showInformationMessage('Set your AI Gateway key first.'); return; }
    this.output.show(); if (this.connectionTest) return;
    const controller = new AbortController(); this.connectionTest = controller; this.output.clear();
    this.output.appendLine(`Kernel Lens ${this.state.version} · Gateway test · fixed sample data only`);
    try {
      const results = await testGatewayConnection(this.key, controller.signal, this.fetcher, result => this.output.appendLine(`${result.ok ? 'PASS' : 'FAIL'} · ${result.name}\n${result.detail}`), settings(this.editor?.document.uri));
      if (results.every(r => r.ok)) { this.authBlocked = false; this.schedule(true, true); }
    } catch { this.output.appendLine('Connection test canceled or unavailable.'); }
    finally { this.connectionTest = undefined; }
  }
  private async setIntent(value?: string): Promise<void> {
    const next = value ?? await vscode.window.showInputBox({ title: 'Optional context', value: this.state.intent, prompt: 'Shapes, hardware, accumulation policy or another contract outside the code.', ignoreFocusOut: true });
    if (next === undefined) return; this.state.intent = next.slice(0, 1600);
    await this.context.workspaceState.update('intent', this.state.intent); this.clearCaches(); this.schedule(false, true);
  }
  private async attach(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) { void vscode.window.showInformationMessage('Select source code to attach as context.'); return; }
    if (!['file', 'vscode-remote'].includes(editor.document.uri.scheme) || isExcluded(editor.document.uri.path, settings(editor.document.uri).excludedPaths)) return;
    const ref: AttachedReference = { id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, uri: editor.document.uri.toString(),
      name: vscode.workspace.asRelativePath(editor.document.uri, false), startLine: editor.selection.start.line, endLine: Math.min(editor.selection.end.line, editor.selection.start.line + 65) };
    this.refs = [...this.refs.filter(r => r.uri !== ref.uri || r.startLine !== ref.startLine), ref].slice(-3);
    await this.context.workspaceState.update('references', this.refs); this.syncRefs(); this.clearCaches(); this.schedule(false, true);
  }
  private syncRefs(): void { this.state.references = this.refs.map(r => ({ ...r, startLine: r.startLine + 1, endLine: r.endLine + 1 })); }
  private settingTarget(): vscode.ConfigurationTarget { return vscode.workspace.workspaceFolders ? vscode.ConfigurationTarget.Workspace : vscode.ConfigurationTarget.Global; }
  private async handleMessage(message: unknown): Promise<void> {
    if (!message || typeof message !== 'object') return; const msg = message as Record<string, unknown>;
    switch (msg.type) {
      case 'enable': case 'jev': await this.enable(); break;
      case 'pause': await this.pause(); break;
      case 'analyze': this.authBlocked = false; if (!this.state.enabled || !this.key || !this.state.consented) await this.enable(); else this.schedule(true, true); break;
      case 'key': await this.setKey(); break;
      case 'review': await this.runReview(false); break;
      case 'handoff': await this.copyHandoff(); break;
      case 'intent': if (typeof msg.value === 'string') await this.setIntent(msg.value); break;
      case 'attach': await this.attach(); break;
      case 'importEvidence': await this.importEvidence(); break;
      case 'removeReference':
        if (typeof msg.id === 'string') { this.refs = this.refs.filter(r => r.id !== msg.id); await this.context.workspaceState.update('references', this.refs); this.syncRefs(); this.clearCaches(); this.schedule(false, true); }
        break;
      case 'hardware': if (typeof msg.value === 'string' && hardwareId(msg.value) === msg.value) await vscode.workspace.getConfiguration('kernelLens').update('hardwareProfile', msg.value, this.settingTarget()); break;
      case 'domain': if (typeof msg.value === 'string' && ['auto', 'cuda', 'triton', 'pytorch', 'jax', 'distributed', 'serving', 'training', 'inference', 'data', 'general'].includes(msg.value)) await vscode.workspace.getConfiguration('kernelLens').update('domain', msg.value, this.settingTarget()); break;
      case 'jump': {
        if (!this.editor || !this.state.unit || this.state.stale) break;
        const advice = this.state.advisor.result?.items.find(a => a.metricId === msg.id);
        const editor = await vscode.window.showTextDocument(this.editor.document);
        const anchor = this.state.report?.anchors?.[String(msg.id)];
        const line = Math.min(editor.document.lineCount - 1, (advice?.line ?? anchor?.startLine ?? this.state.unit.startLine) - 1), pos = new vscode.Position(line, 0);
        editor.selection = new vscode.Selection(pos, pos); editor.revealRange(editor.document.lineAt(line).range, vscode.TextEditorRevealType.InCenterIfOutsideViewport); break;
      }
      case 'reference': { const metric = metrics.find(m => m.id === msg.id); if (metric) await vscode.env.openExternal(vscode.Uri.parse(metric.reference)); break; }
      case 'settings': await vscode.commands.executeCommand('workbench.action.openSettings', 'kernelLens'); break;
    }
  }
  dispose(): void {
    this.connectionTest?.abort(); this.invalidate(true); this.output.dispose(); this.subscriptions.forEach(s => s.dispose());
    this.diagnostics.dispose(); this.decoration.dispose(); this.status.dispose(); this.lensEvents.dispose();
    for (const decoration of this.findingDecorations.values()) decoration.dispose();
  }
}
