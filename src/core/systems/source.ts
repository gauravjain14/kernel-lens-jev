import { createHash } from 'node:crypto';
import type { SyntaxNode } from '@lezer/common';
import { SourceParser, ancestor, children, type ParsedSource } from '../syntax';
import type { CodeContext } from '../../live-types';
import type { Evidence, SystemsIR, SystemsOp } from './types';

export function sourceRevision(context: CodeContext): string {
  return createHash('sha256').update(JSON.stringify([context.file, context.code, context.enclosing, context.preamble, context.references,context.kernelHelpers])).digest('hex');
}
export class SourceView {
  readonly parsed: ParsedSource;
  readonly lineNumbers: number[];
  readonly ir: SystemsIR;
  private controls = new Map<number, string>();
  constructor(readonly context: CodeContext) {
    const lines = new Map<number, string>();
    for (const part of [context.preamble, context.enclosing, context.code]) for (const line of part.split('\n')) {
      const m = /^L(\d+): ?(.*)$/.exec(line);
      // A focused statement can begin midway through a line. Prefer the fuller
      // enclosing line so it cannot erase its function/control context.
      if (m && (lines.get(Number(m[1]))?.length ?? -1) < m[2]!.length) lines.set(Number(m[1]), m[2]!);
    }
    const ordered = [...lines].sort((a, b) => a[0] - b[0]);
    this.lineNumbers = ordered.map(([line]) => line);
    this.parsed = new SourceParser().parse(context.file, ordered.map(([, code]) => code).join('\n'), context.language === 'python' || /\.pyi?$/.test(context.file));
    const gap = this.lineNumbers.some((line, i) => i > 0 && line !== this.lineNumbers[i - 1]! + 1);
    const unitGap = Array.from({length:context.unit.endLine-context.unit.startLine+1},(_,i)=>context.unit.startLine+i).some(line=>!lines.has(line));
    const errors = this.parsed.nodes.some(n => n.type.isError);
    this.ir = { version: 1, frameworks: [], operations: [], tensors: [], coverage: {
      // Omitted neighboring definitions do not make a fully supplied function
      // incomplete. Absence claims remain bounded to this current source unit.
      complete: !context.truncated && !unitGap && !errors,
      limitations: [...(context.truncated ? ['Source context is truncated.'] : []), ...(gap ? ['Some source lines are outside the analyzed context.'] : []),
        ...(errors ? ['Some constructs could not be parsed.'] : []), 'Only recognized APIs and local dataflow are modeled; unknown callees are not expanded.'],
    }, enrichment: context.enrichment };
    // Evidence extraction is independent of issue recognition. Jev can assess
    // the fixed dimensions even when no specialized local rule matched.
    this.ir.sourceAnchors = this.parsed.nodes.filter(n => ['ForStatement', 'WhileStatement', 'IfStatement', 'ExpressionStatement', 'AssignStatement', 'Declaration', 'ReturnStatement', 'WithStatement', 'Decorator'].includes(n.name))
      .map(n => ({ ...this.evidence(n, 'Source construct'), role: n.name }))
      .filter(e => e.location.startLine >= context.unit.startLine && e.location.endLine <= context.unit.endLine && e.direct && e.source.trim().length > 2);
  }
  text(n: SyntaxNode) { return this.parsed.text(n); }
  masked(n: SyntaxNode) { return this.parsed.masked.slice(n.from, n.to); }
  nodes(name: string) { return this.parsed.nodes.filter(n => n.name === name); }
  region(n: SyntaxNode): string { const f = n.name === 'FunctionDefinition' ? n : ancestor(n, ['FunctionDefinition']); return f ? `function:${f.from}` : 'module'; }
  evidence(n: SyntaxNode, explanation: string, direct = true): Evidence {
    const p = this.parsed, line = p.line(n.from), end = p.line(Math.max(n.from, n.to - 1));
    const source = this.text(n);
    const key = createHash('sha256').update(`${n.from}:${n.to}:${explanation}`).digest('hex').slice(0, 12);
    return { id: `e_${key}`, source, location: { file: this.context.file, startLine: this.lineNumbers[line - 1]!, endLine: this.lineNumbers[end - 1]!,
      startColumn: n.from - p.lines[line - 1]!, endColumn: n.to - p.lines[end - 1]! }, explanation, origin: 'source', direct: direct && this.lineNumbers[end - 1]! - this.lineNumbers[line - 1]! === end - line };
  }
  add(n: SyntaxNode, kind: SystemsOp['kind'], op: SystemsOp['op'], attributes: SystemsOp['attributes'] = {}, inputs: string[] = [], outputs: string[] = [], direct = true): SystemsOp {
    const evidence = this.evidence(n, `${kind}: ${op}`, direct);
    const controls: string[] = [];
    for (let parent = n.parent; parent; parent = parent.parent) { if (parent.name === 'FunctionDefinition') break; const id = this.controls.get(parent.from); if (id) controls.push(id); }
    let unreachable = false;
    for (let parent = n.parent; parent && parent.name !== 'FunctionDefinition'; parent = parent.parent) {
      if (parent.name !== 'IfStatement') continue;
      const body = children(parent).find(c => ['Body', 'CompoundStatement', 'ExpressionStatement'].includes(c.name));
      if (body && n.from >= body.from && n.to <= body.to && /^if\s*\(?\s*(?:False|false|0)\s*\)?\s*$/.test(this.parsed.masked.slice(parent.from, body.from).trim())) unreachable = true;
    }
    const result = { id: `${kind}_${op}_${n.from}_${this.ir.operations.length}`, kind, op, evidence, inputs, outputs, controls, region: this.region(n), attributes: { ...attributes, source_order: n.from, source_end: n.to, ...(unreachable ? { unreachable: true } : {}) } } as SystemsOp;
    this.ir.operations.push(result);
    if (kind === 'control') this.controls.set(n.from, result.id);
    return result;
  }
  controlsAndLoops() {
    for (const n of this.parsed.nodes) {
      if (!['ForStatement', 'WhileStatement', 'IfStatement'].includes(n.name)) continue;
      const body = children(n).find(c => ['Body', 'CompoundStatement', 'ExpressionStatement'].includes(c.name));
      const header = this.parsed.masked.slice(n.from, body?.from ?? n.to).trim();
      this.add(n, 'control', n.name === 'IfStatement' ? 'branch' : 'loop', { header });
    }
  }
}
export function topArguments(s: string): string[] {
  const out: string[] = []; let from = 0, depth = 0, quote = '';
  for (let i = 0; i < s.length; i++) { const c = s[i]!;
    if (quote) { if (c === '\\') i++; else if (c === quote) quote = ''; continue; }
    if (c === '"' || c === "'") quote = c;
    else if ('([{'.includes(c)) depth++;
    else if (')]}'.includes(c)) depth--;
    else if (c === ',' && depth === 0) { out.push(s.slice(from, i).trim()); from = i + 1; }
  }
  if (s.slice(from).trim()) out.push(s.slice(from).trim()); return out;
}
export const compact = (s: string) => s.replace(/\s+/g, '');
