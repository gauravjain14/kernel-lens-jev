import { createHash } from 'node:crypto';
import { SourceParser, ancestor, functionName, type ParsedSource } from './syntax';
import { redact } from './context';
import type { CodeContext, Reference, SourceUnit } from '../live-types';
import { kernelHelpers } from './systems/cuda-semantics';

const blockNames = new Set(['FunctionDefinition', 'ForStatement', 'WhileStatement', 'IfStatement', 'WithStatement', 'TryStatement', 'CompoundStatement']);
const functionNames = ['FunctionDefinition'];
const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const numbered = (source: string, start: number) => source.split('\n').map((line, i) => `L${start + i}: ${line}`).join('\n');

export interface BlockInput {
  file: string; language: string; source: string; cursorLine: number; hardwareProfile?: import('./systems/hardware').HardwareId; intent?: string; hardware?: string;
  references?: Reference[]; maxCharacters?: number; secret?: string;
  assessmentScope?: 'block' | 'function';
}

/** Syntax is used to choose a stable assessment unit, never to decide performance. */
export class BlockReader {
  private parser = new SourceParser();
  close(key: string): void { this.parser.clear(key); }
  read(input: BlockInput): CodeContext {
    const { source } = input;
    if (source.length > 200_000) throw new Error('This file exceeds the 200,000-character live context limit.');
    const py = input.language === 'python' || /\.pyi?$/.test(input.file);
    const parsed = this.parser.parse(input.file, source, py);
    const cursorLine = Math.max(1, Math.min(parsed.lines.length, input.cursorLine + 1));
    const offset = parsed.lines[cursorLine - 1]!;
    const lineEnd = parsed.lines[cursorLine] ?? source.length;
    const firstCode = parsed.masked.slice(offset, lineEnd).search(/\S/);
    const focusOffset = firstCode < 0 ? offset : offset + firstCode;
    const candidates = parsed.nodes.filter(n => blockNames.has(n.name));
    let focus = candidates.filter(n => n.from < lineEnd && n.to > focusOffset && parsed.text(n).trim())
      .sort((a, b) => (a.to - a.from) - (b.to - b.from))[0];
    const decorated = parsed.nodes.find(n => n.name === 'DecoratedStatement' && n.from < lineEnd && n.to > focusOffset);
    if (!focus && decorated) focus = decorated.getChild('FunctionDefinition') ?? undefined;
    // Opening a CUDA file often places the cursor on a comment or constants.
    // Select a real kernel instead of a line window cutting through helpers.
    if(!py&&!focus&&input.assessmentScope==='function') {
      focus=candidates.filter(n=>n.name==='FunctionDefinition'&&/\b__global__\b/.test(parsed.masked.slice(parsed.lines[parsed.line(n.from)-1],n.from)))
        .sort((a,b)=>Math.abs(a.from-focusOffset)-Math.abs(b.from-focusOffset))[0];
    }
    // A standalone compound statement is the body of its named construct.
    if (focus?.name === 'CompoundStatement' && focus.parent && blockNames.has(focus.parent.name)) focus = focus.parent;
    // A containing function also spans blank lines. Prefer its just-finished
    // child instead of waiting for the entire enclosing function to be complete.
    if (!parsed.masked.slice(offset, lineEnd).trim()) {
      let finished = candidates.filter(n => n.to <= offset && !parsed.masked.slice(n.to, offset).trim()
        && balancedDelimiters(parsed.masked.slice(n.from, n.to)))
        .sort((a, b) => b.to - a.to || (b.to - b.from) - (a.to - a.from))[0];
      if (finished?.name === 'CompoundStatement' && finished.parent && blockNames.has(finished.parent.name)) finished = finished.parent;
      if (finished && (!focus || finished.from > focus.from)) focus = finished;
    }
    // Include complete statements immediately after a finished C++ block (for
    // example its following barrier). This also works inside an unfinished kernel.
    let trailingTo: number | undefined;
    if (!py) {
      const tail = parsed.nodes.filter(n => n.name === 'ExpressionStatement' && n.to <= lineEnd
        && (n.to >= offset || !parsed.masked.slice(n.to, offset).trim()))
        .sort((a, b) => b.to - a.to)[0];
      if (tail && parsed.masked.slice(tail.from, tail.to).trimEnd().endsWith(';')) {
        let previous = tail.prevSibling;
        while (previous && (previous.name === 'ExpressionStatement' || /Comment$/.test(previous.name))) previous = previous.prevSibling;
        if (previous && blockNames.has(previous.name) && previous.name !== 'FunctionDefinition'
          && (!focus || previous.from > focus.from) && balancedDelimiters(parsed.masked.slice(previous.from, tail.to))) {
          focus = previous; trailingTo = tail.to;
        }
      }
    }
    // Review a complete enclosing function as one stable unit. Cursor movement
    // inside it must not discard findings in neighboring branches. While typing
    // an unfinished function, keep the existing complete-child behavior.
    const enclosingFunction = focus?.name === 'FunctionDefinition' ? focus : focus && ancestor(focus, functionNames);
    if (input.assessmentScope === 'function' && enclosingFunction
      && balancedDelimiters(parsed.masked.slice(enclosingFunction.from, enclosingFunction.to))
      && !parsed.nodes.some(n => n.type.isError && n.from >= enclosingFunction.from && n.to <= enclosingFunction.to)) {
      focus = enclosingFunction; trailingTo = undefined;
    }
    let from = focus?.from ?? Math.max(0, parsed.lines[Math.max(0, cursorLine - 61)] ?? 0);
    let to = trailingTo ?? focus?.to ?? Math.min(source.length, parsed.lines[Math.min(parsed.lines.length, cursorLine + 60)] ?? source.length);
    if (!focus) to = Math.max(to, lineEnd);
    // Keep decorators with the function, and C++ CUDA attributes erased for parsing.
    if (focus?.name === 'FunctionDefinition') {
      if (focus.parent?.name === 'DecoratedStatement') from = focus.parent.from;
      else from = parsed.lines[parsed.line(from) - 1]!;
    }
    const syntax = parsed.nodes.filter(n => n.type.isError && n.from >= from && n.from <= to).slice(0, 12).map(n => ({
      from: n.from, to: Math.min(source.length, Math.max(n.from + 1, n.to)), line: parsed.line(n.from),
      message: 'Incomplete or unrecognized syntax near this position.',
    }));
    const fragment = source.slice(from, to);
    const masked = parsed.masked.slice(from, to).trim();
    const hasBody = py ? !/:\s*(?:#[^\n]*)?$/.test(masked) : !/[{(,=]\s*$/.test(masked);
    const balanced = balancedDelimiters(masked);
    // A recoverable parser error does not suppress semantic assessments forever.
    // Unclosed delimiters and missing Python suites are the typing-time gate.
    const ready = !!masked && hasBody && balanced && !(py && syntax.length > 0)
      && !parsed.unfinishedLiterals.some(offset=>offset>=from&&offset<to);
    const fn = focus?.name === 'FunctionDefinition' ? focus : focus && ancestor(focus, functionNames);
    const name = fn ? functionName(parsed, fn) : 'module';
    const scopes: string[] = [];
    for (let parent = focus; parent && !(parent.name === 'FunctionDefinition' && parent.from === fn?.from); parent = parent.parent ?? undefined) {
      if (!blockNames.has(parent.name) || parent.name === 'CompoundStatement') continue;
      let ordinal = 0;
      for (let sibling = parent.prevSibling; sibling; sibling = sibling.prevSibling) if (sibling.name === parent.name) ordinal++;
      scopes.unshift(`${parent.name}:${ordinal}`);
    }
    const classNode = fn && ancestor(fn, ['ClassDefinition']);
    const className = classNode ? parsed.text(classNode).split(/[:{\n]/)[0]!.trim() : '';
    const identity = `${className}:${name}:${scopes.join('/') || 'function'}`;
    const unit: SourceUnit = { name, kind: focus?.name ?? 'Module', from, to, identity,
      startLine: parsed.line(from), endLine: parsed.line(Math.max(from, to - 1)), ready, syntax };
    const automatic = this.references(parsed, from, to, fn?.from, fn?.to);
    const fnStart = fn?.parent?.name === 'DecoratedStatement' ? fn.parent.from : fn?.from;
    let enclosing = fn && fnStart !== undefined && (fnStart < from || fn.to > to)
      ? numbered(source.slice(parsed.lines[parsed.line(fnStart) - 1], fn.to), parsed.line(fnStart)) : '';
    // Imports, top-level setup, and class initialization are evidence, not just imports.
    const imports = parsed.nodes.filter(n => /Import/.test(n.name) && n.parent?.name === parsed.tree.topNode.name);
    const importedNames = new Set(parsed.nodes.filter(n => n.name === 'VariableName' && imports.some(i => n.from >= i.from && n.to <= i.to)).map(n => parsed.text(n)));
    const usedNames = new Set(parsed.nodes.filter(n => ['VariableName', 'Identifier'].includes(n.name)
      && n.from >= (fn?.from ?? from) && n.to <= (fn?.to ?? to) && !importedNames.has(parsed.text(n))).map(n => parsed.text(n)));
    const outside = parsed.nodes.filter(n => n.parent?.name === parsed.tree.topNode.name
      && !['FunctionDefinition', 'DecoratedStatement', 'ClassDefinition'].includes(n.name));
    const relevantRoots = new Set<number>();
    for (const child of parsed.nodes) {
      if (!['VariableName', 'Identifier'].includes(child.name) || !usedNames.has(parsed.text(child))) continue;
      let root = child;
      while (root.parent && root.parent.name !== parsed.tree.topNode.name) root = root.parent;
      relevantRoots.add(root.from);
    }
    const relevant = outside.filter(n => relevantRoots.has(n.from));
    // Python globals may be initialized after a function declaration and before
    // its call. Keep relevant setup on either side, in original source order.
    const setup = [...new Map([...imports, ...relevant, ...outside.filter(n => n.to <= from).slice(-8)]
      .map(n => [n.from, n])).values()].slice(0, 24).sort((a, b) => a.from - b.from);
    const preamble = setup.map(n => numbered(py ? parsed.text(n) : source.slice(parsed.lines[parsed.line(n.from) - 1], n.to), parsed.line(n.from))).join('\n');
    const clean = (s: string) => redact(input.secret ? s.replaceAll(input.secret, '[REDACTED]') : s);
    const budget = Math.max(4000, Math.min(40000, input.maxCharacters ?? 16000));
    // For a long function, retain the nearest enclosing conditions/loops before
    // its setup excerpt. Prefix-only truncation can otherwise erase the branch
    // controlling a late inner loop, even though that branch is available locally.
    if (enclosing.length > Math.floor(budget * .22) && focus && fn) {
      const parents: string[] = [];
      for (let parent = focus.parent; parent && parent.from >= fn.from; parent = parent.parent) {
        if (parent.name === 'FunctionDefinition') break;
        if (blockNames.has(parent.name) && parent.name !== 'CompoundStatement') {
          parents.push(numbered(parsed.text(parent), parsed.line(parent.from)));
        }
      }
      if (parents.length) enclosing = [...parents, enclosing].join('\n[… enclosing scope …]\n');
    }
    let remaining = budget, truncated = false;
    const take = (s: string, limit: number) => {
      const value = clean(s); const result = value.slice(0, Math.min(remaining, limit));
      if (result.length < value.length) truncated = true;
      remaining -= result.length; return result;
    };
    // A large block is centered on the changed line; its full range stays explicit.
    const rawLines = fragment.split('\n');
    let first = 0, last = rawLines.length;
    const focusBudget = Math.floor(budget * (input.assessmentScope==='function'&&focus?.name==='FunctionDefinition'&&!enclosing.length ? .85 : .55));
    const costs = rawLines.map((line, i) => line.length + String(unit.startLine + i).length + 4);
    let focusLength = costs.reduce((sum, n) => sum + n, 0) - 1;
    while (focusLength > focusBudget && last - first > 1) {
      if (cursorLine - unit.startLine - first > last - (cursorLine - unit.startLine)) focusLength -= costs[first++]!;
      else focusLength -= costs[--last]!;
      truncated = true;
    }
    const code = take(numbered(rawLines.slice(first, last).join('\n'), unit.startLine + first), focusBudget);
    const intent = take(input.intent ?? '', 1200);
    const hardware = take(input.hardware ?? '', 600);
    const outer = take(enclosing, Math.floor(budget * .22));
    const header = take(preamble, 2200);
    const references = [...input.references ?? [], ...automatic].slice(0, 5).map(ref => ({
      ...ref, name: clean(ref.name).slice(0, 240), reason: clean(ref.reason).slice(0, 180), code: take(ref.code, 1800),
    })).filter(ref => ref.code);
    const helpers = py ? [] : JSON.parse(clean(JSON.stringify(kernelHelpers(parsed,input.file,fn?.from??from,fn?.to??to))));
    return { file: input.file, language: input.language, unit, code, enclosing: outer, preamble: header,
      ...(helpers.length ? {kernelHelpers:helpers} : {}),
      references, intent, hardware, hardwareProfile: input.hardwareProfile ?? 'b200', truncated, characters: budget - remaining,
      routingKey: hash(JSON.stringify([input.file, input.language, name, code, outer, header, references, intent])),
    };
  }
  private references(p: ParsedSource, from: number, to: number, fnFrom?: number, fnTo?: number): Reference[] {
    const all = p.nodes.filter(n => n.name === 'FunctionDefinition');
    const focus = p.source.slice(fnFrom ?? from, fnTo ?? to);
    const current = all.find(n => n.from === fnFrom);
    const currentName = current ? functionName(p, current) : '';
    const currentClass = current && ancestor(current, ['ClassDefinition']);
    return all.flatMap(n => {
      if (n.from === fnFrom || n.from >= from && n.to <= to) return [];
      const name = functionName(p, n);
      const called = /^[\w]+$/.test(name) && new RegExp(`\\b${name}\\s*\\(`).test(focus);
      const caller = currentName && /^[\w]+$/.test(currentName) && new RegExp(`\\b${currentName}\\s*\\(`).test(p.text(n));
      const targetClass = ancestor(n, ['ClassDefinition']);
      const init = p.python && !!currentClass && name === '__init__' && targetClass?.from === currentClass.from;
      if (!called && !caller && !init) return [];
      if (currentClass && targetClass && currentClass.from !== targetClass.from) return [];
      const declaration = n.parent?.name === 'DecoratedStatement' ? n.parent : n;
      return [{ name, startLine: p.line(declaration.from), code: numbered(p.text(declaration), p.line(declaration.from)),
        reason: init ? 'Class initialization' : called ? 'Called helper' : 'Calling function', priority: init ? 0 : called ? 1 : 2 }];
    }).sort((a, b) => a.priority - b.priority).slice(0, 4).map(({ priority: _priority, ...reference }) => reference);
  }
}

export function balancedDelimiters(source: string): boolean {
  const stack: string[] = [], pairs: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
  for (const c of source) {
    if ('([{'.includes(c)) stack.push(c);
    else if (pairs[c] && stack.pop() !== pairs[c]) return false;
  }
  return stack.length === 0;
}
export function contextHash(context: CodeContext): string {
  const { routingKey: _routing, characters: _count, recentEdit: _edit, unit, ...rest } = context;
  return hash(JSON.stringify({ ...rest, unit: { name: unit.name, kind: unit.kind, startLine: unit.startLine, endLine: unit.endLine } }));
}

/** A bounded edit hint; current source remains authoritative and cacheable. */
export function withRecentEdit(previous: CodeContext | undefined, current: CodeContext, budget: number): CodeContext {
  if (!previous || previous.file !== current.file || previous.unit.identity !== current.unit.identity || previous.code === current.code) return current;
  const plain = (s: string) => s.split('\n').map(line => line.replace(/^L\d+: /, ''));
  const before = plain(previous.code), after = plain(current.code);
  let first = 0, oldEnd = before.length, newEnd = after.length;
  while (first < oldEnd && first < newEnd && before[first] === after[first]) first++;
  while (oldEnd > first && newEnd > first && before[oldEnd - 1] === after[newEnd - 1]) { oldEnd--; newEnd--; }
  if (first === before.length && first === after.length) return current;
  const cap = Math.min(600, Math.floor((budget - current.characters) / 2));
  if (cap < 80) return current;
  const start = Number(/^L(\d+):/.exec(current.code.split('\n')[first] ?? '')?.[1] ?? current.unit.startLine);
  const edit = { before: before.slice(first, oldEnd).join('\n').slice(0, cap), after: after.slice(first, newEnd).join('\n').slice(0, cap), startLine: start, endLine: start + Math.max(0, newEnd - first - 1) };
  return { ...current, recentEdit: edit, characters: current.characters + edit.before.length + edit.after.length };
}
