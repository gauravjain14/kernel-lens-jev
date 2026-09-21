import { parser as python } from '@lezer/python';
import { parser as cpp } from '@lezer/cpp';
import { TreeFragment, type SyntaxNode, type Tree } from '@lezer/common';

export interface ParsedSource {
  source: string; tree: Tree; python: boolean; nodes: SyntaxNode[];
  lines: number[]; masked: string; normalized: string;
  unfinishedLiterals: number[];
  text(node: SyntaxNode): string;
  line(offset: number): number;
}

/** Lexical masking must not depend on a recovering C++ parse. In particular,
 * braces in inline PTX strings are not C++ block delimiters. */
function cppLiterals(source: string): { masked: string; unfinished: number[] } {
  const chars=source.split(''), unfinished:number[]=[];
  const erase=(start:number,end:number)=>{for(let k=start;k<end;k++)if(chars[k]!=='\n'&&chars[k]!=='\r')chars[k]=' ';};
  for(let i=0;i<source.length;) {
    const start=i;
    if(source.startsWith('//',i)) {
      do {const end=source.indexOf('\n',i);i=end<0?source.length:end+1;} while(i<source.length&&/\\\r?\n$/.test(source.slice(start,i)));
    } else if(source.startsWith('/*',i)) {
      const end=source.indexOf('*/',i+2);if(end<0)unfinished.push(i);i=end<0?source.length:end+2;
    } else if(source.startsWith('R"',i)&&/^R"([^\s()\\]{0,16})\(/.test(source.slice(i))) {
      const raw=/^R"([^\s()\\]{0,16})\(/.exec(source.slice(i))!;
      const close=`)${raw[1]}"`,end=source.indexOf(close,i+raw[0].length);
      if(end<0)unfinished.push(i);i=end<0?source.length:end+close.length;
    } else if(source[i]==='"'||source[i]==="'") {
      // C++ numeric separators are not character-literal delimiters.
      if(source[i]==="'"&&/[0-9a-fA-F]/.test(source[i+1]??'')&&/\b(?:0[xX][\da-fA-F']+|0[bB][01']+|[0-9][\d']*)$/.test(source.slice(0,i))) {i++;continue;}
      const quote=source[i++];let closed=false;
      while(i<source.length) {
        if(source[i]==='\\') {i=Math.min(source.length,i+(source[i+1]==='\r'&&source[i+2]==='\n'?3:2));continue;}
        if(source[i]===quote) {i++;closed=true;break;}
        if(source[i]==='\n'||source[i]==='\r')break;
        i++;
      }
      if(!closed)unfinished.push(start);
    } else {i++;continue;}
    erase(start,i);
  }
  return {masked:chars.join(''),unfinished};
}

// CUDA attributes and launch configurations are not part of the C++ grammar.
// Replace only their spans, preserving every original diagnostic offset.
export function normalizeCuda(source: string, protectedText = cppLiterals(source).masked): string {
  const chars = source.split('');
  const erase = (start: number, end: number) => { for (let i = start; i < end; i++) if (chars[i] !== '\n' && chars[i] !== '\r') chars[i] = ' '; };
  for (const m of protectedText.matchAll(/\b__(?:global|device|host|shared|constant|managed|restrict|forceinline|noinline)__\b|\b__(?:launch_bounds|align)__\s*\([^)]*\)|<<<[\s\S]*?>>>/g)) erase(m.index, m.index + m[0].length);
  // Lezer does not understand GNU extended asm constraints. Treat a complete
  // asm(...) as an opaque call while preserving all original offsets and text.
  for(const m of protectedText.matchAll(/\b(asm|__asm__|__asm)\b\s*(?:(?:volatile|__volatile__|__volatile|inline|__inline__|goto)\b\s*)*\(/g)) {
    const open=m.index+m[0].length-1;let depth=1,end=open+1;
    for(;end<protectedText.length&&depth;end++) {if(protectedText[end]==='(')depth++;else if(protectedText[end]===')')depth--;}
    if(depth)continue; // Incomplete input still waits for its closing delimiter.
    erase(m.index+m[1]!.length,open);erase(open+1,end-1);
  }
  return chars.join('');
}

export function ancestor(node: SyntaxNode, names: string[]): SyntaxNode | undefined {
  for (let parent = node.parent; parent; parent = parent.parent) if (names.includes(parent.name)) return parent;
  return undefined;
}
export function children(node: SyntaxNode): SyntaxNode[] {
  const result: SyntaxNode[] = [];
  for (let child = node.firstChild; child; child = child.nextSibling) result.push(child);
  return result;
}
export function functionName(parsed: ParsedSource, node: SyntaxNode): string {
  const name = parsed.python ? node.getChild('VariableName') : node.getChild('FunctionDeclarator')?.getChild('Identifier');
  return name ? parsed.text(name) : parsed.text(node).split(/[({\n]/)[0]!.trim().slice(0, 80);
}

export class SourceParser {
  private cache = new Map<string, { source: string; tree: Tree; python: boolean }>();
  parse(key: string, source: string, isPython: boolean): ParsedSource {
    const lexical=isPython?undefined:cppLiterals(source);
    const normalized = isPython ? source : normalizeCuda(source,lexical!.masked);
    const previous = this.cache.get(key);
    let fragments: readonly TreeFragment[] = [];
    if (previous?.python === isPython) {
      let from = 0, oldEnd = previous.source.length, newEnd = normalized.length;
      while (from < oldEnd && from < newEnd && previous.source[from] === normalized[from]) from++;
      while (oldEnd > from && newEnd > from && previous.source[oldEnd - 1] === normalized[newEnd - 1]) { oldEnd--; newEnd--; }
      fragments = TreeFragment.applyChanges(TreeFragment.addTree(previous.tree), [{ fromA: from, toA: oldEnd, fromB: from, toB: newEnd }]);
    }
    const tree = previous?.source === normalized && previous.python === isPython ? previous.tree : (isPython ? python : cpp).parse(normalized, fragments);
    this.cache.delete(key); this.cache.set(key, { source: normalized, tree, python: isPython });
    while (this.cache.size > 8) this.cache.delete(this.cache.keys().next().value!);
    const nodes: SyntaxNode[] = [];
    const masked = (lexical?.masked??source).split('');
    let protectedUntil = -1;
    tree.iterate({ enter(ref) {
      if (ref.from < protectedUntil) {
        if (ref.type.isError) nodes.push(ref.node);
        return;
      }
      nodes.push(ref.node);
      if (/^(?:Comment|LineComment|BlockComment|String|FormatString|StringLiteral|CharLiteral|RawStringLiteral)$/.test(ref.name)) {
        if(isPython)for (let i = ref.from; i < ref.to; i++) if (masked[i] !== '\n' && masked[i] !== '\r') masked[i] = ' ';
        protectedUntil = ref.to;
      }
    } });
    const lines = [0];
    for (let i = 0; i < source.length; i++) if (source[i] === '\n') lines.push(i + 1);
    return { source, tree, python: isPython, nodes, lines, normalized, unfinishedLiterals:lexical?.unfinished??[], masked: masked.join(''), text: node => source.slice(node.from, node.to),
      line(offset) { let lo = 0, hi = lines.length; while (lo + 1 < hi) { const mid = (lo + hi) >>> 1; if (lines[mid]! <= offset) lo = mid; else hi = mid; } return lo + 1; } };
  }
  clear(key: string): void { this.cache.delete(key); }
}
