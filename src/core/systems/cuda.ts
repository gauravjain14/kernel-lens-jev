import { ancestor, children } from '../syntax';
import { compact, SourceView, topArguments } from './source';
import { cudaEffects } from './cuda-semantics';
import type { Attribute, Evidence } from './types';

/** CUDA syntax is lowered once; lenses consume memory spaces and control ancestry. */
export function adaptCuda(v: SourceView): void {
  const p = v.parsed;
  if (!/\b(?:__global__|__device__|threadIdx|blockIdx|__syncthreads)\b/.test(p.masked)) return;
  v.ir.frameworks.push('cuda'); v.controlsAndLoops();
  for (const n of p.nodes.filter(n => ['ParameterDeclaration', 'Declaration'].includes(n.name))) {
    const decl = children(n).find(c => /Declarator$/.test(c.name));
    if (!decl) continue;
    const name = p.nodes.find(c => c.name === 'Identifier' && c.from >= decl.from && c.to <= decl.to);
    if (!name) continue;
    const start = Math.max(p.source.lastIndexOf(';', n.from - 1), p.source.lastIndexOf('{', n.from - 1), p.source.lastIndexOf('}', n.from - 1)) + 1;
    const prefix = p.masked.slice(start, n.from);
    const raw = v.masked(n), dimensions=[...v.text(decl).matchAll(/\[([^\]]*)\]/g)].map(m=>m[1]!.trim());
    const dtypeBytes=/\b(?:float4|int4|uint4)\b/.test(raw)?16:/\b(?:double|float2|int2)\b/.test(raw)?8:/\b(?:half|short|__half|nv_bfloat16)\b/.test(raw)?2:/\b(?:char|int8_t|uint8_t)\b/.test(raw)?1:4;
    const shape=dimensions.length&&dimensions.every(d=>/^\d+$/.test(d))?dimensions.map(Number):undefined;
    const memory = /\b__shared__\b/.test(prefix + raw) ? 'shared' : n.name === 'ParameterDeclaration' && /\*/.test(raw) ? 'global' : 'local';
    const evidence = v.evidence(n, `Declared ${memory} storage`);
    v.ir.tensors.push({ name: v.text(name), region: v.region(n), memory, device: 'gpu', evidenceId: evidence.id,
      ...(shape ? { shape, dtypeBytes } : {}) });
    if (dimensions.length && n.name!=='ParameterDeclaration') v.add(n, 'memory', 'allocate', { memory,dimensions,dtype_bytes:dtypeBytes,
      symbolic_bytes: dimensions.some(d=>!d)?'dynamic shared-memory launch allocation':`${dtypeBytes} * ${dimensions.map(d=>`(${d})`).join(' * ')}`,
      ...(shape?{elements:shape.reduce((a,b)=>a*b,1),bytes:shape.reduce((a,b)=>a*b,dtypeBytes)}:{}) }, [], [v.text(name)]);
  }
  for (const n of p.nodes) {
    if (['AssignmentExpression', 'UpdateExpression'].includes(n.name)) {
      const target = children(n).find(c => c.name === 'Identifier');
      if (target && (n.firstChild?.name === 'Identifier' || n.name === 'UpdateExpression')) {
        v.add(n, 'control', 'conditional_execution', { writes_symbol: v.text(target) }, [], [v.text(target)]);
      }
      if(n.name==='AssignmentExpression'&&/\+=/.test(v.masked(n))&&/\*/.test(v.masked(n)))v.add(n,'compute','elementwise',{scalar_accumulation:true});
    }
    if (n.name === 'ForStatement' || n.name === 'WhileStatement') {
      const op = v.ir.operations.find(o => o.kind === 'control' && o.evidence.location.startLine === v.evidence(n, '').location.startLine && o.evidence.source === v.text(n));
      if (op) {
        const h = String(op.attributes.header), variable = /(?:int|unsigned|size_t)\s+(\w+)\s*=/.exec(h)?.[1];
        if (variable) op.attributes.variable = variable;
        op.attributes.halving = /(?:\/=\s*2|>>=\s*1|=\s*\w+\s*(?:\/\s*2|>>\s*1))/.test(h);
        const body=children(n).find(c=>['CompoundStatement','ExpressionStatement'].includes(c.name));
        op.attributes.reduction = !!body && p.nodes.some(x => x.name === 'AssignmentExpression' && x.from >= body.from && x.to <= body.to && /(?:\+=|\*=)/.test(v.masked(x)));
        const bound = /<\s*(\d+)/.exec(h); if (bound) op.attributes.bound = Number(bound[1]);
        const start = /\b\w+\s*=\s*(\d+)/.exec(h); if (bound && start) op.attributes.iterations = Math.max(0, Number(bound[1]) - Number(start[1]));
        op.attributes.mutated_names = p.nodes.filter(x => x.from >= n.from && x.to <= n.to && ['AssignmentExpression', 'UpdateExpression', 'InitDeclarator'].includes(x.name))
          .flatMap(x => x.firstChild?.name === 'Identifier' ? [v.text(x.firstChild)] : x.name === 'UpdateExpression' ? children(x).filter(c => c.name === 'Identifier').map(c => v.text(c)) : []);
      }
    }
    if (n.name === 'CallExpression') {
      const name = n.firstChild && compact(v.text(n.firstChild));
      const argumentsNode=children(n).find(c=>c.name==='ArgumentList');
      const args=argumentsNode?topArguments(v.text(argumentsNode).slice(1,-1)).map(s=>compact(s).slice(0,240)):[];
      const known=cudaEffects(name??'',v.text(n));
      const helper=v.context.kernelHelpers?.find(h=>h.name===name);
      for(const e of [...known,...helper?.effects??[]]) {
        const related=(e as {evidence?:Evidence}).evidence;
        if(related&&!v.ir.sourceAnchors?.some(a=>a.id===related.id))v.ir.sourceAnchors?.push({...related,role:'HelperInstruction'});
        const transfer:Record<string,Attribute>=e.op==='copy'?{source_address:args[1]??'',destination_address:args[0]??'',transfer_bytes:args[2]??''}:{};
        v.add(n,e.kind,e.op,{...e.attributes,call:name??'',arguments:args,...transfer,...(related?{helper_evidence:[related.id]}:{})},e.op==='copy'?[args[1]??'']:[],e.op==='copy'?[args[0]??'']:[]);
      }
      if (name === '__syncthreads') v.add(n, 'synchronization', 'block_barrier');
      else if (name === '__syncwarp') v.add(n, 'synchronization', 'warp_barrier');
      else if (/^__shfl_(?:down|xor)_sync$/.test(name ?? '')) {
        const update = ancestor(n, ['AssignmentExpression']);
        v.add(n, 'compute', 'reduction', { algorithm: 'warp_shuffle', accumulates: !!update && /\+=/.test(v.masked(update)) });
      } else if (/^atomic(?:Add|Max|Min|CAS)$/.test(name ?? '')) v.add(n, 'memory', 'store', { atomic: true });
      else if (/^(?:sqrtf?|expf?|logf?|powf?|sinf?|cosf?)$/.test(name ?? '')) v.add(n, 'compute', 'elementwise', { expensive_math: name! });
      else if(!known.length||/^(?:asm|__asm__|__asm)$/.test(name??''))v.add(n, 'control', 'conditional_execution', { unknown_call: true, call: name ?? '',arguments:args,semantic_helper:!!helper });
    }
    if (n.name !== 'SubscriptExpression' || n.parent?.name === 'SubscriptExpression') continue;
    const expr = compact(v.masked(n)), m = /^(\w+)\[(.*)\]$/.exec(expr); if (!m) continue;
    const [base, index] = [m[1]!, m[2]!];
    const storage = v.ir.tensors.filter(t => t.name === base && t.region === v.region(n));
    if (storage.length !== 1) continue; // A shadow or ambiguous pointer is not proof of global output.
    const parent = n.parent, assignment = parent?.name === 'AssignmentExpression' && parent.firstChild?.from === n.from ? parent : undefined;
    const target = assignment ? compact(v.masked(assignment)).slice(expr.length) : '';
    const write = !!assignment, rmw = /^[+*\-\/|&^]=/.test(target);
    v.add(n, 'memory', write ? 'store' : 'load', { memory: storage[0]!.memory!, address: expr, index, read_modify_write: rmw, variable: base }, [base]);
  }
}
