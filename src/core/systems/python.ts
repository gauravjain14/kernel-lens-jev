import type { SyntaxNode } from '@lezer/common';
import { ancestor, children } from '../syntax';
import { SourceView, compact, topArguments } from './source';
import { apiSemantics, type ApiSemantics } from './semantics';
import type { Framework, TensorFact, Attribute } from './types';

const simple = /^[A-Za-z_]\w*$/;
const literal = (s: string): Attribute | undefined => {
  if (/^(True|False)$/.test(s)) return s === 'True';
  if (/^-?\d+(?:\.\d+)?$/.test(s) && Number.isFinite(Number(s))) return Number(s);
  if (/^(['"])[^'"\n]*\1$/.test(s)) return s.slice(1, -1);
  const seq = s.replace(/^[([]|[)\]]$/g, '').split(',').map(s => s.trim()).filter(Boolean);
  if (/^[([]/.test(s) && seq.length && seq.every(s => /^\d+$/.test(s))) return seq.map(Number);
  return undefined;
};
const tensorBytes = (fact?: TensorFact) => { const n = fact?.shape && fact.dtypeBytes ? fact.shape.reduce((a, b) => a * b, fact.dtypeBytes) : undefined; return n !== undefined && Number.isSafeInteger(n) ? n : undefined; };

export function adaptPython(v: SourceView): void {
  v.controlsAndLoops();
  const p = v.parsed;
  const imports: { local: string; qualified: string; from: number; region: string }[] = [];
  for (const n of v.nodes('ImportStatement')) {
    const text = v.text(n).trim(), from = /^from\s+([\w.]+)\s+import\s+([\s\S]+)$/.exec(text);
    const entries = (from ? from[2]! : text.replace(/^import\s+/, '')).replace(/[()]/g, '').split(',');
    for (const entry of entries) {
      const m = /^\s*([\w.]+)(?:\s+as\s+(\w+))?\s*$/.exec(entry); if (!m) continue;
      const qualified = from ? `${from[1]}.${m[1]}` : m[2] ? m[1]! : m[1]!.split('.')[0]!;
      imports.push({ local: m[2] ?? (from ? m[1]! : m[1]!.split('.')[0]!), qualified, from: n.from, region: v.region(n) });
    }
  }
  const assignments = v.nodes('AssignStatement');
  const lhs = (n: SyntaxNode) => n.firstChild?.name === 'VariableName' ? v.text(n.firstChild) : '';
  const shadowed = (root: string, n: SyntaxNode, importedAt: number) => {
    const fn = ancestor(n, ['FunctionDefinition']);
    if (fn?.getChild('ParamList') && new RegExp(`\\b${root}\\b`).test(v.text(fn.getChild('ParamList')!))) return true;
    if (p.nodes.some(d => ['FunctionDefinition', 'ClassDefinition'].includes(d.name) && d.getChild('VariableName') && v.text(d.getChild('VariableName')!) === root && d.from > importedAt && d.from < n.from)) return true;
    if (assignments.some(a => a.firstChild?.name === 'MemberExpression' && compact(v.text(a.firstChild)).startsWith(`${root}.`) && a.from > importedAt && a.from < n.from)) return true;
    return assignments.some(a => lhs(a) === root && (v.region(a) === v.region(n) && v.region(n) !== 'module' || v.region(a) === 'module' && a.from > importedAt && a.from < n.from));
  };
  const resolve = (name: string, n: SyntaxNode): string | undefined => {
    const root = name.split('.')[0]!;
    const matches = imports.filter(i => i.local === root && i.from < n.from && (i.region === 'module' || i.region === v.region(n)));
    const found = matches.sort((a, b) => Number(b.region === v.region(n)) - Number(a.region === v.region(n)) || b.from - a.from)[0];
    if (!found || shadowed(root, n, found.from)) return undefined;
    return found.qualified + name.slice(root.length);
  };
  const noteFramework = (api: string) => {
    const root = api.split('.')[0];
    const framework = root === 'torch' ? 'pytorch' : root as Framework;
    if (['pytorch', 'jax', 'triton', 'vllm', 'sglang'].includes(framework) && !v.ir.frameworks.includes(framework)) v.ir.frameworks.push(framework);
  };
  const bindings: { name: string; region: string; from: number; assignment?: number; fact: TensorFact }[] = [];
  const lookup = (name: string, n: SyntaxNode): TensorFact | undefined => {
    if (!simple.test(name)) return undefined;
    const region = v.region(n), fn = ancestor(n, ['FunctionDefinition']);
    const local = assignments.some(a => lhs(a) === name && v.region(a) === region) || !!fn?.getChild('ParamList') && new RegExp(`\\b${name}\\b`).test(v.text(fn.getChild('ParamList')!));
    const latest = bindings.filter(b => b.name === name && b.from < n.from && (b.region === region || !local && b.region === 'module')).sort((a, b) => b.from - a.from)[0];
    if (!latest) {
      const supplied = v.context.enrichment?.tensors?.[name];
      return supplied ? { name, region, shape: supplied.shape, dtypeBytes: supplied.dtype_bytes, device: supplied.device ?? 'unknown', evidenceId: 'supplied_tensor' } : undefined;
    }
    // A later assignment to an unrecognized expression invalidates previous provenance.
    if (assignments.some(a => lhs(a) === name && v.region(a) === latest.region && a.from > (latest.assignment ?? latest.from) && a.to < n.from)) return undefined;
    return latest.fact;
  };
  for (const n of v.nodes('FunctionDefinition')) {
    const params = n.getChild('ParamList'); if (!params) continue;
    for (const param of topArguments(v.text(params).slice(1, -1))) {
      const m = /^(\w+)\s*:\s*([\w.]+)/.exec(param); if (!m) continue;
      // Resolve the annotation in the enclosing declaration scope, before the parameter takes effect.
      const root = m[2]!.split('.')[0]!, imp = imports.find(i => i.local === root && i.from < n.from);
      const annotation = imp ? imp.qualified + m[2]!.slice(root.length) : '';
      if (!['torch.Tensor', 'jax.Array'].includes(annotation)) continue;
      const e = v.evidence(params, 'Tensor parameter annotation');
      const supplied = v.context.enrichment?.tensors?.[m[1]!];
      const fact: TensorFact = { name: m[1]!, region: `function:${n.from}`, device: supplied?.device ?? 'unknown', evidenceId: e.id,
        ...(supplied ? { shape: supplied.shape, dtypeBytes: supplied.dtype_bytes } : {}) };
      bindings.push({ name: fact.name, region: fact.region, from: params.to, fact }); v.ir.tensors.push(fact);
    }
  }
  // Decorators without parentheses have no CallExpression but are real compilation boundaries.
  for (const n of v.nodes('Decorator')) {
    const name = compact(v.text(n)).replace(/^@/, '').split('(')[0]!, api = resolve(name, n);
    if (api) noteFramework(api);
    if (!api || apiSemantics[api]?.kind !== 'compilation') continue;
    const fn = n.parent?.getChild('FunctionDefinition');
    if (fn) { noteFramework(api); v.add(n, 'compilation', apiSemantics[api]!.op, { api, body_region: `function:${fn.from}`, decorator: true }); }
  }
  // Inner calls precede outer calls; provenance is carried through recognized expressions only.
  const calls = v.nodes('CallExpression').sort((a, b) => a.to - b.to || b.from - a.from);
  const expressions = new Map<number, TensorFact>();
  for (const n of calls) {
    const callee = n.firstChild; if (!callee) continue;
    const rawName = compact(v.text(callee)), argNode = n.getChild('ArgList'); if (!argNode) continue;
    const args = topArguments(v.text(argNode).slice(1, -1));
    const keywords: Record<string, Attribute> = {};
    for (const a of args) { const m = /^(\w+)\s*=\s*([^=][\s\S]*)$/.exec(a); if (m) { const value = literal(m[2]!); if (value !== undefined) keywords[m[1]!] = value; } }
    const positional = args.filter(a => !/^\w+\s*=/.test(a));
    const recreated=/^iter\((\w+)\)$/.exec(compact(positional[0]??''));
    if(rawName==='next' && recreated && !shadowed('next',n,-1) && !shadowed('iter',n,-1)) {
      const loader=v.ir.operations.find(o=>o.op==='batch' && o.attributes.api==='torch.utils.data.DataLoader'
        && o.outputs.includes(recreated[1]!) && (o.region===v.region(n)||o.region==='module') && Number(o.attributes.source_end)<n.from);
      if(loader && !assignments.some(a=>lhs(a)===recreated[1] && a.from>Number(loader.attributes.source_end) && a.from<n.from)) {
        v.add(n,'scheduling','batch',{iterator_recreated:true,loader_evidence:loader.evidence.id},[recreated[1]!]);
        continue;
      }
    }
    const inputFact = lookup(positional[0] ?? '', n);
    let api = resolve(rawName, n), semantics: ApiSemantics | undefined = api ? apiSemantics[api] : undefined;
    if (api) noteFramework(api);
    let receiver: TensorFact | undefined;
    const member = callee.name === 'MemberExpression' ? callee : undefined;
    const method = member?.lastChild ? v.text(member.lastChild) : '';
    if (member?.firstChild) receiver = member.firstChild.name === 'CallExpression' ? expressions.get(member.firstChild.from) : lookup(v.text(member.firstChild), n);
    if (receiver && assignments.some(a => a.firstChild?.name === 'MemberExpression' && compact(v.text(a.firstChild)) === rawName && a.from < n.from)) receiver = undefined;
    if (!semantics && receiver) {
      api = `tensor.${method}`;
      if (['item', 'cpu', 'numpy'].includes(method)) semantics = { kind: 'memory', op: 'device_to_host' };
      else if (['cuda', 'to'].includes(method)) semantics = { kind: 'memory', op: 'copy', tensor: true };
      else if (['clone', 'contiguous', 'repeat'].includes(method)) semantics = { kind: 'memory', op: 'materialize', tensor: true };
      else if (['sum', 'mean', 'max', 'softmax'].includes(method)) semantics = { kind: 'compute', op: 'reduction', tensor: true };
      else if (['reshape', 'view', 'detach', 'transpose', 'permute'].includes(method)) semantics = { kind: 'memory', op: 'load', tensor: true };
      else if (method === 'backward') semantics = { kind: 'compute', op: 'reduction' };
    }
    if (!semantics) {
      if (!['range', 'len', 'enumerate', 'zip', 'int', 'float'].includes(rawName) && !ancestor(n, ['Decorator'])) v.add(n, 'control', 'conditional_execution', { unknown_call: true, call: rawName });
      continue;
    }
    if (api && !api.startsWith('tensor.')) noteFramework(api);
    const assignment = ancestor(n, ['AssignStatement']);
    const outer = ancestor(n, ['CallExpression']);
    const assigned = assignment && (!outer || outer.from < assignment.from) ? lhs(assignment) : '';
    const input = receiver ?? (semantics.inputArg !== undefined ? lookup(positional[semantics.inputArg] ?? '', n) : inputFact);
    const outputs = semantics.outputArg !== undefined ? [positional[semantics.outputArg] ?? ''].filter(x => simple.test(x)) : assigned ? [assigned] : [];
    const inputs = receiver ? [receiver.name] : positional.flatMap(arg => simple.test(arg) ? [arg] : topArguments(arg.replace(/^[([]|[)\]]$/g, '')).filter(x => simple.test(x)));
    const attributes: Record<string, Attribute> = { api: api!, ...keywords, input_device: input?.device ?? 'unknown' };
    // Configuration dimensions belong to this engine only when explicitly passed
    // as model overrides. Unused similarly named globals are not model evidence.
    if (api?.startsWith('vllm.')) {
      const overrides = args.find(a => /^hf_overrides\s*=/.test(a))?.replace(/^hf_overrides\s*=\s*/, '');
      if (overrides?.startsWith('{') && overrides.endsWith('}')) for (const entry of topArguments(overrides.slice(1, -1))) {
        const m = /^['"](num_hidden_layers|num_key_value_heads|head_dim)['"]\s*:\s*(\d+)$/.exec(entry.trim());
        if (m && Number(m[2]) > 0) attributes[m[1]!] = Number(m[2]);
      }
      const dtype = typeof keywords.kv_cache_dtype === 'string' && keywords.kv_cache_dtype !== 'auto' ? keywords.kv_cache_dtype : keywords.dtype;
      if (typeof dtype === 'string') {
        const width = /^(?:fp8|fp8_e4m3|fp8_e5m2)$/.test(dtype) ? 1 : /^(?:half|float16|bfloat16)$/.test(dtype) ? 2 : dtype === 'float32' ? 4 : undefined;
        if (width) attributes.kv_cache_dtype_bytes = width;
      }
    }
    if (input && tensorBytes(input) !== undefined) attributes.tensor_bytes = tensorBytes(input)!;
    if (input?.shape) attributes.input_shape = input.shape;
    if (['torch.matmul','torch.mm','jax.numpy.matmul','jax.numpy.dot'].includes(api ?? '')) {
      const right=lookup(positional[1] ?? '', n), left=input;
      if(left?.shape?.length===2 && right?.shape?.length===2 && left.shape[1]===right.shape[0]
        && left.device==='gpu' && right.device==='gpu' && left.dtype==='float32' && right.dtype==='float32') {
        const [m,k]=left.shape as [number,number], nn=right.shape[1]!;
        const flops=2*m*k*nn, bytes=(m*k+k*nn+m*nn)*4;
        if(Number.isSafeInteger(flops)&&Number.isSafeInteger(bytes)&&flops>0&&bytes>0)
          Object.assign(attributes,{gemm_flops:flops,gemm_minimum_bytes:bytes,gemm_fp32:true});
      }
    }
    if (api === 'torch.cuda.graph') {
      const capture = ancestor(n, ['WithStatement']);
      if (capture) { attributes.capture_from = capture.from; attributes.capture_to = capture.to; }
    }
    if (semantics.kind === 'collective') {
      const async = args.find(a => /^async_op\s*=/.test(a));
      attributes.blocking = !!api?.startsWith('torch.') && (!async || keywords.async_op === false) && positional.length <= (semantics.inputArg === 1 ? 2 : 1);
      attributes.explicit_group = args.some(a => /^group\s*=/.test(a)) || positional.length > (semantics.inputArg === 1 ? 2 : 1);
      if (semantics.op === 'reshard') attributes.sharding = positional[1] ?? '';
    }
    if (semantics.op === 'jit_boundary') attributes.recreates_callable = /\blambda\b/.test(positional[0] ?? '') || !!positional[0] && !simple.test(positional[0]);
    if (semantics.op === 'device_to_host' && input?.device === 'cpu') continue;
    const op = v.add(n, semantics.kind, semantics.op, attributes, inputs, outputs, semantics.op !== 'device_to_host' || input?.device === 'gpu');
    if (semantics.op === 'device_to_host') v.add(n, 'synchronization', 'host_sync', { ...attributes, paired: op.id, conditional_device: input?.device !== 'gpu' }, inputs, [], input?.device === 'gpu');
    if (semantics.tensor) {
      const supplied = assigned ? v.context.enrichment?.tensors?.[assigned] : undefined;
      const shapeArg = positional[0] ? literal(positional[0]) : undefined;
      const shape = supplied ? supplied.shape : semantics.op === 'allocate' && Array.isArray(shapeArg) && shapeArg.every(x => typeof x === 'number') ? shapeArg as number[] : semantics.kind === 'compute' && semantics.op === 'elementwise' ? input?.shape : undefined;
      const dtype = args.find(a => /^dtype\s*=/.test(a))?.split('=')[1]?.trim();
      const explicitBytes = dtype ? ({ float16: 2, bfloat16: 2, float32: 4, float64: 8, int32: 4, int64: 8, int8: 1, uint8: 1 } as Record<string, number>)[dtype.split('.').pop()!] : undefined;
      const deviceArg = keywords.device ?? (api === 'tensor.to' ? literal(positional[0] ?? '') : undefined);
      const device = supplied?.device ?? (api === 'tensor.cuda' || typeof deviceArg === 'string' && /^(cuda|gpu)(:|$)/.test(deviceArg) ? 'gpu' : deviceArg === 'cpu' ? 'cpu' : input?.device ?? (api?.startsWith('torch.') && semantics.op === 'allocate' ? 'unknown' : 'unknown'));
      const fact: TensorFact = { name: assigned || `expr:${n.from}`, region: v.region(n), shape, device, dtype: dtype?.split('.').pop() ?? input?.dtype, dtypeBytes: supplied ? supplied.dtype_bytes : explicitBytes ?? input?.dtypeBytes, evidenceId: op.evidence.id };
      expressions.set(n.from, fact);
      if (assigned) { bindings.push({ name: assigned, region: fact.region, from: n.to, assignment: assignment?.from, fact }); v.ir.tensors.push(fact); }
    }
  }
  // Tensor-dependent conditions are represented only for names with tensor provenance.
  for (const n of assignments) {
    const name = lhs(n);
    if (name && !v.ir.operations.some(o => o.outputs.includes(name) && Number(o.attributes.source_order) >= n.from && Number(o.attributes.source_end) <= n.to)) v.add(n, 'control', 'conditional_execution', { definition: true, unknown_value: true }, [], [name]);
  }
  for (const n of v.nodes('IfStatement')) {
    const body = n.getChild('Body'), header = p.masked.slice(n.from, body?.from ?? n.to);
    const names = [...header.matchAll(/\b(\w+)\.(shape|size|item)\b/g)];
    for (const m of names) if (lookup(m[1]!, n)) v.add(n, 'control', m[2] === 'item' ? 'conditional_execution' : 'dynamic_shape', { tensor: m[1]!, tensor_value: m[2] === 'item' }, [m[1]!]);
  }
}
