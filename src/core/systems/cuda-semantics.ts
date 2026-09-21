import { createHash } from 'node:crypto';
import { ancestor, functionName, type ParsedSource } from '../syntax';
import type { Attribute, KernelHelper, SystemsOp } from './types';

type Effect = { kind: SystemsOp['kind']; op: SystemsOp['op']; attributes: Record<string, Attribute> };
const effect = (kind: Effect['kind'], op: Effect['op'], attributes: Effect['attributes']): Effect => ({kind,op,attributes});

/** Read only assembly template literals, never comments, constraints or variable names. */
export function asmInstructions(source: string): string[] {
  const start=source.indexOf('('); if(start<0)return [];
  let rest=source.slice(start+1).trimStart(), template='';
  while(rest.startsWith('"')) {
    const m=/^"((?:\\.|[^"\\])*)"/.exec(rest); if(!m)break;
    template+=m[1]!.replace(/\\n/g,'\n').replace(/\\t/g,' ').replace(/\\"/g,'"');rest=rest.slice(m[0].length).trimStart();
  }
  const raw=/^R"([^ (\\\t\r\n]{0,16})\(([\s\S]*)\)\1"/.exec(rest); if(raw)template+=raw[2];
  template=template.replace(/\/\*[\s\S]*?\*\//g,' ').replace(/\/\/[^\n]*/g,'');
  return [...new Set([...template.matchAll(/\b(?:tcgen05\.[\w.:]+|wgmma\.[\w.:]+|mma\.sync[\w.:]*|cp\.async[\w.:]*|mbarrier\.[\w.:]+|ldmatrix[\w.:]*|bar\.sync[\w.:]*|shfl\.sync[\w.:]*)/g)].map(m=>m[0]))];
}

export function cudaEffects(name: string, source: string): Effect[] {
  if (/^(?:asm|__asm__|__asm)$/.test(name)) return asmInstructions(source).flatMap(instruction=>{
    const a={instruction,opaque_memory:true};
    if(/^tcgen05\.mma/.test(instruction))return [effect('compute','gemm',{...a,matrix_family:'tcgen05',async:true,issuer:'single_thread',memory:'tensor'})];
    if(/^wgmma\.mma_async/.test(instruction))return [effect('compute','gemm',{...a,matrix_family:'wgmma',async:true,issuer:'warpgroup'})];
    if(/^mma\.sync/.test(instruction))return [effect('compute','gemm',{...a,matrix_family:'mma',issuer:'warp'})];
    if(/^cp\.async\.(?:commit_group|wait)/.test(instruction))return [effect('synchronization','event_wait',{...a,protocol:'async_copy',action:instruction.includes('commit')?'commit':'wait'})];
    if(/^cp\.async/.test(instruction))return [effect('memory','copy',{...a,async:true,transfer:instruction.includes('bulk')?'tma':'cp_async',from_memory:'global',to_memory:'shared'})];
    if(/^tcgen05\.(?:ld|st)\./.test(instruction))return [effect('memory',instruction.startsWith('tcgen05.ld')?'load':'store',{...a,memory:'tensor',vector:true})];
    if(/^tcgen05\.(?:alloc|dealloc|relinquish)/.test(instruction))return [effect('memory','allocate',{...a,memory:'tensor',action:instruction.split('.')[1]!})];
    if(/^tcgen05\.(?:wait|commit|fence)/.test(instruction)||/^wgmma\.(?:wait|commit|fence)/.test(instruction)||/^mbarrier\./.test(instruction))return [effect('synchronization','event_wait',{...a,protocol:instruction.startsWith('mbarrier')?'mbarrier':'matrix',action:/wait/.test(instruction)?'wait':/commit/.test(instruction)?'commit':/fence/.test(instruction)?'fence':'arrive_or_init'})];
    if(/^ldmatrix/.test(instruction))return [effect('memory','load',{...a,memory:'shared',matrix_load:true})];
    if(/^bar\.sync/.test(instruction))return [effect('synchronization','block_barrier',a)];
    if(/^shfl\.sync/.test(instruction))return [effect('compute','reduction',{...a,algorithm:'warp_shuffle'})];
    return [];
  });
  if(name==='__pipeline_memcpy_async'||/^(?:cuda|cooperative_groups)::memcpy_async$/.test(name))return [effect('memory','copy',{api:name,async:true,transfer:'cp_async_api',from_memory:'global',to_memory:'shared'})];
  if(name==='__pipeline_commit'||name==='__pipeline_wait_prior')return [effect('synchronization','event_wait',{api:name,protocol:'async_copy',action:name==='__pipeline_commit'?'commit':'wait'})];
  if(/^(?:nvcuda::)?wmma::mma_sync$/.test(name))return [effect('compute','gemm',{api:name,matrix_family:'wmma',issuer:'warp'})];
  return [];
}

/** Resolve unique same-file definitions only. No semantics inferred from helper names. */
export function kernelHelpers(p: ParsedSource, file: string, from: number, to: number): KernelHelper[] {
  if(p.python)return [];
  const functions=p.nodes.filter(n=>n.name==='FunctionDefinition'), calls=p.nodes.filter(n=>n.name==='CallExpression');
  const names=calls.filter(n=>n.from>=from&&n.to<=to).map(n=>n.firstChild?p.text(n.firstChild).replace(/\s+/g,''):'');
  const memo=new Map<string,KernelHelper>();let budget=7000;
  const resolve=(name:string,depth:number):KernelHelper|undefined=>{
    if(depth>3||memo.has(name))return memo.get(name);
    const defs=functions.filter(n=>functionName(p,n)===name&&!(n.from>=from&&n.to<=to));
    if(defs.length!==1||!/^[A-Za-z_]\w*$/.test(name))return;
    const fn=defs[0]!,h:KernelHelper={name,effects:[],partial:false};memo.set(name,h);
    for(const call of calls.filter(n=>n.from>=fn.from&&n.to<=fn.to)) {
      const callee=call.firstChild?p.text(call.firstChild).replace(/\s+/g,''):'',source=p.text(call);
      const effects=cudaEffects(callee,source);
      if(!effects.length) {const nested=resolve(callee,depth+1);if(nested)h.effects.push(...nested.effects);h.partial=true;continue;}
      if(source.length>1800) {h.partial=true;continue;}
      const location={file,startLine:p.line(call.from),endLine:p.line(call.to-1)};
      const evidence={id:'helper_'+createHash('sha256').update(JSON.stringify([file,location,source])).digest('hex').slice(0,12),source,location,explanation:'Instruction in resolved helper '+name,origin:'source' as const,direct:true};
      h.effects.push(...effects.map(e=>({...e,attributes:{...e.attributes,helper:name,conditional:!!ancestor(call,['IfStatement','WhileStatement','ForStatement'])},evidence})));
    }
    h.effects=[...new Map(h.effects.map(e=>[e.evidence.id+e.op+e.attributes.instruction,e])).values()];return h;
  };
  const result:KernelHelper[]=[];
  for(const name of new Set(names)) {
    const h=resolve(name,0);if(!h?.effects.length)continue;
    const size=JSON.stringify(h).length;if(size>budget)continue;
    budget-=size;result.push(h);
  }
  return result;
}
