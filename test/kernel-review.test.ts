import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {BlockReader,contextHash} from '../src/core/blocks';
import {analyzeSystems} from '../src/core/systems/analyze';
import {systemsPayload,systemsReport} from '../src/core/systems/evaluate';
import {kernelDimensions,kernelFindings,kernelPlans,kernelQuestions} from '../src/core/systems/kernel-review';
import {kernelSummary} from '../src/core/presentation';
import {asmInstructions} from '../src/core/systems/cuda-semantics';
import type {Choice} from '../src/live-types';

const fixture=readFileSync('test/fixtures/phase1_cpasync_2stage.cu','utf8');
const read=(source:string,line=0)=>new BlockReader().read({source,file:'kernel.cu',language:'cpp',cursorLine:line,assessmentScope:'function'});
const choice=(key:string,p=1):Choice=>({type:'choice',choice:key,probabilities:{[key]:p,unknown:1-p}});
const expected:Record<string,string>={
  'tensor-cores':'tcgen05','thread-work':'elected_issue','coalescing':'contiguous','transfer-width':'vector',reuse:'shared_tiles',
  'shared-layout':'swizzled','shared-stages':'cooperative',overlap:'overlapped','pipeline-depth':'double','mma-cadence':'tile_wait',
  'sync-cost':'per_tile','async-protocol':'copy_and_matrix',accumulator:'tensor_memory',resources:'multibuffer',epilogue:'tmem_wait',
  'control-flow':'predicated_edges',intensity:'reuse_rich','shape-contract':'contract',
};
test('Blackwell helpers lower to source-backed MMA, TMEM, copy and wait operations',()=>{
  const c=read(fixture,101),a=analyzeSystems(c),p=systemsPayload(c,a);
  assert.equal(c.unit.ready,true);assert.equal(c.truncated,false);
  assert.equal(a.ir.operations.filter(o=>o.attributes.matrix_family==='tcgen05').length,4);
  for(const name of ['mma_f16','mbar_wait','tcgen05_commit','tmem_alloc'])assert.ok(c.kernelHelpers?.some(h=>h.name===name&&h.effects.length),name);
  for(const h of c.kernelHelpers!)for(const e of h.effects)assert.ok(fixture.includes(e.evidence.source));
  assert.ok(p.state.kernelExecution?.operations.some(o=>o.op==='gemm'));
  assert.ok(p.state.kernelExecution?.operations.some(o=>o.attributes.protocol==='async_copy'&&o.attributes.action==='wait'));
  assert.ok(p.state.kernelExecution?.helperInstructions?.some(e=>e.source.includes('tcgen05.mma')));
  assert.ok(!a.ir.operations.some(o=>o.op==='loop'&&o.attributes.reduction),'loop-counter += is not a reduction');
  assert.ok(JSON.stringify(p).length<75000,'bounded kernel request');
});
test('kernel taxonomy keeps separate, descriptive classification axes without anchor probability splitting',()=>{
  const a=analyzeSystems(read(fixture,101)),q=kernelQuestions(a);
  assert.equal(kernelDimensions.length,24);
  assert.equal(new Set(kernelDimensions.map(d=>d.id)).size,kernelDimensions.length);
  for(const id of ['overlap','pipeline-depth','mma-cadence','async-protocol','resources','epilogue'])assert.ok(q['review.cuda-'+id],id);
  for(const question of Object.values(q))assert.ok(!Object.keys(question.criteria).some(k=>/_E\d/.test(k)));
  assert.ok(!Object.hasOwn(q['review.cuda-tensor-cores']!.criteria,'simt'));
});
test('full kernel classifications retain exact evidence, uncertainty and an actionable next check',()=>{
  const c=read(fixture,101),a=analyzeSystems(c),answers=Object.fromEntries(Object.entries(expected).map(([k,v])=>['review.cuda-'+k,choice(v)]));
  const r=systemsReport(c,a,answers);
  for(const [id,bucket] of Object.entries(expected)) {
    const f=r.findings!.find(f=>f.id==='review.cuda-'+id)!;assert.ok(f,id);assert.equal(f.model_choice,bucket);
    assert.equal(f.assessment,id==='shape-contract'?'unknown':['mma-cadence','sync-cost','resources','epilogue'].includes(id)?'possible_issue':'good');
    assert.ok(['high','medium'].includes(f.confidence));assert.ok(f.evidence.source&&fixture.includes(f.evidence.source));
    assert.ok(f.evidence.location);assert.ok(f.next_check);assert.ok(f.requires.length);assert.equal(f.runtime_impact_measured,false);
  }
  assert.match(kernelSummary(r)!,/Blackwell asynchronous MMA/);assert.match(kernelSummary(r)!,/performance checks/);
  const mma=r.findings!.find(f=>f.id==='review.cuda-tensor-cores')!;
  assert.equal(mma.confidence,'high');assert.ok(mma.supporting_evidence?.some(e=>e.source.includes('tcgen05.mma')));
  assert.match(r.findings!.find(f=>f.id==='review.cuda-mma-cadence')!.evidence.source!,/^mbar_wait/);
  assert.equal(r.findings!.find(f=>f.id==='review.cuda-sync-cost')!.evidence.location!.startLine,276);
  assert.equal(r.dimensions!.length,24);assert.equal(r.dimensions!.find(d=>d.label==='Launch parallelism')?.assessment,'unknown');
});
test('helper names, comments and non-assembly strings cannot establish hardware semantics',()=>{
  const sources=[
    '__global__ void f(){mma_f16(); /* tcgen05.mma */}',
    '__device__ void mma_f16(){}\n__global__ void f(){mma_f16();}',
    '__device__ void mma_f16(){const char* s="tcgen05.mma.cta_group::1";}\n__global__ void f(){mma_f16();}',
    '__device__ void mma_f16(int x){}\n__device__ void mma_f16(float x){}\n__global__ void f(){mma_f16(1);}',
  ];
  for(const s of sources) {
    const a=analyzeSystems(read(s,s.split('\n').length-1));
    assert.ok(!a.ir.operations.some(o=>o.attributes.matrix_family));
    assert.equal(kernelFindings(a,{'review.cuda-tensor-cores':choice('tcgen05'),'review.cuda-epilogue':choice('tmem_wait')},.75).length,0);
  }
  assert.deepEqual(asmInstructions('asm volatile("// tcgen05.mma.fake;\\n nop;" ::: "memory")'),[]);
});
test('helper-only edits invalidate cached classifications',()=>{
  const a='__device__ void compute(){asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32;");}\n__global__ void f(){compute();}';
  const b=a.replace('mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32','tcgen05.mma.cta_group::1.kind::f16');
  assert.notEqual(contextHash(read(a,1)),contextHash(read(b,1)));
});
test('resource facts preserve all dimensions and half width without inventing symbolic sizes',()=>{
  const a=analyzeSystems(read('__global__ void f(){__shared__ half a[2][128][32];__shared__ half b[2][BK][BN];uint32_t r[8];}'));
  const allocations=a.ir.operations.filter(o=>o.op==='allocate');
  assert.equal(allocations.find(o=>o.outputs.includes('a'))!.attributes.bytes,16384);
  const symbolic=allocations.find(o=>o.outputs.includes('b'))!;assert.equal(symbolic.attributes.bytes,undefined);assert.equal(symbolic.attributes.symbolic_bytes,'2 * (2) * (BK) * (BN)');
  const q=kernelQuestions(a);assert.ok(!q['review.cuda-registers'],'eight uint32 temporaries do not establish register pressure');
  const findings=kernelFindings(a,{'review.cuda-registers':choice('arrays'),'review.cuda-parallelism':choice('small_grid')},.75);assert.equal(findings.length,0);
});
test('unsupported model choices and uncertain classes cannot become confident findings',()=>{
  const a=analyzeSystems(read('__global__ void f(float* x){x[threadIdx.x]=1;}'));
  for(const [id,key] of [['tensor-cores','tcgen05'],['accumulator','global_rmw'],['mma-cadence','tile_wait'],['atomics','contended'],['resources','multibuffer']])
    assert.equal(kernelFindings(a,{['review.cuda-'+id]:choice(key!)},.75).length,0,id);
  assert.equal(kernelFindings(a,{'review.cuda-coalescing':choice('contiguous',.6)},.75).length,0);
});
test('late instructions remain selectable when early loops are plentiful',()=>{
  const c=read('__global__ void f(){'+Array.from({length:12},(_,i)=>`for(int i${i}=0;i${i}<2;i${i}++){}`).join('\n')+'\nasm volatile("tcgen05.mma.cta_group::1.kind::f16;");}',2),a=analyzeSystems(c);
  const p=kernelPlans(a).find(p=>p.id==='review.cuda-tensor-cores')!;
  assert.ok(p.buckets.find(b=>b.bucket.key==='tcgen05')!.evidence.some(e=>e.source.includes('tcgen05.mma')));
});
