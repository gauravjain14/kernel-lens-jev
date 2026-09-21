import test from 'node:test';
import assert from 'node:assert/strict';
import { BlockReader, contextHash } from '../src/core/blocks';
import { analyzeSystems } from '../src/core/systems/analyze';
import { systemsPayload, systemsReport } from '../src/core/systems/evaluate';
import { hardwareProfiles, hardwareReview } from '../src/core/systems/hardware';
import { reviewPlans, reviewedFindings } from '../src/core/systems/review';
import { kernelPlans } from '../src/core/systems/kernel-review';
import { releaseCases } from './release-cases';
const reader=new BlockReader();
const context=(source:string,file='case.cu',cursorLine=0)=>reader.read({source,file,language:file.endsWith('.cu')?'cpp':'python',cursorLine,assessmentScope:'function'});

test('release kernels retain the complete function and cache identity across every body line',()=>{
  for(const item of releaseCases.slice(0,2)) {
    const initial=context(item.source,item.file,item.cursor), hash=contextHash(initial);
    for(let line=initial.unit.startLine-1;line<initial.unit.endLine;line++) {
      const c=context(item.source,item.file,line);
      assert.equal(contextHash(c),hash,`${item.name} L${line+1}`);
    }
    const a=analyzeSystems(initial);
    assert.ok(Object.keys(systemsPayload(initial,a).questions).some(id=>id==='review.cuda-reuse'));
    if(item.name.startsWith('rms')) {
      const f=a.findings.find(f=>f.id.startsWith('cuda.serial_reduction'))!;
      assert.ok(f);assert.equal(f.confidence,'high');assert.equal(f.assessment,'possible_issue');
      assert.match(f.evidence.source!,/threadIdx.x == 0/);assert.ok(f.next_check);assert.deepEqual(f.requires,['none']);
      assert.ok(a.ir.tensors.some(t=>t.name==='arr_cpy'&&t.memory==='shared'));
    } else assert.ok(!a.findings.some(f=>f.id.startsWith('cuda.global_accumulation')));
  }
});
test('scope promotion excludes neighboring functions and handles an unfinished edit',()=>{
  const source='__global__ void first(float* out){out[0]=1;}\n__global__ void second(float* out){\nif(threadIdx.x==0){\nout[0]=2;\n}\n}';
  const c=context(source,'scope.cu',3);assert.match(c.code,/second/);assert.doesNotMatch(c.code,/first/);
  assert.equal(context('__global__ void incomplete(){\nif(threadIdx.x==0){out[0]=1;}','scope.cu',0).unit.ready,false);
});
test('all release paths receive fixed dimensions even without a deterministic concern',()=>{
  for(const item of releaseCases) {
    const c=context(item.source,item.file,item.cursor), a=analyzeSystems(c);
    const questions=systemsPayload(c,a).questions;
    assert.equal(a.ir.coverage.complete,true,item.name);
    assert.ok(Object.keys(questions).some(id=>id.startsWith('review.')),item.name);
    assert.ok(Object.keys(questions).length<=36);
    assert.equal(systemsPayload(c,a).state.hardware.profile?.id,'b200');
  }
  for(const source of ['import jax\nimport jax.numpy as jnp\ny=jnp.sin(x)', 'from vllm import LLM\nengine=LLM(model="x")','from sglang.srt.server_args import ServerArgs\nargs=ServerArgs(model_path="x")']) {
    const c=context(source,'framework.py');assert.ok(reviewPlans(analyzeSystems(c)).length>=3);
  }
});
test('model review cannot invent an anchor, claim likely or produce measured impact',()=>{
  const c=context(releaseCases[1]!.source),a=analyzeSystems(c),p=kernelPlans(a).find(p=>p.id==='review.cuda-reuse')!;
  const answer=(choice:string)=>({[p.id]:{type:'choice' as const,choice,probabilities:{[choice]:1}}});
  for(const choice of ['issue_E999','likely','unknown','not_applicable'])assert.equal(reviewedFindings(a,answer(choice),[],.75).length,0);
  const f=reviewedFindings(a,answer('global_reload'),[],.75)[0]!;
  assert.equal(f.assessment,'possible_issue');assert.equal(f.confidence,'medium');assert.equal(f.evidence.source,p.buckets.find(b=>b.bucket.key==='global_reload')!.evidence[0]!.source);
  assert.ok(f.next_check);assert.ok(f.requires.length);assert.equal(f.runtime_impact_measured,false);
  assert.equal(reviewedFindings(a,{[p.id]:{type:'choice',choice:'global_reload',probabilities:{global_reload:.6,unknown:.4}}},[],.75).length,0);
});
test('shared reduction cannot be labeled repeated global output updates by the model',()=>{
  const c=context(releaseCases[0]!.source), a=analyzeSystems(c);
  const findings=reviewedFindings(a,{'review.cuda-accumulator':{type:'choice',choice:'global_rmw',probabilities:{global_rmw:1}}},[],.75);
  assert.equal(findings.length,0);
});
test('hardware selection changes cache and payload without fabricating runtime metrics',()=>{
  const c=context(releaseCases[1]!.source),a=analyzeSystems(c),before=contextHash(c);
  c.hardwareProfile='h100';assert.notEqual(contextHash(c),before);
  const report=systemsReport(c,a);assert.equal(report.hardware?.profile?.name,'H100 SXM 80GB');
  assert.ok(report.findings!.every(f=>!f.runtime_impact_measured));assert.equal(hardwareProfiles.b200.denseBf16Tflops,2250);
  c.hardwareProfile='unspecified';assert.equal(hardwareReview(c,a.ir).profile,undefined);
});
test('DataLoader iterator recreation requires a real loader, an unshadowed builtin and a repeated path',()=>{
  const item=releaseCases[3]!, analyze=(source:string)=>analyzeSystems(context(source,'loader.py',source.split('\n').findIndex(l=>l.startsWith('def '))));
  const f=analyze(item.source).findings.find(f=>f.id.startsWith('training.iterator_recreated'))!;
  assert.ok(f);assert.equal(f.assessment,'possible_issue');assert.equal(f.confidence,'high');assert.match(f.evidence.source!,/next\(iter\(loader\)\)/);
  assert.ok(f.next_check);assert.deepEqual(f.requires,['none']);
  for(const source of [item.source.replace('next(iter(loader))','next(iterator)'),item.source.replace('def train(model, dataset, optimizer):','def train(model, dataset, optimizer, next):'),item.source.replace('DataLoader(dataset, batch_size=32)','custom_loader(dataset)'),item.source.replace('    for step in range(100):','    loader = other\n    for step in range(100):')])
    assert.ok(!analyze(source).findings.some(f=>f.id.startsWith('training.iterator_recreated')));
  assert.ok(!analyze('from torch.utils.data import DataLoader\nloader=DataLoader(data)\nx=next(iter(loader))').findings.some(f=>f.id.startsWith('training.iterator_recreated')));
});
test('roofline uses known GPU FP32 shapes and suppresses unknown, CPU, mismatched and integer shapes',()=>{
  const source=(device='cuda',dtype='float32',right=32)=>`import torch\na=torch.ones((16,32),device="${device}",dtype=torch.${dtype})\nb=torch.ones((${right},64),device="${device}",dtype=torch.${dtype})\nc=torch.matmul(a,b)`;
  const c=context(source(),'matrix.py'),r=hardwareReview(c,analyzeSystems(c).ir);
  assert.equal(r.estimates.length,1);assert.equal(r.estimates[0]!.flops,2*16*32*64);
  assert.equal(r.estimates[0]!.minimumTensorBytes,(16*32+32*64+16*64)*4);
  assert.equal(r.estimates[0]!.fp32Ridge,75/8);
  for(const code of [source('cpu'),source('cuda','int32'),source('cuda','float32',33),'import torch\nc=torch.matmul(a,b)',source()+'\na=unknown\nc=torch.matmul(a,b)']) {
    const cc=context(code,'matrix.py'),rr=hardwareReview(cc,analyzeSystems(cc).ir);
    assert.ok(rr.estimates.length===(code.includes('a=unknown')?1:0));
  }
  c.enrichment={hardware:{name:'custom'}};assert.equal(hardwareReview(c,analyzeSystems(c).ir).estimates.length,0);
});
