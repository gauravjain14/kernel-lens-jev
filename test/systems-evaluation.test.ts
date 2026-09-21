import test from 'node:test';
import assert from 'node:assert/strict';
import { BlockReader, contextHash } from '../src/core/blocks';
import { analyzeSystems } from '../src/core/systems/analyze';
import { assessSystems, systemsPayload, systemsReport } from '../src/core/systems/evaluate';
import { parseEnrichment } from '../src/core/systems/enrichment';
import { sourceRevision } from '../src/core/systems/source';
import { insightPresentation, performanceAssessments } from '../src/core/presentation';
import type { Choice } from '../src/live-types';
const reader = new BlockReader();
const serial = '__global__ void reduce(float* out) {\n__shared__ float s[32];\nif(threadIdx.x==0) {float sum=0; for(int i=0;i<32;i++) sum+=s[i]; out[0]=sum;}\nfor(int j=0;j<32;j++)out[threadIdx.x]+=j;\n__syncthreads();\n}';
const shuffle = '__global__ void reduce(float* out) {\n__shared__ float s[32];\nfloat sum=s[threadIdx.x]; for(int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);\nif(threadIdx.x==0)out[0]=sum;\n}';
const context = (source = serial, file = 'test.cu') => reader.read({ file, language: file.endsWith('.cu') ? 'cpp' : 'python', source, cursorLine: 0 });
const answer = (choice: string, p = .98): Choice => ({ type: 'choice', choice, probabilities: { retain: choice === 'retain' ? p : (1-p)/3, likely: choice === 'likely' ? p : (1-p)/3, dismiss: choice === 'dismiss' ? p : (1-p)/3, unknown: choice === 'unknown' ? p : (1-p)/3 } });
test('one bounded Jev call consumes real IR evidence and cannot invent fields', async () => {
  const c=context(), a=analyzeSystems(c); let calls=0;
  const report=await assessSystems(c,'test-key',new AbortController().signal,.75,async (_url, options)=>{
    calls++; const body=JSON.parse(String(options?.body));
    assert.ok(body.state.systems.operations.length); assert.equal(body.model,'typesafe-ai/jev');
    assert.ok(Object.keys(body.questions).every(id=>id.startsWith('review.')||a.candidates.some(c=>c.finding.id===id)));
    return Response.json({answers:Object.fromEntries(Object.entries(body.questions).map(([id,q]:[string,any])=>{const choice=id.startsWith('review.')?'unknown':'retain';return [id,{type:'choice',choice,probabilities:Object.fromEntries(Object.keys(q.criteria).map(k=>[k,k===choice?1:0])),explanation:'INVENTED: 80% slower',line:999}];})),usage:{inputTokens:321}});
  });
  assert.equal(calls,1); assert.equal(report.inputTokens,321); assert.ok(report.findings?.length);
  assert.ok(!JSON.stringify(report).includes('INVENTED')); assert.ok(report.findings!.every(f=>f.evidence.source && !f.runtime_impact_measured));
  assert.equal(report.detailStatus,'ready'); assert.ok(report.findings!.some(f=>f.assessment==='possible_issue'&&f.confidence==='high'));
});
test('Jev dismissal removes the concern, while uncertainty never becomes a good finding', ()=>{
  const c=context(), a=analyzeSystems(c), id=a.candidates.find(c=>c.modelMayAssess)!.finding.id;
  assert.ok(!systemsReport(c,a,{[id]:answer('dismiss')}).findings!.some(f=>f.id===id));
  const unknown=systemsReport(c,a,{[id]:answer('unknown')}).findings!.find(f=>f.id===id)!;
  assert.equal(unknown.assessment,'unknown'); assert.equal(unknown.confidence,'high'); assert.ok(unknown.evidence.source);
  const weak=systemsReport(c,a,{[id]:answer('likely',.4)}).findings!.find(f=>f.id===id)!;
  assert.equal(weak.assessment,'unknown');
  const split=systemsReport(c,a,{[id]:{type:'choice',choice:'retain',probabilities:{retain:.5,likely:.4,unknown:.08,dismiss:.02}}}).findings!.find(f=>f.id===id)!;
  assert.equal(split.assessment,'possible_issue'); assert.equal(split.model_concern_probability,.9); assert.equal(split.model_probability,.5);
});
test('high Jev probability cannot upgrade inferred or shape-dependent evidence', ()=>{
  const c=context('from vllm import LLM\nengine=LLM(model="x",tensor_parallel_size=8,max_num_seqs=2)','serve.py');
  const a=analyzeSystems(c), id=a.candidates[0]!.finding.id;
  const f=systemsReport(c,a,{[id]:answer('likely',1)}).findings![0]!;
  assert.equal(f.assessment,'possible_issue'); assert.equal(f.confidence,'medium'); assert.deepEqual(f.requires,['shapes','hardware']);
});
test('direct positives survive without a verdict; unsupported frameworks make no request', async ()=>{
  let calls=0; const fetcher:typeof fetch=async()=>{calls++;throw Error('unnecessary model call');};
  const goodContext=context(shuffle),good=systemsReport(goodContext,analyzeSystems(goodContext));
  assert.ok(good.findings!.some(f=>f.assessment==='good'));
  const empty=await assessSystems(context('def f(x):\n    return custom(x)','x.py'),'key',new AbortController().signal,.75,fetcher);
  assert.equal(empty.findings!.length,0); assert.ok(empty.coverage?.limitations.length); assert.equal(calls,0);
});
test('improvements need a positive structure in the same scope, never a disappeared warning', ()=>{
  const c=context(), a=analyzeSystems(c), bad=systemsReport(c,a);
  const goodContext=context(shuffle), good=systemsReport(goodContext,analyzeSystems(goodContext));
  assert.ok(insightPresentation(good,bad).improvements.length);
  const empty={...good,findings:[]}; assert.equal(insightPresentation(empty,bad).improvements.length,0);
  good.scope.identity='different'; assert.equal(insightPresentation(good,bad).improvements.length,0);
});
test('enrichment quantities retain compiler/runtime provenance and do not measure other findings', ()=>{
  const c=context(); c.enrichment=parseEnrichment({compiler:{registers_per_thread:64,spill_loads:16},runtime:{latency_ms:1.2,cache_hit_percent:80}});
  const result=analyzeSystems(c);
  const compiler=result.findings.find(f=>f.id.startsWith('evidence.compiler'))!;
  assert.equal(compiler.assessment,'likely_issue'); assert.equal(compiler.confidence,'high'); assert.equal(compiler.evidence_level,2); assert.deepEqual(compiler.requires,['runtime_measurement']); assert.ok(compiler.next_check);
  const measured=result.findings.find(f=>f.id.startsWith('evidence.runtime'))!;
  assert.equal(measured.runtime_impact_measured,true); assert.equal(measured.evidence_level,4); assert.ok(measured.evidence.source);
  assert.equal(result.findings.find(f=>f.id.startsWith('cuda.serial'))!.runtime_impact_measured,false);
  assert.throws(()=>parseEnrichment({runtime:{stall_percent:101}}));
  assert.throws(()=>parseEnrichment({compiler:{registers_per_thread:'128'}}));
  assert.throws(()=>parseEnrichment({runtime:{invented_metric:5}}));
  assert.throws(()=>parseEnrichment({tensors:{x:{shape:[1,-2],dtype_bytes:4}}}));
});
test('source revision binding ignores imported facts but caches include them', ()=>{
  const c=context(), before=sourceRevision(c), cache=contextHash(c);
  c.enrichment={runtime:{latency_ms:1}};
  assert.equal(sourceRevision(c),before); assert.notEqual(contextHash(c),cache);
  c.code+='\nL6: // revised'; assert.notEqual(sourceRevision(c),before);
});
test('known shapes estimate bytes and theoretical traffic without time conversion', ()=>{
  const c=context('import torch\nimport torch.distributed as dist\ndist.init_process_group(world_size=8,rank=0)\nx=torch.ones((4,8),device="cuda",dtype=torch.float32)\ndist.all_reduce(x)','collective.py');
  const f=analyzeSystems(c).findings.find(f=>f.id.startsWith('distributed.all_reduce'))!;
  assert.equal(f.evidence_level,1); assert.equal(f.quantities![0]!.value,128);
  assert.equal(f.quantities![1]!.value,224); assert.ok(f.quantities!.every(q=>q.unit==='bytes'));
  assert.equal(f.runtime_impact_measured,false); assert.ok(f.assumptions.some(s=>s.includes('Ring')));
});
test('API aliases work; reassignments, same-named functions and shadowed parameters stay quiet', ()=>{
  const yes=context('from torch.distributed import all_reduce as reduce\nreduce(x)','alias.py');
  assert.ok(analyzeSystems(yes).findings.some(f=>f.id.startsWith('distributed.all_reduce')));
  for(const source of ['import torch.distributed as dist\ndef dist(): pass\ndist.all_reduce(x)',
    'import torch\nx=torch.ones((1,),device="cuda")\nx=Other()\nfor i in range(8):\n    x.item()',
    'import jax\nimport jax.numpy as jnp\ny=jax.lax.all_gather(x,"tp")\ny=other\nz=jnp.array(y,copy=True)']) {
    assert.ok(!analyzeSystems(context(source,'shadow.py')).findings.some(f=>f.id.startsWith('graph.host_sync')||f.id.startsWith('distributed.gather_materialization')));
  }
});
test('unknown calls invalidate redundant-load and output-invariance hypotheses', ()=>{
  const c=context('__global__ void f(float* out,const float* input) {\nfloat a=input[threadIdx.x]; mutate(input); float b=input[threadIdx.x];\nfor(int i=0;i<32;i++){int idx=i;out[idx]+=a;}\n}');
  assert.ok(!analyzeSystems(c).findings.some(f=>f.id.startsWith('cuda.redundant_load')||f.id.startsWith('cuda.global_accumulation')));
});
test('nested function definitions do not inherit the loop executing their declaration', ()=>{
  const c=context('import torch.distributed as dist\nfor i in range(10):\n    def f(x):\n        dist.all_reduce(x)','nested.py');
  const a=analyzeSystems(c), collective=a.ir.operations.find(o=>o.op==='all_reduce')!;
  assert.ok(collective); assert.equal(collective.controls.length,0);
});
test('truncated context never produces an absence-based good finding', ()=>{
  const c=context('import jax\nimport jax.numpy as jnp\n@jax.jit\ndef f(x):\n    return jnp.sin(x)','truncated.py');c.truncated=true;
  assert.ok(!analyzeSystems(c).findings.some(f=>f.id.startsWith('graph.no_host_reads')));
});
test('literal dead branches and monkey-patched API names do not produce performance concerns',()=>{
  for(const source of ['import torch.distributed as dist\ndist.all_reduce = custom\ndist.all_reduce(x)',
    'import torch\nx=torch.ones((1,),device="cuda")\nx.item=lambda:1\nfor i in range(8):\n    x.item()',
    'import torch.distributed as dist\nif False:\n    dist.all_reduce(x)']) {
    assert.ok(!analyzeSystems(context(source,'dead.py')).findings.some(f=>f.assessment.includes('issue')));
  }
  const dead=context('__global__ void f(float* out) {__shared__ float s[32]; if(false) {if(threadIdx.x==0){float sum=0;for(int i=0;i<32;i++)sum+=s[i];out[0]=sum;}}}');
  assert.ok(!analyzeSystems(dead).findings.some(f=>f.id.startsWith('cuda.serial_reduction')));
});
test('materialization, fusion, input pipeline and Triton findings have concrete controls', ()=>{
  const cases=[
    ['graph.materialization','import torch\ny=torch.sin(x)\nz=torch.clone(y)','import torch\ny=torch.sin(x)\nz=torch.clone(other)'],
    ['graph.fusion_chain','import torch\na=torch.matmul(x,y)\nb=torch.sin(a)\nc=torch.cos(b)','import torch\na=torch.matmul(x,y)\nb=torch.sin(other)\nc=torch.cos(b)'],
    ['graph.growing_materialization','import torch\nfor i in range(8):\n    cache=torch.cat([cache,chunk])','import torch\nfor i in range(8):\n    result=torch.cat([left,right])'],
    ['training.input_pipeline','from torch.utils.data import DataLoader\nloader=DataLoader(data,num_workers=0)','from torch.utils.data import DataLoader\nloader=DataLoader(data,num_workers=4)'],
    ['training.persistent_workers','from torch.utils.data import DataLoader\nloader=DataLoader(data,num_workers=4,persistent_workers=True)','from torch.utils.data import DataLoader\nloader=DataLoader(data,num_workers=0,persistent_workers=True)'],
    ['triton.compute_primitive','import triton.language as tl\ny=tl.dot(a,b)','class Local: pass\ntl=Local()\ny=tl.dot(a,b)'],
  ];
  for(const [rule,positive,negative] of cases) {
    const f=analyzeSystems(context(positive!,'case.py')).findings.find(f=>f.id.startsWith(rule!)); assert.ok(f,rule);
    assert.ok(['possible_issue','good'].includes(f.assessment)); assert.equal(f.confidence,'high'); assert.ok(f.evidence.source && positive!.includes(f.evidence.source)); assert.ok(f.next_check);assert.ok(f.requires.length);
    assert.ok(!analyzeSystems(context(negative!,'case.py')).findings.some(f=>f.id.startsWith(rule!)),rule);
  }
});
test('request candidates have a fixed vocabulary and zero-retention stays explicit', ()=>{
  const c=context(), a=analyzeSystems(c), p=systemsPayload(c,a,{zeroDataRetention:true});
  assert.equal(p.providerOptions?.gateway.zeroDataRetention,true);
  assert.ok(Object.entries(p.questions).every(([id,q])=>id.startsWith('review.')?Object.keys(q.criteria).every(k=>/^[a-z][a-z_0-9]*$/.test(k)):Object.keys(q.criteria).join(',')==='retain,likely,dismiss,unknown'));
  assert.ok(Object.keys(p.questions).length<=36);
});
test('CUDA graph capture distinguishes explicit static inputs from a dynamic host-read path', ()=>{
  const prefix='import torch\nx=torch.ones((4,8),device="cuda",dtype=torch.float32)\ng=torch.cuda.CUDAGraph()\n';
  const good=analyzeSystems(context(prefix+'with torch.cuda.graph(g):\n    y=torch.sin(x)','capture.py'));
  const f=good.findings.find(f=>f.id.startsWith('graph.capture_static'))!;
  assert.ok(f); assert.equal(f.assessment,'good'); assert.equal(f.confidence,'high'); assert.ok(f.evidence.source); assert.deepEqual(f.requires,['runtime_measurement']); assert.ok(f.next_check);
  const bad=analyzeSystems(context(prefix+'with torch.cuda.graph(g):\n    count=x.item()\n    y=torch.sin(x)','capture.py'));
  assert.ok(!bad.findings.some(f=>f.id.startsWith('graph.capture_static')));
  const issue=bad.findings.find(f=>f.id.startsWith('graph.capture_host_read'))!;
  assert.ok(issue); assert.equal(issue.assessment,'possible_issue'); assert.equal(issue.confidence,'high'); assert.deepEqual(issue.requires,['none']); assert.ok(issue.next_check); assert.match(issue.evidence.source!,/x.item/);
});

test('CUDA memory claims stop when the address changes or a loop has only one iteration', ()=>{
  for (const body of [
    'int idx=0; float a=input[idx]; idx=1; float b=input[idx];',
    'float a=input[0]; input+=32; float b=input[0];',
    'for(int i=0;i<32;i++){out[0]+=1;out+=1;}',
    'for(int i=0;i<1;i++)out[0]+=1;',
  ]) {
    const c=context(`__global__ void f(float* out,const float* input){${body}}`);
    assert.ok(!analyzeSystems(c).findings.some(f=>f.id.startsWith('cuda.redundant_load')||f.id.startsWith('cuda.global_accumulation')),body);
  }
});

test('supplied launch width refines only the execution assumption', ()=>{
  const c=context('__global__ void f(float* out){__shared__ float s[256];for(int stride=blockDim.x/2;stride>0;stride/=2){if(threadIdx.x<stride)s[threadIdx.x]+=s[threadIdx.x+stride];__syncthreads();}}');
  for (const width of [256,192]) {
    c.enrichment={launch:{block:[width,1,1]}};
    const result=analyzeSystems(c), f=result.findings.find(f=>f.id.startsWith('cuda.power_of_two'))!;
    assert.ok(f);assert.equal(f.section,'assumptions');assert.equal(f.assessment,width===256?'good':'unknown');assert.equal(f.confidence,'high');assert.equal(f.evidence_level,1);
    assert.ok(f.evidence.source);assert.ok(f.next_check);assert.deepEqual(f.requires,['none']);assert.equal(f.runtime_impact_measured,false);
    assert.ok(result.findings.some(f=>f.id.startsWith('cuda.reduction_barriers')&&f.assessment==='possible_issue'));
  }
});

test('selected lenses restrict findings without inventing a framework', ()=>{
  const c=context('import torch\nimport torch.distributed as dist\nx=torch.ones((1,),device="cuda")\nfor i in range(4):\n    dist.all_reduce(x)\n    value=x.item()','mixed.py');
  const all=analyzeSystems(c).findings;assert.ok(all.some(f=>f.scope==='graph'));assert.ok(all.some(f=>f.scope==='distributed'));
  const distributed=analyzeSystems(c,'distributed').findings;assert.ok(distributed.length);assert.ok(distributed.every(f=>f.scope==='distributed'));
  assert.equal(analyzeSystems(c,'cuda').findings.length,0);assert.equal(analyzeSystems(c,'serving').findings.length,0);
});

test('execution assumptions never become a performance rating or improvement', ()=>{
  const c=context('__global__ void f(float* out){__shared__ float s[256];for(int stride=blockDim.x/2;stride>0;stride/=2){if(threadIdx.x<stride)s[threadIdx.x]+=s[threadIdx.x+stride];}}');
  c.enrichment={launch:{block:[256]}};
  const report=systemsReport(c,analyzeSystems(c));
  assert.ok(report.findings!.some(f=>f.section==='assumptions'&&f.assessment==='good'));
  assert.ok(performanceAssessments(report).every(a=>report.findings!.find(f=>f.id===a.id)!.section==='performance'));
  assert.equal(insightPresentation(report).improvements.length,0);
});

test('resource and serving policy findings require explicit source evidence', ()=>{
  const cases=[
    ['cuda.local_storage','__global__ void f(float* out){float local[256];out[0]=local[0];}','__global__ void f(float* out){float local[4];out[0]=local[0];}','test.cu','possible_issue',['compiler_output']],
    ['serving.prefix_cache','from vllm import LLM\nengine=LLM(model="x",enable_prefix_caching=True)','from vllm import LLM\nengine=LLM(model="x",enable_prefix_caching=False)','test.py','good',['runtime_measurement']],
    ['serving.pd_transfer','from sglang.srt.server_args import ServerArgs\nargs=ServerArgs(model_path="x",disaggregation_mode="prefill")','from sglang.srt.server_args import ServerArgs\nargs=ServerArgs(model_path="x")','test.py','possible_issue',['shapes','hardware']],
  ] as const;
  for(const [rule,source,negative,file,assessment,requires] of cases) {
    const f=analyzeSystems(context(source,file)).findings.find(f=>f.id.startsWith(rule))!;
    assert.ok(f,rule);assert.equal(f.assessment,assessment);assert.equal(f.confidence,'high');assert.ok(f.evidence.source&&source.includes(f.evidence.source));assert.ok(f.next_check);assert.deepEqual(f.requires,[...requires]);
    assert.ok(!analyzeSystems(context(negative,file)).findings.some(f=>f.id.startsWith(rule)));
  }
});
