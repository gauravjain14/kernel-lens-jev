import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeSystems } from '../src/core/systems/analyze';
import { gateFinding } from '../src/core/systems/gate';
import type { CodeContext } from '../src/live-types';
import type { Enrichment, JevFinding } from '../src/core/systems/types';

export function systemsContext(source: string, language = 'python', enrichment?: Enrichment): CodeContext {
  return { file: language === 'python' ? 'example.py' : 'example.cu', language, code: source.split('\n').map((s, i) => `L${i + 1}: ${s}`).join('\n'),
    enclosing: '', preamble: '', references: [], intent: '', hardware: '', truncated: false, characters: source.length, routingKey: 'test', enrichment,
    unit: { name: 'test', kind: 'FunctionDefinition', from: 0, to: source.length, startLine: 1, endLine: source.split('\n').length, ready: true, syntax: [] } };
}
const cuda = (body: string) => `__global__ void reduce(float* out, const float* input) {\n    __shared__ float s[256];\n${body}\n}`;
const run = (source: string, language = 'python', enrichment?: Enrichment) => analyzeSystems(systemsContext(source, language, enrichment));
function finding(source: string, rule: string, assessment: JevFinding['assessment'], confidence: JevFinding['confidence'], language = 'python', enrichment?: Enrichment) {
  const result = run(source, language, enrichment);
  const f = result.findings.find(f => f.id.startsWith(`${rule}:`));
  assert.ok(f, `${rule}: ${JSON.stringify(result.findings)}`);
  assert.equal(f.assessment, assessment); assert.equal(f.confidence, confidence);
  assert.ok(f.evidence.source && source.includes(f.evidence.source), 'exact source evidence');
  assert.ok(f.evidence.location?.startLine); assert.ok(f.evidence.explanation.length > 10);
  assert.ok(f.next_check.length > 10); assert.ok(f.requires.length > 0);
  assert.equal(f.runtime_impact_measured, false);
  assert.ok(f.evidence_ids.length); return f;
}
function absent(source: string, rule: string, language = 'python') { assert.ok(!run(source, language).findings.some(f => f.id.startsWith(`${rule}:`)), rule); }

test('CUDA serial cooperative tail is evidence-backed; a final store alone is quiet', () => {
  const f = finding(cuda('if (threadIdx.x == 0) { float sum=0; for (int i=0;i<32;i++) sum += s[i]; out[0]=sum; }\n__syncthreads();'), 'cuda.serial_reduction', 'possible_issue', 'high', 'cpp');
  assert.deepEqual(f.requires, ['none']); assert.equal(f.scope, 'kernel');
  absent(cuda('float sum=s[threadIdx.x]; if (threadIdx.x == 0) out[0]=sum;'), 'cuda.serial_reduction', 'cpp');
  absent(cuda('for (int i=0;i<32;i++) out[threadIdx.x] += input[threadIdx.x+i*blockDim.x];'), 'cuda.serial_reduction', 'cpp');
});
test('CUDA shuffle reduction is positive, but a standalone shuffle is not a proven reduction', () => {
  finding(cuda('float sum=s[threadIdx.x]; for(int offset=16;offset>0;offset/=2) sum += __shfl_down_sync(0xffffffff,sum,offset);'), 'cuda.warp_reduction', 'good', 'high', 'cpp');
  absent(cuda('float x=__shfl_down_sync(0xffffffff,1.0f,1);'), 'cuda.warp_reduction', 'cpp');
});
test('CUDA barriers in halving reductions are observed; a single post-loop barrier is quiet', () => {
  finding(cuda('for(int stride=blockDim.x/2;stride>0;stride/=2) { if(threadIdx.x<stride) s[threadIdx.x]+=s[threadIdx.x+stride]; __syncthreads(); }'), 'cuda.reduction_barriers', 'possible_issue', 'high', 'cpp');
  absent(cuda('for(int i=0;i<32;i++) s[threadIdx.x]+=1; __syncthreads();'), 'cuda.reduction_barriers', 'cpp');
});
test('CUDA redundant global loads require the same global address and no intervening effects', () => {
  finding(cuda('float x=input[threadIdx.x]; float y=input[threadIdx.x]; out[threadIdx.x]=x+y;'), 'cuda.redundant_load', 'possible_issue', 'high', 'cpp');
  absent(cuda('float x=s[threadIdx.x]; float y=s[threadIdx.x];'), 'cuda.redundant_load', 'cpp');
  absent(cuda('float x=input[threadIdx.x]; float y=input[threadIdx.x+1];'), 'cuda.redundant_load', 'cpp');
});
test('CUDA output updates require repeated RMWs to an identified global output', () => {
  finding(cuda('for(int i=0;i<32;i++) out[threadIdx.x] += input[i];'), 'cuda.global_accumulation', 'possible_issue', 'high', 'cpp');
  absent(cuda('for(int i=0;i<32;i++) s[threadIdx.x] += input[i];'), 'cuda.global_accumulation', 'cpp');
  absent(cuda('for(int i=0;i<32;i++) out[i] += input[i];'), 'cuda.global_accumulation', 'cpp');
});
test('CUDA unguarded access and power-of-two conditions are execution assumptions', () => {
  const f = finding(cuda('if(threadIdx.x<32) s[threadIdx.x]=input[threadIdx.x];\n out[threadIdx.x]=s[threadIdx.x];'), 'cuda.guard_contract', 'unknown', 'medium', 'cpp');
  assert.equal(f.section, 'assumptions'); assert.ok(f.requires.includes('launch_config'));
  absent(cuda('if(threadIdx.x<32) { s[threadIdx.x]=input[threadIdx.x]; out[threadIdx.x]=s[threadIdx.x]; }'), 'cuda.guard_contract', 'cpp');
  const p = finding(cuda('for(int stride=blockDim.x/2;stride>0;stride/=2) { if(threadIdx.x<stride) s[threadIdx.x]+=s[threadIdx.x+stride]; }'), 'cuda.power_of_two', 'unknown', 'high', 'cpp');
  assert.equal(p.section, 'assumptions');
  absent(cuda('for(int stride=16;stride>0;stride/=2) { if(threadIdx.x<stride) s[threadIdx.x]+=s[threadIdx.x+stride]; }'), 'cuda.power_of_two', 'cpp');
});
test('PyTorch host read in a loop requires actual device provenance', () => {
  finding('import torch\nx=torch.ones((1,),device="cuda")\nfor i in range(10):\n    y=x.item()', 'graph.host_sync', 'possible_issue', 'high');
  absent('import torch\nx=torch.ones((1,),device="cpu")\nfor i in range(10):\n    y=x.item()', 'graph.host_sync');
  absent('class Store:\n    def item(self): return 1\nx=Store()\nfor i in range(10):\n    y=x.item()', 'graph.host_sync');
});
test('JAX device_get is mapped by imported identity and retains device uncertainty', () => {
  const f = finding('import jax\nimport jax.numpy as jnp\nx=jnp.ones((8,))\nfor i in range(10):\n    y=jax.device_get(x)', 'graph.host_sync', 'possible_issue', 'medium');
  assert.ok(f.requires.includes('hardware'));
  absent('class Local:\n    def device_get(self,x): return x\njax=Local()\nfor i in range(10):\n    y=jax.device_get(i)', 'graph.host_sync');
});
test('positive compiled region is scoped to recognized operations, not arbitrary callees', () => {
  finding('import jax\nimport jax.numpy as jnp\n@jax.jit\ndef step(x):\n    return jnp.sin(x)+jnp.cos(x)', 'graph.no_host_reads', 'good', 'high');
  absent('import jax\n@jax.jit\ndef step(x):\n    return unknown_helper(x)', 'graph.no_host_reads');
});
test('JIT creation inside a loop and explicit dynamic-shape compiled control are bounded risks', () => {
  finding('import jax\nfor i in range(10):\n    fn=jax.jit(lambda x: x+1)', 'graph.jit_lifetime', 'possible_issue', 'high');
  absent('import jax\nfn=jax.jit(lambda x: x+1)\nfor i in range(10):\n    y=fn(i)', 'graph.jit_lifetime');
  finding('import torch\n@torch.compile\ndef f(x: torch.Tensor):\n    if x.shape[0] > 32:\n        return x+1\n    return x', 'graph.shape_specialization', 'possible_issue', 'medium');
  absent('import torch\ndef f(x: torch.Tensor):\n    if x.shape[0] > 32:\n        return x+1\n    return x', 'graph.shape_specialization');
});
for (const [api, op] of [['all_reduce', 'all_reduce'], ['all_gather_into_tensor', 'all_gather'], ['reduce_scatter_tensor', 'reduce_scatter'], ['all_to_all_single', 'all_to_all']] as const) {
  test(`distributed ${op} has concrete API evidence and a shape requirement`, () => {
    const source = `import torch.distributed as dist\nfor i in range(10):\n    dist.${api}(${api === 'all_reduce' ? 'x' : 'output, x'})`;
    const f = finding(source, `distributed.${op}`, 'possible_issue', 'high');
    assert.ok(f.requires.includes('shapes')); assert.equal(f.scope, 'distributed');
    absent(`def ${api}(x): return x\nfor i in range(10):\n    ${api}(x)`, `distributed.${op}`);
  });
}
test('all-gather materialization uses actual producer-consumer linkage', () => {
  finding('import jax\nimport jax.numpy as jnp\ny=jax.lax.all_gather(x,"tp",tiled=True)\nz=jnp.array(y,copy=True)', 'distributed.gather_materialization', 'possible_issue', 'high');
  absent('import jax\nimport jax.numpy as jnp\ny=jax.lax.all_gather(x,"tp",tiled=True)\nz=jnp.array(other,copy=True)', 'distributed.gather_materialization');
});
test('round-trip reshard identifies both transitions, while one transition remains quiet', () => {
  finding('import jax\nfrom jax.sharding import PartitionSpec as P\na=jax.lax.with_sharding_constraint(x,P("tp"))\nb=jax.lax.with_sharding_constraint(a,P())\nc=jax.lax.with_sharding_constraint(b,P("tp"))', 'distributed.reshard_roundtrip', 'possible_issue', 'medium');
  absent('import jax\nfrom jax.sharding import PartitionSpec as P\na=jax.lax.with_sharding_constraint(x,P("tp"))', 'distributed.reshard_roundtrip');
});
test('independent work after blocking collective offers an overlap investigation', () => {
  finding('import torch\nimport torch.distributed as dist\nx=torch.ones((1,),device="cuda")\ny=torch.ones((8,),device="cuda")\ndist.all_reduce(x)\nz=torch.sin(y)\nw=torch.cos(x)', 'distributed.overlap', 'possible_issue', 'medium');
  absent('import torch\nimport torch.distributed as dist\nx=torch.ones((1,),device="cuda")\ndist.all_reduce(x)\nz=torch.sin(x)', 'distributed.overlap');
});
test('serving TP warning requires both an explicit small decode limit and TP configuration', () => {
  finding('from vllm import LLM\nengine=LLM(model="x",tensor_parallel_size=8,max_num_seqs=2)', 'serving.tp_decode', 'possible_issue', 'medium');
  absent('from vllm import LLM\nengine=LLM(model="x",tensor_parallel_size=8)', 'serving.tp_decode');
  absent('class LLM: pass\nengine=LLM(tensor_parallel_size=8,max_num_seqs=2)', 'serving.tp_decode');
});
test('chunking policy is explicit, shared prefill/decode workload is conditional', () => {
  finding('from vllm import LLM\nengine=LLM(model="x",enable_chunked_prefill=True)', 'serving.chunked_prefill', 'good', 'high');
  finding('from vllm import LLM\nengine=LLM(model="x",enable_chunked_prefill=False,max_model_len=32768,max_num_seqs=8)', 'serving.prefill_decode', 'possible_issue', 'medium');
  absent('from vllm import LLM\nengine=LLM(model="x")', 'serving.prefill_decode');
});
test('SGLang explicit CUDA graph policy produces a qualified finding', () => {
  finding('from sglang.srt.server_args import ServerArgs\nargs=ServerArgs(model_path="x",disable_cuda_graph=True)', 'serving.cuda_graphs', 'possible_issue', 'high');
  finding('from vllm import LLM\nengine=LLM(model="x",enforce_eager=False)', 'serving.cuda_graphs', 'good', 'high');
});
test('KV byte estimate requires every factor, and never invents available capacity', () => {
  const s = 'from vllm import LLM\nengine=LLM(model="x",max_model_len=4096,max_num_seqs=2,dtype="float16",kv_cache_dtype="auto",hf_overrides={"num_hidden_layers":32,"num_key_value_heads":8,"head_dim":128})';
  const f = finding(s, 'serving.kv_capacity', 'unknown', 'high');
  assert.equal(f.quantities?.[0]?.value, 2*32*8*128*2*4096*2);
  absent(s.replace('"num_key_value_heads":8,', ''), 'serving.kv_capacity');
  absent('from vllm import LLM\nnum_hidden_layers=32\nnum_key_value_heads=8\nhead_dim=128\nkv_cache_dtype_bytes=2\nengine=LLM(model="x",max_model_len=4096,max_num_seqs=2)', 'serving.kv_capacity');
});
test('unsupported, quoted, commented and shadowed evidence is suppressed', () => {
  assert.equal(run('# torch.cuda.synchronize()\ntext="jax.device_get(x)"\nvalue=1').findings.length, 0);
  absent('import torch.distributed as dist\ndef f(dist):\n    for i in range(8):\n        dist.all_reduce(x)', 'distributed.all_reduce');
  absent('import jax\njax=SomethingElse()\nfor i in range(10):\n    y=jax.device_get(x)', 'graph.host_sync');
});
test('the gate rejects missing evidence, invented locations, and fabricated measured impact', () => {
  const r=run(cuda('if(threadIdx.x==0) {float sum=0; for(int i=0;i<32;i++) sum+=s[i]; out[0]=sum;}'), 'cpp');
  const f=r.findings[0]!; assert.ok(f);
  assert.equal(gateFinding({...f,evidence_ids:['invented']},r.ir), undefined);
  assert.equal(gateFinding({...f,evidence:{...f.evidence,source:'fake()'}},r.ir), undefined);
  assert.equal(gateFinding({...f,runtime_impact_measured:true},r.ir), undefined);
  assert.equal(gateFinding({...f,confidence:'low'},r.ir), undefined);
});
