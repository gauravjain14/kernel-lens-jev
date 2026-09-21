import { parseArgs } from 'node:util';
import { writeFileSync } from 'node:fs';
import { BlockReader } from '../src/core/blocks';
import { assessSystems } from '../src/core/systems/evaluate';
import { GatewayError } from '../src/core/gateway';

const cases = [
  { name:'serial reduction', file:'reduce.cu', source:'__global__ void reduce(float* out,const float* input) {\n__shared__ float s[32];\ns[threadIdx.x]=input[threadIdx.x]; __syncthreads();\nif(threadIdx.x==0){float sum=0;for(int i=0;i<32;i++)sum+=s[i];out[0]=sum;}\n__syncthreads();\n}', rule:'cuda.serial_reduction', expected:['possible_issue','likely_issue'] },
  { name:'warp reduction', file:'reduce.cu', source:'__global__ void reduce(float* out,const float* input) {\nfloat sum=input[threadIdx.x];\nfor(int offset=16;offset>0;offset/=2)sum+=__shfl_down_sync(0xffffffff,sum,offset);\nif(threadIdx.x==0)out[0]=sum;\n}', rule:'cuda.warp_reduction', expected:['good'] },
  { name:'torch per-step host read', file:'step.py', source:'import torch\nx=torch.ones((1,),device="cuda")\nfor step in range(100):\n    x=torch.sin(x)\n    value=x.item()\n', rule:'graph.host_sync', expected:['possible_issue','likely_issue'] },
  { name:'CPU item control', file:'step.py', source:'import torch\nx=torch.ones((1,),device="cpu")\nfor step in range(100):\n    value=x.item()\n', rule:'graph.host_sync', expected:['omitted'] },
  { name:'JAX host read without hardware', file:'step.py', source:'import jax\nimport jax.numpy as jnp\nx=jnp.ones((8,))\nfor step in range(100):\n    x=jnp.sin(x)\n    value=jax.device_get(x)\n', rule:'graph.host_sync', expected:['possible_issue','unknown','omitted'] },
  { name:'loop collective', file:'distributed.py', source:'import torch\nimport torch.distributed as dist\nx=torch.ones((128,),device="cuda",dtype=torch.float32)\nfor step in range(100):\n    dist.all_reduce(x)\n    y=torch.sin(x)\n', rule:'distributed.all_reduce', expected:['possible_issue','likely_issue'] },
  { name:'TP and small request limit', file:'serve.py', source:'from vllm import LLM\nengine=LLM(model="example-model",tensor_parallel_size=8,max_num_seqs=2)\n', rule:'serving.tp_decode', expected:['possible_issue','unknown','omitted'] },
  { name:'SGLang eager policy', file:'serve.py', source:'from sglang.srt.server_args import ServerArgs\nargs=ServerArgs(model_path="example-model",disable_cuda_graph=True)\n', rule:'serving.cuda_graphs', expected:['possible_issue','likely_issue','unknown','omitted'] },
];
async function main() {
  const {values}=parseArgs({options:{'env-file':{type:'string'},output:{type:'string',default:'artifacts/systems-live-0.6.json'},filter:{type:'string'}}});
  if(values['env-file'])process.loadEnvFile(values['env-file']);
  const key=process.env.AI_GATEWAY_API_KEY;if(!key)throw new Error('Set AI_GATEWAY_API_KEY or pass --env-file.');
  const results:unknown[]=[];let failures=0;
  const save=()=>writeFileSync(values.output!,JSON.stringify({date:new Date().toISOString(),note:'Synthetic contract and smoke checks, not a domain accuracy estimate. Ambiguous policy cases allow uncertainty or suppression. Every result and failure is retained.',failures,results},null,2));
  for(const item of cases.filter(c=>!values.filter||c.name.includes(values.filter))) {
    const context=new BlockReader().read({file:item.file,language:item.file.endsWith('.cu')?'cpp':'python',source:item.source,cursorLine:0});
    let report;
    try {
      for(let attempt=0;;attempt++) {
        try{report=await assessSystems(context,key,new AbortController().signal);break;}
        catch(error){if(!(error instanceof GatewayError)||attempt>=1||!(error.status===429||error.status>=500))throw error;await new Promise(r=>setTimeout(r,Math.min(error.status===429?error.retryAfterMs:1500,60000)));}
      }
      const finding=report.findings!.find(f=>f.id.startsWith(item.rule)),outcome=finding?.assessment??'omitted',match=item.expected.includes(outcome);
      if(!match)failures++;
      results.push({name:item.name,expected:item.expected,outcome,match,report});save();
      console.log(JSON.stringify({name:item.name,outcome,match,latencyMs:report.latencyMs,inputTokens:report.inputTokens}));
    }catch(error){failures++;results.push({name:item.name,error:error instanceof Error?error.message:'Request failed'});save();console.error(`Failed ${item.name}: ${error instanceof Error?error.message:'Request failed'}`);}
    await new Promise(r=>setTimeout(r,2200));
  }
  if(failures)process.exitCode=1;
}
void main().catch(error=>{console.error(error instanceof Error?error.message:'Systems evaluation failed');process.exitCode=1;});
