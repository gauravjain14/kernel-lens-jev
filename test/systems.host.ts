import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import type { KernelLensApi } from '../src/extension';
import manifest from '../package.json';

const serial='__global__ void reduce(float* out) {\n    __shared__ float s[32];\n    if(threadIdx.x==0) {\n        float sum=0;\n        for(int i=0;i<32;i++) sum+=s[i];\n        out[0]=sum;\n    }\n    __syncthreads();\n}\n';
const shuffle='__global__ void reduce(float* out) {\n    __shared__ float s[32];\n    float sum=s[threadIdx.x];\n    for(int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);\n    if(threadIdx.x==0)out[0]=sum;\n}\n';
const modelRisk=serial.replace('    __syncthreads();','    for(int j=0;j<32;j++) out[threadIdx.x]+=j;\n    __syncthreads();');
const pause=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
export async function run() {
  const extension=vscode.extensions.getExtension<KernelLensApi>(`${manifest.publisher}.${manifest.name}`)!;
  const api=await extension.activate(); assert.ok(api.testing);
  const root=process.env.KERNEL_LENS_TEST_ROOT ?? extension.extensionPath, live=process.env.KERNEL_LENS_LIVE==='1';
  const config=vscode.workspace.getConfiguration('kernelLens');
  await config.update('debounceMs',300,vscode.ConfigurationTarget.Workspace);
  await config.update('maxRequestsPerMinute',120,vscode.ConfigurationTarget.Workspace);
  await config.update('advisor.mode','onDemand',vscode.ConfigurationTarget.Workspace);
  let calls=0, held:(()=>void)|undefined, delayNext=false, fail=0, failCount=0;
  const fetcher:typeof fetch=async(url,options)=>{
    const body=JSON.parse(String(options?.body));
    if(String(url).endsWith('/chat/completions')) {
      const state=JSON.parse(body.messages.at(-1).content), f=state.findings.find((f:any)=>f.assessment.includes('issue'));
      const line=state.context.current.code.split('\n').find((s:string)=>s.startsWith(`L${f.evidence.location.startLine}: `));
      return Response.json({choices:[{finish_reason:'stop',message:{content:JSON.stringify({summary:'Inspect the serial reduction tail.',items:[{metricId:f.id,verdict:'conditional',title:f.title,explanation:f.evidence.explanation,action:f.next_check,line:f.evidence.location.startLine,evidence:line.replace(/^L\d+: /,'')}]})}}]});
    }
    calls++; assert.ok(body.state.systems); assert.ok(!body.questions.technology,'no redundant routing request');
    assert.ok(Object.keys(body.questions).every(id=>!id.startsWith('anchor_')),'evidence is local');
    if(delayNext) {delayNext=false;await new Promise<void>(resolve=>{held=resolve;});}
    if(fail) {const status=fail;if(--failCount<=0)fail=0;return Response.json({error:{type:'test_failure'}},{status,headers:{'retry-after':'0'}});}
    return Response.json({answers:Object.fromEntries(Object.entries(body.questions).map(([id,q]:[string,any])=>[id,{type:'choice',choice:id.startsWith('review.')?'unknown':'retain',probabilities:Object.fromEntries(Object.keys(q.criteria).map(k=>[k,k===(id.startsWith('review.')?'unknown':'retain')?1:0]))}])),usage:{inputTokens:500}});
  };
  const wait=async(predicate:()=>boolean,label:string,timeout=live?45000:15000)=>{
    const start=Date.now();while(!predicate()&&Date.now()-start<timeout) await pause(40);
    assert.ok(predicate(),`${label}: ${JSON.stringify(api.getState())}`);
  };
  const snapshots:any[]=[];
  const save=async(label:string)=>{snapshots.push({label,state:api.getState()});await fs.writeFile(path.join(root,`artifacts/systems-host-${live?'live':'mocked'}-${manifest.version}.json`),JSON.stringify({version:manifest.version,extensionPath:extension.extensionPath,live,snapshots},null,2));};
  const uri=vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri,'reduction.cu');
  await vscode.workspace.fs.writeFile(uri,Buffer.from(serial));
  const doc=await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(uri),'cpp');
  let editor=await vscode.window.showTextDocument(doc);editor.selection=new vscode.Selection(0,0,0,0);
  await wait(()=>api.getState().phase==='disabled','requires configured key and consent');
  if(!live) {
    await api.testing!.configure('synthetic-systems-key',false,fetcher); await pause(700);assert.equal(calls,0);
  }
  await api.testing!.configure(live?process.env.AI_GATEWAY_API_KEY!:'synthetic-systems-key',true,live?undefined:fetcher);
  const has=(prefix:string,assessment?:string)=>api.getState().report?.findings?.some(f=>f.id.startsWith(prefix)&&(!assessment||f.assessment===assessment))===true;
  await wait(()=>api.getState().phase==='ready'&&has('cuda.serial_reduction'),'serial tail automatically visible');
  assert.ok(api.getState().insights?.some(f=>f.id.startsWith('cuda.serial_reduction')));
  assert.equal(api.getState().advisorRequests,0);assert.ok(api.getState().report?.findings?.every(f=>f.evidence.source&&f.next_check&&f.requires.length));
  assert.ok(vscode.languages.getDiagnostics(uri).some(d=>String(d.code).startsWith('cuda.serial_reduction')));
  const lenses=await vscode.commands.executeCommand<vscode.CodeLens[]>('vscode.executeCodeLensProvider',uri);
  assert.ok(lenses?.some(l=>l.command?.title.includes('Serial reduction')));
  const hovers=await vscode.commands.executeCommand<vscode.Hover[]>('vscode.executeHoverProvider',uri,new vscode.Position(2,5));
  const hoverText=hovers?.flatMap(h=>h.contents.map(c=>typeof c==='string'?c:c.value)).join('\n') ?? '';
  assert.ok(hoverText.includes('Runtime impact is unmeasured'),hoverText);
  await save('cuda serial source evidence, diagnostic, CodeLens and hover');
  await vscode.commands.executeCommand('kernelLens.open');await pause(400);
  if(process.env.KERNEL_LENS_CAPTURE==='1') execFileSync('python',['-c','from PIL import ImageGrab; import sys; ImageGrab.grab().save(sys.argv[1])',path.join(root,'artifacts/systems-native-0.6.png')]);
  if(!live) {
    const count=calls;await vscode.commands.executeCommand('kernelLens.analyze');await wait(()=>api.getState().phase==='ready'&&!api.getState().stale,'cache reused');assert.equal(calls,count);
    await vscode.commands.executeCommand('kernelLens.review');assert.equal(api.getState().advisor.phase,'ready');
    await vscode.commands.executeCommand('kernelLens.copyHandoff');assert.match(await vscode.env.clipboard.readText(),/evidence_ids/);assert.equal(doc.getText(),serial);
  }
  const replace=async(source:string)=>{
    editor=await vscode.window.showTextDocument(doc);editor.selection=new vscode.Selection(0,0,0,0);
    await editor.edit(edit=>edit.replace(new vscode.Range(new vscode.Position(0,0),doc.lineAt(doc.lineCount-1).range.end),source));
    editor.selection=new vscode.Selection(0,0,0,0);
  };
  await replace(shuffle);await wait(()=>!api.getState().stale&&has('cuda.warp_reduction','good'),'shuffle positive updates automatically');
  assert.ok(!api.getState().insights?.some(f=>f.id.startsWith('cuda.serial_reduction')));assert.ok(api.getState().improvements?.length);await save('cuda corrected to warp reduction');
  if(!live) {
    delayNext=true;await replace(modelRisk.replace('float sum=0;', 'float sum=0; // delayed'));await wait(()=>!!held,'pending Jev request');
    await replace(shuffle.replace('float sum=s[threadIdx.x];','float sum=s[threadIdx.x]; // latest'));held!();held=undefined;
    await wait(()=>api.getState().phase==='ready'&&!api.getState().stale&&has('cuda.warp_reduction','good'),'stale Jev response rejected');
    assert.ok(!has('cuda.serial_reduction'));
    fail=401;await replace(modelRisk.replace('float sum=0;', 'float sum=0; // denied'));await wait(()=>api.getState().phase==='error','auth error visible');
    await vscode.commands.executeCommand('kernelLens.analyze');await wait(()=>api.getState().phase==='ready','explicit retry recovers');
    fail=429;await replace(modelRisk.replace('float sum=0;', 'float sum=0; // rate limited'));await wait(()=>api.getState().phase==='limited','provider limit visible');await wait(()=>api.getState().phase==='ready','rate limit resumes');
    fail=529;failCount=2;await replace(modelRisk.replace('float sum=0;', 'float sum=0; // provider overloaded'));
    await wait(()=>api.getState().phase==='error'&&has('cuda.serial_reduction'),'direct source observations survive provider overload');
    assert.match(api.getState().message,/source observations only/);assert.equal(api.getState().improvements?.length,0);
    assert.ok(!has('cuda.global_accumulation'),'unassessed model hypothesis stays hidden');
    assert.ok(vscode.languages.getDiagnostics(uri).some(d=>String(d.code).startsWith('cuda.serial_reduction')));
    await vscode.commands.executeCommand('kernelLens.analyze');await wait(()=>api.getState().phase==='ready'&&has('cuda.global_accumulation'),'failed prediction was not cached; Analyze recovers');
    await replace('__global__ void f(){');await wait(()=>api.getState().unit?.ready===false,'unfinished block waits');const count=calls;await pause(650);assert.equal(calls,count);
    await save('cache, cancellation, auth, rate recovery and syntax readiness');
  }
  const examples=[
    ['torch.py','import torch\nx=torch.ones((1,),device="cuda")\nfor i in range(10):\n    result=x.item()\n','graph.host_sync'],
    ['jax.py','import jax\nimport jax.numpy as jnp\nx=jnp.ones((8,))\nfor i in range(10):\n    result=jax.device_get(x)\n','graph.host_sync'],
    ['distributed.py','import torch.distributed as dist\nfor i in range(8):\n    dist.all_reduce(x)\n','distributed.all_reduce'],
    ['serve.py','from vllm import LLM\nengine=LLM(model="x",tensor_parallel_size=8,max_num_seqs=2,enable_chunked_prefill=True)\n','serving.tp_decode'],
    ['sglang.py','from sglang.srt.server_args import ServerArgs\nargs=ServerArgs(model_path="x",disable_cuda_graph=True)\n','serving.cuda_graphs'],
  ];
  for(const [file,source,prefix] of examples) {
    if(live) await pause(2200);
    const fileUri=vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri,file!);
    await vscode.workspace.fs.writeFile(fileUri,Buffer.from(source!));
    const python=await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(fileUri),'python');
    const ed=await vscode.window.showTextDocument(python);ed.selection=new vscode.Selection(0,0,0,0);
    await wait(()=>api.getState().phase==='ready'&&!api.getState().stale&&api.getState().file===file&&has(prefix!),`${file} automatic evidence-backed report`);
    await save(file!);
  }
  await vscode.commands.executeCommand('kernelLens.pause');assert.equal(api.getState().phase,'paused');assert.equal(api.getState().report,undefined);
  assert.equal(vscode.languages.getDiagnostics().filter(([,ds])=>ds.some(d=>d.source?.startsWith('Kernel Lens'))).length,0);
  await save('pause clears annotations');
  console.log(`Systems extension host passed (${live?'live Jev':'controlled Jev'}): ${snapshots.length} snapshots, package ${manifest.version}.`);
}
