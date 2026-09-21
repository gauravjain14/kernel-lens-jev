import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import type { KernelLensApi } from '../src/extension';
import manifest from '../package.json';

const pause=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
export async function run() {
  const root=process.env.KERNEL_LENS_TEST_ROOT!,live=process.env.KERNEL_LENS_LIVE==='1';
  const extension=vscode.extensions.getExtension<KernelLensApi>(`${manifest.publisher}.${manifest.name}`)!;
  const api=await extension.activate(),config=vscode.workspace.getConfiguration('kernelLens');
  await config.update('debounceMs',300,vscode.ConfigurationTarget.Workspace);
  await config.update('maxRequestsPerMinute',120,vscode.ConfigurationTarget.Workspace);
  const source=fs.readFileSync(path.join(root,'test/fixtures/phase1_cpasync_2stage.cu'),'utf8');
  const uri=vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri,'phase1_cpasync_2stage.cu');
  await vscode.workspace.fs.writeFile(uri,Buffer.from(source));
  const doc=await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(uri),'cpp');
  const editor=await vscode.window.showTextDocument(doc);editor.selection=new vscode.Selection(0,0,0,0);
  let calls=0;
  const fetcher:typeof fetch=async(_url,options)=>{
    calls++;const body=JSON.parse(String(options?.body));
    assert.ok(body.state.current.code.includes('asm volatile('),'original assembly sent to Jev');
    assert.ok(Object.keys(body.questions).some(id=>id.startsWith('review.')));
    if(calls===1) {
      assert.equal(body.state.current.startLine,102);assert.equal(body.state.current.endLine,334);
      assert.ok(body.state.current.code.includes('tcgen05.ld.sync.aligned.32x32b.x8.b32'));
      assert.equal(body.state.current.truncated,false);
    }
    const buckets:Record<string,string>={'review.cuda-tensor-cores':'tcgen05','review.cuda-overlap':'overlapped','review.cuda-mma-cadence':'tile_wait','review.cuda-pipeline-depth':'double','review.cuda-epilogue':'tmem_wait','review.cuda-resources':'multibuffer'};
    return Response.json({answers:Object.fromEntries(Object.entries(body.questions).map(([id,q]:[string,any])=>{
      const selected=buckets[id]&&q.criteria[buckets[id]!]?buckets[id]!:'unknown';
      return [id,{type:'choice',choice:selected,probabilities:Object.fromEntries(Object.keys(q.criteria).map(k=>[k,k===selected?1:0]))}];
    }))});
  };
  const artifact=path.join(root,`artifacts/cuda-asm-host-${live?'live':'controlled'}-${manifest.version}.json`);
  const save=()=>fs.writeFileSync(artifact,JSON.stringify({version:manifest.version,extensionPath:extension.extensionPath,live,state:api.getState()},null,2));
  const wait=async(check:()=>boolean,label:string)=>{
    const start=Date.now();while(!check()&&Date.now()-start<45000) {
      if(api.getState().phase==='error') {save();throw Error(`${label}: ${api.getState().message}`);}
      await pause(40);
    }
    if(!check())save();assert.ok(check(),`${label}: ${api.getState().message}`);
  };
  await api.testing!.configure(live?process.env.AI_GATEWAY_API_KEY!:'inline-asm-test-key',true,live?undefined:fetcher);
  await wait(()=>api.getState().phase==='ready'&&!api.getState().stale,'downloaded file assessed automatically');
  assert.equal(api.getState().unit?.name,'gemm_tcgen5_v0');assert.equal(api.getState().unit?.ready,true);
  assert.equal(api.getState().unit?.syntax.length,0);assert.ok(api.getState().requests>0);
  if(!live) {
    const report=api.getState().report!;
    for(const id of ['tensor-cores','overlap','mma-cadence','pipeline-depth','epilogue','resources'])assert.ok(report.findings?.some(f=>f.id==='review.cuda-'+id),id);
    assert.equal(report.dimensions?.length,24);
    const lenses=await vscode.commands.executeCommand<vscode.CodeLens[]>('vscode.executeCodeLensProvider',uri);
    assert.ok(lenses?.some(l=>l.command?.title.includes('Blackwell asynchronous MMA')),'function-level classification visible in the editor');
  }
  const fingerprint=api.getState().report!.fingerprint,count=api.getState().requests;
  for(let line=101;line<334;line++) {editor.selection=new vscode.Selection(line,0,line,0);await pause(5);}
  await pause(400);assert.equal(api.getState().requests,count);assert.equal(api.getState().report?.fingerprint,fingerprint);
  assert.equal(vscode.languages.getDiagnostics(uri).filter(d=>d.code==='syntax').length,0);
  save();
  if(!live) {
    editor.selection=new vscode.Selection(47,0,47,0);
    await wait(()=>api.getState().phase==='ready'&&api.getState().unit?.name==='mbar_wait','inline-PTX helper assessed');
    assert.equal(api.getState().unit?.syntax.length,0);
    const before=calls;
    const unfinished=vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri,'unfinished-asm.cu');
    await vscode.workspace.fs.writeFile(unfinished,Buffer.from('__global__ void f(){ asm volatile("nop;"'));
    const unfinishedDoc=await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(unfinished),'cpp');
    const unfinishedEditor=await vscode.window.showTextDocument(unfinishedDoc);unfinishedEditor.selection=new vscode.Selection(0,0,0,0);
    await wait(()=>api.getState().unit?.ready===false,'unfinished assembly waits');await pause(700);assert.equal(calls,before);
  }
  console.log(`Inline PTX editor check passed (${live?'live Jev':'controlled Jev'}): entire downloaded kernel, all 233 body lines, original assembly retained.`);
}
