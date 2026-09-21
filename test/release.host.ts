import * as vscode from 'vscode';
import assert from 'node:assert/strict';
import path from 'node:path';
import fs from 'node:fs';
import type { KernelLensApi } from '../src/extension';
import manifest from '../package.json';
// Read fixtures relative to the source checkout, including when testing a VSIX.
const pause=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));
export async function run() {
  const root=process.env.KERNEL_LENS_TEST_ROOT!;
  process.chdir(root);
  const {releaseCases}=await import('./release-cases.js');
  const live=process.env.KERNEL_LENS_LIVE==='1';
  const extension=vscode.extensions.getExtension<KernelLensApi>(`${manifest.publisher}.${manifest.name}`)!;
  const api=await extension.activate();
  const config=vscode.workspace.getConfiguration('kernelLens');
  await config.update('debounceMs',300,vscode.ConfigurationTarget.Workspace);
  await config.update('maxRequestsPerMinute',120,vscode.ConfigurationTarget.Workspace);
  let calls=0;
  const recorded=JSON.parse(fs.readFileSync(path.join(root,'artifacts/release-live-0.6.1.json'),'utf8'));
  let scenario='';
  const fetcher:typeof fetch=async(_url,options)=>{
    calls++;
    const body=JSON.parse(String(options?.body));
    const findings=recorded.results.find((r:any)=>r.name===scenario).report.findings;
    return Response.json({answers:Object.fromEntries(Object.entries(body.questions).map(([id,q]:[string,any])=>{
      const f=findings.find((f:any)=>f.id===id&&f.model_choice);
      if(f) {assert.ok(q.criteria[f.model_choice]);return [id,{type:'choice',choice:f.model_choice,probabilities:f.model_probabilities}];}
      return [id,{type:'choice',choice:'unknown',probabilities:Object.fromEntries(Object.keys(q.criteria).map(k=>[k,k==='unknown'?1:0]))}];
    }))});
  };
  const wait=async(predicate:()=>boolean,label:string)=>{
    let manualRetry=false;
    const start=Date.now();while(!predicate()&&Date.now()-start<45000) {
      // Exercise the real Analyze recovery path once if the external provider
      // exhausted the controller's automatic retry. Never retry assertions.
      if(live&&!manualRetry&&api.getState().phase==='error'&&/HTTP 50[234]/.test(api.getState().message)) {
        manualRetry=true;console.log(`${label}: Gateway unavailable; retrying Analyze once.`);await pause(3000);
        await vscode.commands.executeCommand('kernelLens.analyze');
      }
      await pause(50);
    }
    assert.ok(predicate(),`${label}: ${JSON.stringify(api.getState())}`);
  };
  const snapshots=[];
  await api.testing!.configure(live?process.env.AI_GATEWAY_API_KEY!:'replayed-release-key',true,live?undefined:fetcher);
  for(const item of releaseCases.filter(c=>(!live||['rms-norm-release','sgemm-coalesced-release','training-risk','inference-risk'].includes(c.name))
    && (!process.env.KERNEL_LENS_WORKFLOW_FILTER||process.env.KERNEL_LENS_WORKFLOW_FILTER.split(',').includes(c.name)))) {
    scenario=item.name;
    const uri=vscode.Uri.joinPath(vscode.workspace.workspaceFolders![0]!.uri,item.name+path.extname(item.file));
    await vscode.workspace.fs.writeFile(uri,Buffer.from(item.source));
    const doc=await vscode.languages.setTextDocumentLanguage(await vscode.workspace.openTextDocument(uri),item.file.endsWith('.cu')?'cpp':'python');
    const editor=await vscode.window.showTextDocument(doc);editor.selection=new vscode.Selection(item.cursor,0,item.cursor,0);
    await wait(()=>api.getState().phase==='ready'&&!api.getState().stale&&api.getState().file===path.basename(uri.fsPath),item.name);
    const state=api.getState(),risks=state.report!.findings!.filter(f=>f.assessment.includes('issue'));
    assert.ok(item.issues.every(p=>risks.some(f=>f.id.startsWith(p))),`${item.name}: expected ${item.issues}, got ${risks.map(f=>f.id)}`);
    assert.ok(item.absent.every(p=>!risks.some(f=>f.id.startsWith(p))),item.name);
    assert.equal(state.report!.hardware?.profile?.id,'b200');
    assert.equal(state.advisorRequests,0);
    const count=state.requests,fingerprint=state.report!.fingerprint;
    for(let line=state.unit!.startLine-1;line<state.unit!.endLine;line++) {editor.selection=new vscode.Selection(line,0,line,0);await pause(15);}
    await pause(500);
    assert.equal(api.getState().requests,count,`${item.name}: navigation should not call Jev`);
    assert.equal(api.getState().report?.fingerprint,fingerprint);
    for(const f of risks)assert.ok(vscode.languages.getDiagnostics(uri).some(d=>String(d.code)===f.id),`${f.id} diagnostic`);
    snapshots.push({name:item.name,state:api.getState()});
    fs.writeFileSync(path.join(root,`artifacts/release-host-${live?'live':'replayed'}-0.6.1.json`),JSON.stringify({version:manifest.version,extensionPath:extension.extensionPath,live,snapshots},null,2));
    console.log(`${item.name}: findings, diagnostics and navigation passed.`);
    if(live)await pause(2200);
  }
  if(!live) {
    const before=calls;
    await config.update('hardwareProfile','h100',vscode.ConfigurationTarget.Workspace);
    await wait(()=>api.getState().phase==='ready'&&!api.getState().stale&&api.getState().report?.hardware?.profile?.id==='h100','hardware updates report');
    assert.equal(calls,before+1);
    snapshots.push({name:'hardware selection invalidates cache',state:api.getState()});
  }
  fs.writeFileSync(path.join(root,`artifacts/release-host-${live?'live':'replayed'}-0.6.1.json`),JSON.stringify({version:manifest.version,extensionPath:extension.extensionPath,live,snapshots},null,2));
  console.log(`Release editor checks passed: ${snapshots.length} scenarios (${live?'live':'recorded Jev responses'}).`);
}
