import {readFileSync,writeFileSync,existsSync} from 'node:fs';
import {BlockReader} from '../src/core/blocks';
import {analyzeSystems} from '../src/core/systems/analyze';
import {systemsPayload,systemsReport} from '../src/core/systems/evaluate';
import {requestEvaluation,parseAnswers} from '../src/core/gateway';
import {extractUsage} from '../src/core/assessment';
import {releaseCases} from '../test/release-cases';
import manifest from '../package.json';

async function main(){
  const arg=process.argv.indexOf('--env-file');if(arg>=0)process.loadEnvFile(process.argv[arg+1]);
  const key=process.env.AI_GATEWAY_API_KEY;if(!key)throw Error('AI_GATEWAY_API_KEY is required');
  const cases=[{name:'blackwell-double-buffer',file:'phase1_cpasync_2stage.cu',source:readFileSync('test/fixtures/phase1_cpasync_2stage.cu','utf8'),cursor:101},...releaseCases.slice(0,3)];
  const output=`artifacts/kernel-live-${manifest.version}.json`;
  const results:any[]=process.argv.includes('--case')&&existsSync(output)?JSON.parse(readFileSync(output,'utf8')).results:[];
  for(const item of cases){
    const filter=process.argv.indexOf('--case');if(filter>=0&&!item.name.includes(process.argv[filter+1]!))continue;
    const context=new BlockReader().read({source:item.source,file:item.file,language:'cpp',cursorLine:item.cursor,assessmentScope:'function'});
    const analysis=analyzeSystems(context),payload=systemsPayload(context,analysis),started=performance.now();
    const raw=await requestEvaluation(payload,key,new AbortController().signal);
    const answers=parseAnswers(raw,payload),report=systemsReport(context,analysis,answers,extractUsage(raw,started));
    const classified=report.dimensions!.filter(d=>d.findingIds.length),risks=report.findings!.filter(f=>f.assessment.includes('issue'));
    const checks=item.name==='blackwell-double-buffer'?[classified.some(d=>d.id==='review.cuda-tensor-cores'&&d.bucket==='Blackwell asynchronous MMA'),classified.some(d=>d.id==='review.cuda-overlap'),classified.some(d=>d.id==='review.cuda-mma-cadence'),classified.some(d=>d.id==='review.cuda-epilogue'),!risks.some(f=>/Serial reduction|narrow branch/.test(f.title))]
      :item.name.startsWith('rms')?[risks.some(f=>f.id.startsWith('cuda.serial_reduction'))]
      :item.name.includes('coalesc')||item.name.includes('sgemm')?[risks.some(f=>f.id==='review.cuda-reuse')]
      :[!risks.some(f=>/serial_reduction|cuda-thread-work|cuda-accumulator/.test(f.id))];
    const ok=checks.every(Boolean)&&report.findings!.every(f=>f.evidence.source&&f.evidence.location&&f.next_check&&f.requires.length&&!f.runtime_impact_measured);
    const previous=results.findIndex(r=>r.name===item.name);if(previous>=0)results.splice(previous,1);
    results.push({name:item.name,ok,answers,report});
    writeFileSync(output,JSON.stringify({version:manifest.version,results},null,2)+'\n');
    console.log(JSON.stringify({name:item.name,ok,requestBytes:JSON.stringify(payload).length,latencyMs:report.latencyMs,inputTokens:report.inputTokens,outputTokens:report.outputTokens,classes:classified.map(d=>({dimension:d.label,bucket:d.bucket,probability:d.probability,assessment:d.assessment})),unknown:report.dimensions!.filter(d=>d.applicable&&d.assessment==='unknown').map(d=>d.label)}));
  }
  if(results.some(r=>!r.ok))process.exitCode=1;
}
void main().catch(error=>{console.error(error instanceof Error?error.message:'Kernel evaluation failed');process.exitCode=1;});
