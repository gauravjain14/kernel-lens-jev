import {readFileSync,writeFileSync} from 'node:fs';
import {BlockReader} from '../src/core/blocks';
import {analyzeSystems} from '../src/core/systems/analyze';
import {assessSystems,systemsPayload} from '../src/core/systems/evaluate';
import {releaseCases} from '../test/release-cases';
import {GatewayError} from '../src/core/gateway';
async function main() {
const arg=process.argv.indexOf('--env-file');
if(arg>=0)process.loadEnvFile(process.argv[arg+1]);
if(!process.env.AI_GATEWAY_API_KEY)throw Error('AI_GATEWAY_API_KEY is required');
const reader=new BlockReader();
const results: any[]=process.argv.includes('--case')?JSON.parse(readFileSync('artifacts/release-live-0.6.1.json','utf8')).results:[];
for(const item of releaseCases) {
  const {name,source,file}=item;
  const filter=process.argv.indexOf('--case');if(filter>=0&&!name.includes(process.argv[filter+1]!))continue;
  const c=reader.read({file,source,language:file.endsWith('.cu')?'cpp':'python',cursorLine:item.cursor,assessmentScope:'function'});
  const a=analyzeSystems(c),request=systemsPayload(c,a);
  let report;
  for(let attempt=0;attempt<3;attempt++) {
    try { report=await assessSystems(c,process.env.AI_GATEWAY_API_KEY,new AbortController().signal);break; }
    catch(error) {if(!(error instanceof GatewayError)||![429,502,503,504].includes(error.status)||attempt===2)throw error;await new Promise(resolve=>setTimeout(resolve,Math.max(2200,error.retryAfterMs)));}
  }
  if(!report)throw Error('No report');
  const risks=report.findings!.filter(f=>f.assessment.includes('issue'));
  const ok=item.issues.every(prefix=>risks.some(f=>f.id.startsWith(prefix)))&&item.absent.every(prefix=>!risks.some(f=>f.id.startsWith(prefix)));
  if(!report.findings!.every(f=>f.evidence.source&&f.evidence.location&&f.next_check&&f.requires.length&&!f.runtime_impact_measured))throw Error('Invalid evidence contract');
  const old=results.findIndex(r=>r.name===name);if(old>=0)results.splice(old,1);
  results.push({name,ok,scope:c.unit,questions:Object.keys(request.questions),report});
  writeFileSync('artifacts/release-live-0.6.1.json',JSON.stringify({results},null,2)+'\n');
  console.log(JSON.stringify({name,ok,questions:Object.keys(request.questions).length,latencyMs:report.latencyMs,inputTokens:report.inputTokens,findings:report.findings!.map(f=>({id:f.id,title:f.title,assessment:f.assessment,line:f.evidence.location!.startLine,probability:f.model_probability,confidence:f.confidence}))}));
  await new Promise(resolve=>setTimeout(resolve,2200));
}
if(results.some(r=>!r.ok))process.exitCode=1;

}
void main().catch(error=>{console.error(error);process.exitCode=1;});
