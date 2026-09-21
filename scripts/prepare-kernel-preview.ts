import {readFileSync,writeFileSync,existsSync} from 'node:fs';
import {BlockReader} from '../src/core/blocks';
import {analyzeSystems} from '../src/core/systems/analyze';
import {systemsReport} from '../src/core/systems/evaluate';
import {insightPresentation} from '../src/core/presentation';
import type {Choice} from '../src/live-types';
import manifest from '../package.json';
const source=readFileSync('test/fixtures/phase1_cpasync_2stage.cu','utf8');
const c=new BlockReader().read({source,file:'phase1_cpasync_2stage.cu',language:'cpp',cursorLine:101,assessmentScope:'function'});
const selected:Record<string,string>={'tensor-cores':'tcgen05',overlap:'overlapped','mma-cadence':'tile_wait','pipeline-depth':'double',epilogue:'tmem_wait',resources:'multibuffer',reuse:'shared_tiles'};
const answers:Record<string,Choice>=Object.fromEntries(Object.entries(selected).map(([k,v])=>['review.cuda-'+k,{type:'choice',choice:v,probabilities:{[v]:1}}]));
const path=`artifacts/kernel-live-${manifest.version}.json`,recorded=existsSync(path)?JSON.parse(readFileSync(path,'utf8')).results.find((r:any)=>r.name==='blackwell-double-buffer'):undefined;
const report=recorded?.report??systemsReport(c,analyzeSystems(c),answers);
const state={version:manifest.version,enabled:true,configured:true,consented:true,phase:'ready',message:'Insights updated. Keep writing.',domain:'auto',intent:'',file:c.file,unit:c.unit,stale:false,hardwareProfile:'b200',
  context:{characters:c.characters,truncated:false,references:[]},report,...insightPresentation(report),references:[],requests:1,advisorRequests:0,totalTokens:report.inputTokens??0,totalCost:0,
  advisor:{mode:'onDemand',model:'openai/gpt-6-astra',phase:'idle',message:'Review a specific finding when needed.'}};
writeFileSync('artifacts/systems-preview.json',JSON.stringify(state));
console.log(`Kernel preview uses ${recorded?'recorded live Jev':'controlled fixture'} classifications.`);
