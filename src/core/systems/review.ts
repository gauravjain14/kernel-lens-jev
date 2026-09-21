import type { Choice, Metric, Pack } from '../../live-types';
import { metrics } from '../rubrics';
import { guidance } from '../guidance';
import { gateFinding } from './gate';
import type { Analysis } from './evaluate';
import type { Evidence, Impact, JevFinding, Requirement, Scope, SystemsOp } from './types';
import { isKernelReview, kernelFindings, kernelQuestions } from './kernel-review';

type Dimension = { id: string; label: string; question: string; issue: string; good: string; issueTitle: string; goodTitle: string;
  category: string; scope: Scope; impact: Impact[]; requires: Requirement[]; next: string; select: (op: SystemsOp) => boolean };
const mapping: Record<string, [string, Requirement[]]> = {
  'cuda-cooperative-loads':['shared_memory',['launch_config']], 'cuda-coalescing':['global_memory',['launch_config']],
  'cuda-accumulator':['global_memory',['compiler_output']], 'cuda-thread-work':['reduction_collectives',['none']],
  'cuda-reuse':['data_reuse',['compiler_output']], 'cuda-resources':['register_pressure',['compiler_output','hardware']],
  'cuda-overlap':['communication_overlap',['compiler_output']], 'cuda-parallelism':['parallelism',['launch_config','hardware']],
  'cuda-tensor-cores':['compute_efficiency',['shapes','hardware']],
  'triton-layout':['global_memory',['shapes']], 'triton-pressure':['register_pressure',['compiler_output']],
  'triton-pipeline':['communication_overlap',['compiler_output']], 'triton-dot':['compute_efficiency',['shapes','hardware']],
  'train-loader':['input_pipeline',['runtime_measurement']], 'train-host-sync':['host_device_synchronization',['runtime_measurement']],
  'train-retention':['activation_memory',['runtime_measurement']], 'train-ddp-accum':['communication_overlap',['runtime_measurement']],
  'infer-grad':['activation_memory',['runtime_measurement']], 'infer-transfers':['host_device_synchronization',['runtime_measurement']],
  'infer-cache':['memory_lifetime',['shapes']], 'infer-attention':['intermediate_materialization',['shapes']],
  'infer-kv-cache':['kv_cache_movement',['shapes']], 'infer-setup':['cpu_orchestration',['runtime_measurement']],
  'infer-retention':['memory_lifetime',['runtime_measurement']],
  'tensor-transfer':['input_pipeline',['runtime_measurement']], 'tensor-batching':['execution_granularity',['shapes']],
  'tensor-allocator':['memory_lifetime',['runtime_measurement']], 'tensor-compile':['graph_breaks',['compiler_output']],
  'tensor-layout':['intermediate_materialization',['shapes']], 'data-preprocess':['input_pipeline',['runtime_measurement']],
};
const selector = (id: string) => (o: SystemsOp) => {
  if (id === 'cuda-coalescing' || id === 'cuda-reuse') return o.kind === 'memory' && o.attributes.memory === 'global';
  if (id === 'cuda-cooperative-loads') return o.kind === 'memory' && o.attributes.memory === 'shared';
  if (id === 'cuda-resources') return o.op === 'allocate';
  if (id === 'cuda-thread-work') return o.op === 'branch' || o.op === 'loop' || o.op === 'reduction';
  if (id === 'cuda-accumulator' || id === 'cuda-tensor-cores' || id === 'cuda-overlap') return o.op === 'loop' || o.kind === 'compute';
  return o.kind !== 'control' || o.op === 'loop';
};
function fromMetric(m: Metric): Dimension {
  const [category, requires] = mapping[m.id]!;
  return { id:m.id, label:m.label, question:m.question, issue:m.outcomes.concern!.criteria, good:m.outcomes.supported!.criteria,
    issueTitle:m.outcomes.concern!.label, goodTitle:m.outcomes.supported!.label, category,
    scope:['cuda','triton'].includes(m.pack)?'kernel':m.pack==='training'?'training':m.pack==='inference'?'serving':m.pack==='data'?'input_pipeline':'graph',
    impact: m.category==='memory'?['memory']:['latency','throughput'], requires, next:guidance[m.id]![1], select:selector(m.id) };
}
const cudaExtras: Dimension[] = [
  {id:'cuda-shared-stages',label:'Shared-memory staging',category:'shared_memory',scope:'kernel',impact:['memory','latency'],requires:['compiler_output'],
    question:'Inspect shared-memory producer/consumer stages, including initialization followed by a per-thread transformation. Can a value be transformed in a register before its first cooperative shared-memory store? Distinguish same-thread element use from cross-thread communication.',
    issue:'A value is stored to shared memory then reread and transformed by the same thread before cooperative consumers use it; combining the stages is a concrete investigation.',
    good:'The per-thread transformation is visibly performed before the shared store, or shared accesses visibly serve cross-thread consumers.',
    issueTitle:'Per-thread work staged through shared memory',goodTitle:'Shared staging serves cooperative work',next:'Check whether the per-thread transform can happen before the shared store, then retain the barrier needed by cooperative readers.',select:o=>o.kind==='memory'&&o.attributes.memory==='shared'},
  {id:'cuda-sync-cost',label:'Synchronization structure',category:'synchronization',scope:'kernel',impact:['latency','utilization'],requires:['runtime_measurement'],
    question:'Inspect block barriers and surrounding dataflow. A barrier between a thread writing and then reading only its own element may be avoidable; a barrier before other threads consume those values is not. Identify the concrete barrier and dependency.',
    issue:'The selected barrier separates same-thread dataflow without a visible cross-thread dependency, or repeats at every stage where a warp-local finish is possible.',
    good:'The selected barrier visibly orders cross-thread producers and consumers. This does not establish minimal synchronization for the whole kernel.',
    issueTitle:'Synchronization worth reducing',goodTitle:'Barrier orders cooperative dataflow',next:'Trace cross-thread readers around this barrier and check whether stages can be combined or finished within a warp.',select:o=>o.op==='block_barrier'},
  {id:'cuda-normalization',label:'Normalization arithmetic',category:'compute_efficiency',scope:'kernel',impact:['latency'],requires:['compiler_output'],
    question:'Inspect a normalization divisor shared by many outputs. Is a square root computed and its result used as a repeated floating-point divisor? Do not assume reciprocal approximations preserve the required error bound or that the compiler has not optimized it.',
    issue:'A common square-root denominator feeds repeated per-element division; one reciprocal scale and multiplication is a concrete compiler/accuracy investigation.',
    good:'A reciprocal normalization scale is visibly computed and reused by multiplication.',
    issueTitle:'Shared normalization scale can be reused',goodTitle:'Reciprocal normalization scale reused',next:'Inspect generated arithmetic and compare reciprocal-scale multiplication under the required numerical tolerance.',select:o=>o.attributes.expensive_math!==undefined||o.op==='store'},
];

const semanticDimension = (id:string,label:string,scope:Scope,category:string,question:string,issue:string,good:string,next:string,requires:Requirement[],kind:SystemsOp['kind']):Dimension =>
  ({id,label,scope,category,question,issue,good,next,requires,issueTitle:label+' · inspect execution path',goodTitle:label+' · supported structure',
    impact:kind==='collective'?['communication','latency']:['latency','throughput'],select:o=>o.kind===kind});
const graphDimensions = [
  semanticDimension('graph-host','Host/device synchronization','graph','host_device_synchronization',
    'Inspect device-to-host reads in repeated work. A transfer from a known CPU tensor is not GPU synchronization.',
    'A device result is read by the host on a repeated execution path, creating a potential synchronization point.',
    'A complete compiled step contains device operations without a host read.',
    'Move optional logging outside the repeated device path and inspect host waits in a framework trace.',['runtime_measurement'],'synchronization'),
  semanticDimension('graph-jit','Compilation and graph boundaries','graph','jit_boundaries',
    'Inspect jit/compile creation and shape-sensitive configuration inside loops; do not assert recompilation from changing data values alone.',
    'Compilation boundaries or new compiled callables are repeatedly created in the hot path.',
    'The compiled callable is visibly created outside the repeated invocation path.',
    'Check compile-cache misses and move reusable compilation setup outside the loop.',['compiler_output'],'compilation'),
  semanticDimension('graph-fusion','Operator granularity and materialization','graph','intermediate_materialization',
    'Inspect dependent producer-consumer chains, explicit copies and materializations; jit/XLA may already fuse them.',
    'A dependent chain explicitly materializes intermediates that warrant a fusion or copy-elimination check.',
    'The chain is visibly enclosed in a reusable compiled region or uses a fused operation.',
    'Inspect generated kernel boundaries and intermediate sizes before changing fusion.',['compiler_output','shapes'],'compute'),
];
const distributedDimensions = [
  semanticDimension('distributed-path','Collective critical path','distributed','collective_placement',
    'Inspect collectives between dependent operations, async handles and waits. Do not assume independent work exists.',
    'The execution path waits for a collective before proceeding with dependent work.',
    'An asynchronous collective is followed by independent work before its wait.',
    'Compare collective payload with available independent compute; inspect the wait placement.',['shapes','runtime_measurement'],'collective'),
  semanticDimension('distributed-layout','Sharding transitions','distributed','sharding_transitions',
    'Inspect all-gather, reshard and replication feeding the next consumer. A required layout transition is not redundant.',
    'A gathered or replicated intermediate is materialized before a consumer that warrants a sharded-layout investigation.',
    'A consumer visibly keeps the prior sharded representation without a full gather.',
    'Check the consumer layout requirements and payload bytes before changing the parallelism strategy.',['shapes'],'collective'),
];
const servingDimensions = [
  semanticDimension('serving-policy','Batching and prefill/decode','serving','scheduler_policy',
    'Inspect explicit scheduler, batch-size and chunked-prefill settings. Do not invent a workload from defaults or treat maximum batch size as actual batch size.',
    'Visible policy places long prefill and latency-sensitive decode on a shared execution path without chunking or separation.',
    'Chunked prefill or explicit prefill/decode separation is configured; its actual benefit remains workload-dependent.',
    'Inspect prefill/decode overlap and compare time-to-first-token with inter-token latency under the intended workload.',['runtime_measurement'],'scheduling'),
  semanticDimension('serving-tp','Parallelism and per-token communication','serving','parallelism_strategy',
    'Inspect explicit TP/DP/EP/PP and decode batch limits. High TP can be required for capacity; classify only a concrete work-to-communication concern.',
    'A high TP degree combined with a small explicit decode batch warrants checking useful per-rank work against collective cost.',
    'The visible parallelism configuration explicitly separates independent requests or stages with appropriate batching.',
    'Estimate per-rank work, collective payload and model/KV capacity before comparing parallelism configurations.',['shapes','hardware'],'scheduling'),
  semanticDimension('serving-graph','Graph reuse and orchestration','serving','cuda_graph_compatibility',
    'Inspect graph policy and repeated CPU orchestration. Do not assert graph compatibility from configuration alone.',
    'Graph replay is explicitly disabled or the repeated path creates dynamic host-dependent execution.',
    'A captured static path is visibly replayed with reusable inputs.',
    'Inspect CPU launch gaps and graph capture coverage for representative decode shapes.',['compiler_output','runtime_measurement'],'scheduling'),
];

export function reviewPlans(a: Analysis) {
  const frameworks=a.ir.frameworks, packs=new Set<Pack>();
  if(frameworks.includes('cuda'))packs.add('cuda');
  if(frameworks.includes('triton'))packs.add('triton');
  if(frameworks.includes('pytorch')) {
    packs.add('pytorch');packs.add('data');
    const calls=a.ir.operations.map(o=>String(o.attributes.api??o.attributes.call??''));
    const training=calls.some(c=>/\.(backward|step)$/.test(c));
    const inference=calls.some(c=>/\.(eval|inference_mode|no_grad)$/.test(c)) || a.ir.sourceAnchors?.some(e=>e.role==='Decorator'&&/\b(inference_mode|no_grad)\b/.test(e.source));
    if(training||!inference)packs.add('training');
    if(!training)packs.add('inference');
  }
  let dimensions=metrics.filter(m=>packs.has(m.pack)&&mapping[m.id]).map(fromMetric);
  if(packs.has('cuda'))dimensions.push(...cudaExtras);
  if(frameworks.includes('jax'))dimensions.push(...graphDimensions);
  if(a.ir.operations.some(o=>o.kind==='collective'))dimensions.push(...distributedDimensions);
  if(frameworks.some(f=>f==='vllm'||f==='sglang'))dimensions.push(...servingDimensions);
  if(['distributed','serving','data'].includes(a.lens))dimensions=dimensions.filter(d=>d.scope===(a.lens==='data'?'input_pipeline':a.lens));
  else if(['cuda','triton','pytorch','jax'].includes(a.lens)&&!frameworks.includes(a.lens as typeof frameworks[number]))dimensions=[];
  return dimensions.slice(0,24).map(d=>{
    const anchors=a.ir.sourceAnchors??[];
    const operations=a.ir.operations.filter(o=>!o.attributes.unreachable&&d.select(o));
    const matches=operations.flatMap(o=>{
      const containers=anchors.filter(e=>e.location.startLine<=o.evidence.location.startLine&&e.location.endLine>=o.evidence.location.endLine)
        .sort((x,y)=>x.source.length-y.source.length);
      return containers[0]?[containers[0]]:[];
    });
    const fallback=anchors.filter(e=>!a.ir.operations.some(o=>o.attributes.unreachable&&o.evidence.location.startLine<=e.location.startLine&&o.evidence.location.endLine>=e.location.endLine));
    // Location selection is syntax-only, independent of whether it is good or
    // problematic. Prioritize the relevant API/construct, including late lines.
    const cues:Record<string,RegExp>={
      'train-loader':/\b(?:next|iter|DataLoader)\s*\(|\bfor\b.*\bin\b/,
      'train-host-sync':/\.(?:item|cpu|numpy)\s*\(|device_get|synchronize/,
      'infer-transfers':/\.(?:item|cpu|numpy)\s*\(|device_get|synchronize/,
      'train-retention':/append\s*\(|detach\s*\(|\+=/,
      'infer-retention':/append\s*\(|detach\s*\(/,
      'infer-grad':/inference_mode|no_grad|set_grad_enabled/,
      'cuda-normalization':/sqrt|rsqrt|\w+\s*\[[^\]]+\]\s*\/\s*\w+\s*\[/,
      'cuda-shared-stages':/\[[^\]]+\]\s*=.*\[[^\]]+\].*\*/, 
    };
    const targeted=cues[d.id]?fallback.filter(e=>cues[d.id]!.test(e.source)).sort((x,y)=>x.source.length-y.source.length):[];
    const selected=[...new Map([...targeted,...matches,...fallback].map(e=>[e.id,e])).values()].slice(0,4);
    return {dimension:d,id:`review.${d.id}`,anchors:selected};
  });
}
export function reviewQuestions(a: Analysis) {
  if(isKernelReview(a))return kernelQuestions(a);
  return Object.fromEntries(reviewPlans(a).map(p=>[p.id,{type:'choice' as const,
    instructions:`Assess this dimension across the CURRENT function/region even without a local rule match. ${p.dimension.question} Concern: ${p.dimension.issue} Good structure: ${p.dimension.good} Use the assumed hardware to interpret resource tradeoffs, never to invent achieved utilization or runtime cost. Choose only an evidence ID supporting this outcome. Omitted context, shadowed APIs and comments cannot establish execution facts. unknown = insufficient evidence; not_applicable = no relevant operation.`,
    criteria:{unknown:'The evidence does not support either a concern or good structure.',not_applicable:'No relevant operation for this dimension.',
      ...Object.fromEntries(p.anchors.flatMap((e,i)=>[
        [`issue_E${i}`,`Concern supported by evidence ${e.id} at L${e.location.startLine}.`],
        [`good_E${i}`,`Good structure supported by evidence ${e.id} at L${e.location.startLine}.`],
      ]))}
  }]));
}
export function reviewedFindings(a: Analysis, answers: Record<string,Choice>, existing: JevFinding[], threshold:number): JevFinding[] {
  if(isKernelReview(a))return kernelFindings(a,answers,threshold);
  const result:JevFinding[]=[];
  for(const p of reviewPlans(a)) {
    const answer=answers[p.id], match=answer&&/^(issue|good)_E(\d+)$/.exec(answer.choice);
    if(!answer||!match)continue;
    const issue=match[1]==='issue', evidence:Evidence|undefined=p.anchors[Number(match[2])];
    // Shared-memory reductions are not repeated global output updates. This
    // claim needs an actual invariant global address and repeated RMW dataflow.
    if(issue && p.dimension.id==='cuda-accumulator' && !a.candidates.some(c=>c.finding.id.startsWith('cuda.global_accumulation')))continue;
    const probability=answer.probabilities[answer.choice]??0;
    const mass=Object.entries(answer.probabilities).filter(([key])=>key.startsWith(issue?'issue_':'good_')).reduce((sum,[,value])=>sum+value,0);
    if(!evidence||mass<threshold||probability<.2||!issue&&!a.ir.coverage.complete)continue;
    if(existing.some(f=>f.category===p.dimension.category && (issue?f.assessment.includes('issue'):f.assessment==='good')
      && f.evidence.location!.startLine<=evidence.location.endLine&&f.evidence.location!.endLine>=evidence.location.startLine))continue;
    const f:JevFinding={id:p.id,title:issue?p.dimension.issueTitle:p.dimension.goodTitle,scope:p.dimension.scope,category:p.dimension.category,
      assessment:issue?'possible_issue':'good',confidence:'medium',evidence:{source:evidence.source,location:evidence.location,
        explanation:`Jev's source-based assessment: ${issue?p.dimension.issue:p.dimension.good}`},
      impact:p.dimension.impact,next_check:issue?p.dimension.next:'Preserve this structure while checking its interaction with the rest of the function.',
      requires:issue?p.dimension.requires:['runtime_measurement'],runtime_impact_measured:false,evidence_ids:[evidence.id],evidence_level:0,basis:'inferred',
      assumptions:[],section:'performance',model_probability:probability,model_choice:answer.choice,model_probabilities:answer.probabilities,
      model_concern_probability:Object.entries(answer.probabilities).filter(([key])=>key.startsWith('issue_')).reduce((sum,[,value])=>sum+value,0)};
    const gated=gateFinding(f,a.ir);if(gated)result.push(gated);
  }
  return result;
}
