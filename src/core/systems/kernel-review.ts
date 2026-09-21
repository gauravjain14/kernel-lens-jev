import type { AssessmentReport, Choice } from '../../live-types';
import type { Analysis } from './evaluate';
import { gateFinding } from './gate';
import type { Assessment, Evidence, Impact, JevFinding, Requirement, SystemsOp } from './types';

type Select = (op: SystemsOp) => boolean;
interface Bucket {
  key:string; label:string; assessment:Assessment; criteria:string; next:string; requires:Requirement[]; select:Select; fact:boolean;
}
interface Dimension {
  id:string; label:string; group:string; metric:string; question:string; buckets:Bucket[]; missing:string; requires:Requirement[]; impact:Impact[];
}
const op=(name:string):Select=>o=>o.op===name;
const attr=(name:string,value?:string):Select=>o=>value===undefined?!!o.attributes[name]:o.attributes[name]===value;
const any=(...selectors:Select[]):Select=>o=>selectors.some(s=>s(o));
const memory:Select=o=>o.kind==='memory';
const compute:Select=o=>o.kind==='compute';
const loops=op('loop'), branches=op('branch');
const globalMemory:Select=o=>o.attributes.memory==='global'||o.attributes.from_memory==='global'||o.attributes.api==='triton.language.load'||o.attributes.api==='triton.language.store';
const shared:Select=o=>o.attributes.memory==='shared'||o.attributes.to_memory==='shared';
const matrix:Select=o=>o.op==='gemm';
const asyncCopy:Select=o=>o.op==='copy'&&o.attributes.async===true;
const waits:Select=o=>o.kind==='synchronization';
const local:Select=o=>o.op==='allocate'&&o.attributes.memory==='local';
const reduction:Select=o=>o.op==='reduction'||o.attributes.reduction===true;
const source=(pattern:RegExp):Select=>o=>pattern.test(String(o.attributes.header??o.attributes.address??o.evidence.source).replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g,''));
const b=(key:string,label:string,assessment:Assessment,criteria:string,next:string,requires:Requirement[],select:Select,fact=false):Bucket=>({key,label,assessment,criteria,next,requires,select,fact});
const good=(key:string,label:string,criteria:string,next:string,requires:Requirement[],select:Select,fact=false)=>b(key,label,'good',criteria,next,requires,select,fact);
const issue=(key:string,label:string,criteria:string,next:string,requires:Requirement[],select:Select)=>b(key,label,'possible_issue',criteria,next,requires,select);
const dim=(id:string,label:string,group:string,metric:string,question:string,requires:Requirement[],buckets:Bucket[],impact:Impact[]=['latency','throughput']):Dimension=>({
  id:'review.cuda-'+id,label,group,metric,question,requires,buckets,impact,missing:`${label} needs ${requires.map(r=>r.replaceAll('_',' ')).join(', ')} or a visible execution relationship to distinguish its buckets.`,
});

/** Independent axes: expressing overlap never establishes adequate depth or occupancy. */
export const kernelDimensions:Dimension[]=[
  dim('tensor-cores','Compute path','Compute','Compute Workload Analysis / tensor pipeline',
    'Identify executed compute primitives. Resolve supplied helper contracts; never infer instructions from a helper name. Single-thread tcgen05 issue dispatches matrix work and is not serial scalar GEMM.', ['compiler_output'],[
      good('tcgen05','Blackwell asynchronous MMA','A resolved tcgen05.mma instruction implements matrix compute. This establishes the compute path, not Tensor Core saturation.','Inspect tensor-pipeline activity and whether operands and accumulator storage keep MMA supplied.',['runtime_measurement'],attr('matrix_family','tcgen05'),true),
      good('warpgroup','Warpgroup asynchronous MMA','Resolved wgmma.mma_async implements the matrix work.','Inspect outstanding MMA groups and their waits alongside operand delivery.',['runtime_measurement'],attr('matrix_family','wgmma'),true),
      good('matrix','Warp or compiler matrix primitive','WMMA, mma.sync, or a recognized Triton dot expresses matrix computation. Generated instruction selection remains compiler-dependent.','Inspect generated matrix instructions for the selected dtype and tile.',['compiler_output'],o=>matrix(o)&&!['tcgen05','wgmma'].includes(String(o.attributes.matrix_family))),
      issue('scalar_contraction','Scalar matrix contraction','A visible matrix contraction is implemented as scalar multiply-accumulate loops. Do not apply to arbitrary elementwise multiplication.','Check whether matrix instructions fit the dtype, tile shapes and accuracy contract.',['shapes','hardware'],attr('scalar_accumulation')),
      good('simt','SIMT compute path','Visible elementwise or non-matrix work uses ordinary per-thread arithmetic; matrix instructions are not implied to be useful.','Inspect instruction mix and dependency chains for this operation.',['compiler_output'],o=>compute(o)&&!matrix(o)),
    ]),
  dim('thread-work','Lane participation','Compute','Scheduler Statistics / active threads',
    'Distinguish substantial scalar work under a narrow branch from hardware-defined elected-thread issue, initialization and final stores.', ['launch_config'],[
      issue('serial_reduction','Serial reduction tail','A narrow thread branch performs a multi-element scalar reduction while other lanes do not contribute.','Compare a warp reduction for the tail under the actual participating-lane contract.',['launch_config'],reduction),
      issue('narrow_work','Substantial scalar work on few lanes','A narrow lane branch owns substantial serial work, not just elected hardware issue or an output store.','Inspect whether independent parts of that work can be distributed across lanes.',['launch_config'],branches),
      good('elected_issue','Elected-thread matrix issue','A narrow branch issues resolved tcgen05 MMA; its single issuer does not mean one thread performs all scalar arithmetic.','Inspect asynchronous issue/completion cadence rather than treating the issuer branch as scalar serialization.',['runtime_measurement'],attr('issuer','single_thread'),true),
      good('distributed','Work distributed across lanes','Visible indexing or collective operations distributes useful work across participating lanes.','Check active-lane efficiency for representative shapes and tails.',['launch_config'],any(globalMemory,reduction)),
    ],['utilization','latency']),
  dim('coalescing','Global access pattern','Memory','Memory Workload Analysis / sectors per request',
    'Trace adjacent lanes through address arithmetic and vector width. Contiguous logical indexing alone is not a measured transaction count.', ['launch_config','shapes'],[
      good('contiguous','Adjacent lanes cover contiguous spans','Visible lane mapping assigns adjacent lanes adjacent scalar or vector spans of global memory.','Check sectors per request with the actual base alignment and row stride.',['launch_config','runtime_measurement'],globalMemory),
      good('broadcast','Lane-uniform operand address','Several lanes visibly read the same operand address, enabling a broadcast/reuse pattern; cache behavior is unmeasured.','Inspect broadcast/cache behavior alongside reuse across warps.',['runtime_measurement'],globalMemory),
      issue('strided','Strided lane addresses','Adjacent lanes visibly access separated locations due to a non-unit lane stride.','Compare the lane-to-address map with memory sectors per request.',['shapes','runtime_measurement'],globalMemory),
      issue('indirect','Indirect gather/scatter','Global addresses depend on loaded indices, preventing a contiguous-access conclusion.','Inspect index locality and sectors per request for representative inputs.',['runtime_measurement'],globalMemory),
    ],['memory','throughput']),
  dim('transfer-width','Transfer granularity','Memory','Instruction Statistics / global memory requests',
    'Separate explicit 16-byte copies or vector loads/stores from compiler vectorization. Alignment declarations on shared arrays do not prove global pointer alignment.', ['compiler_output'],[
      good('vector','Vector-width transfers expressed','Explicit vector accesses or fixed-width async copies transfer multiple scalar elements per participating lane.','Verify generated transfer width, global alignment and tail handling.',['compiler_output','shapes'],any(asyncCopy,source(/float[24]|int[24]|\.v[24]\./))),
      good('tma','Bulk tensor transfer expressed','Resolved cp.async.bulk/TMA expresses bulk transfers through a descriptor.','Inspect tensor-map layout and producer issue overhead.',['compiler_output','runtime_measurement'],attr('transfer','tma'),true),
      issue('scalar','Scalar transfer instruction risk','Many contiguous scalar accesses in repeated work are expressed separately with no explicit vector transfer. The compiler may combine them.','Inspect generated load/store widths before changing source vectorization.',['compiler_output'],globalMemory),
    ],['throughput','memory']),
  dim('reuse','Operand reuse','Memory','Memory Workload Analysis / DRAM and L2 traffic',
    'Distinguish one-use streaming from reusable GEMM/stencil operands. Do not flag one-use elementwise input as missing tiling.', ['shapes','compiler_output'],[
      good('shared_tiles','Shared tiles feed repeated computation','Global operands are staged in shared tiles and consumed by multiple computations.','Inspect tile reuse versus shared-memory footprint and repeated loads across blocks.',['shapes','runtime_measurement'],any(shared,matrix)),
      good('register_reuse','Register-held operands reused','Loaded scalar/vector operands visibly serve multiple computations before replacement.','Inspect register usage and generated load elimination.',['compiler_output'],any(compute,local)),
      good('streaming','One-use streaming access','Each input is consumed once for an elementwise or streaming operation; additional tiling is not established as useful.','Compare achieved bandwidth with the actual input/output traffic.',['runtime_measurement'],globalMemory),
      issue('global_reload','Reusable operands repeatedly read globally','A contraction or stencil rereads reusable global operands without a visible cooperative tile or register-reuse scheme.','Inspect whether cooperative tiling can reduce repeated operand traffic.',['compiler_output','shapes'],globalMemory),
    ],['memory','throughput']),
  dim('shared-layout','Shared-memory layout','Memory','Memory Workload Analysis / bank conflicts',
    'Identify swizzles, padding, broadcast and stride. A swizzle is evidence of an intended layout, not proof of conflict freedom or descriptor compatibility.', ['launch_config','compiler_output'],[
      good('swizzled','Swizzled shared layout','XOR/permuted shared addresses or a resolved descriptor swizzle is explicitly constructed.','Check the producer layout against the consuming MMA/load descriptor and bank-conflict counters.',['launch_config','compiler_output'],any(shared,source(/swz|swizzle|\^/))),
      good('padded','Padded shared tile','Shared tile dimensions include padding intended to separate bank addresses.','Verify the consumer lane mapping and compiled bank behavior.',['launch_config','runtime_measurement'],shared),
      good('broadcast','Shared broadcast pattern','Multiple participating lanes visibly consume the same shared-memory location.','Check multicast/broadcast behavior under the actual consumer instruction.',['compiler_output'],shared),
      issue('strided','Shared bank-conflict candidate','A visible lane stride can repeatedly map distinct shared addresses to the same bank; do not claim measured conflicts.','Compute the bank mapping for this access and inspect shared-memory conflict counters.',['launch_config','compiler_output'],shared),
    ],['memory','latency']),
  dim('shared-stages','Shared staging','Memory','Memory Workload Analysis / shared instructions',
    'Separate cooperative exchanges from values needlessly round-tripped by the same thread.', ['compiler_output'],[
      good('cooperative','Shared storage serves cooperative consumers','Shared stores feed different lanes or matrix operations rather than only the writing lane.','Preserve required producer-consumer ordering while inspecting shared traffic.',['runtime_measurement'],shared),
      issue('round_trip','Per-thread shared-memory round trip','A thread stores a value to shared memory and rereads/transforms its own value before any cooperative consumer.','Check whether the per-thread transform can precede the first shared store.',['compiler_output'],shared),
    ],['memory','latency']),
  dim('overlap','Copy/compute scheduling','Pipeline','Warp State Statistics / long scoreboard',
    'Follow issue, commit, compute, waits and stage reuse in source order. A wait after overlapping work is not proof of serialization. Presence of async alone is not proof of overlap.', ['runtime_measurement'],[
      good('overlapped','Next tile copied during current compute','The next tile copy is issued before computation on a different current tile, with waits before reuse/consumption. This exposes overlap; its sufficiency is unmeasured.','Inspect whether compute covers copy latency and where the remaining waits occur.',['runtime_measurement'],asyncCopy),
      issue('immediate_wait','Async copy immediately waited on','Repeated async copies are waited on before any independent compute is placed between issue and wait.','Check whether future-tile prefetch can move ahead of independent current-tile compute.',['runtime_measurement'],any(asyncCopy,waits)),
      issue('synchronous','Load and compute alternate synchronously','A repeated tiled computation loads its operands synchronously, computes, then begins the next tile.','Inspect whether prefetch or async staging can overlap independent tiles.',['compiler_output','runtime_measurement'],loops),
    ],['latency','utilization']),
  dim('pipeline-depth','Pipeline buffering','Pipeline','Scheduler Statistics / eligible warps',
    'Classify storage stage count separately from useful in-flight depth. Two buffers do not establish that latency is hidden. Do not claim more stages are automatically better.', ['shapes','compiler_output','runtime_measurement'],[
      good('double','Two storage stages expressed','Two alternating operand buffers are explicitly declared and indexed; useful overlap depth remains unmeasured.','Compare exposed waits against the extra shared-memory cost before changing stage count.',['compiler_output','runtime_measurement'],shared,true),
      good('multistage','Multiple pipeline stages expressed','Three or more stage slots or explicit compiler pipeline stages are visible; this does not establish optimal depth.','Inspect wait distance, resident blocks and useful in-flight stages together.',['compiler_output','runtime_measurement'],any(shared,loops)),
      issue('single','Single staging buffer','Repeated load/compute work reuses a single staging slot, requiring completion before overwrite.','Check whether a second slot would expose useful overlap within the resource budget.',['compiler_output','runtime_measurement'],shared),
    ],['memory','utilization']),
  dim('mma-cadence','MMA completion cadence','Pipeline','Compute Workload Analysis / tensor active and waits',
    'Trace asynchronous MMA issue and completion. Waiting after each tile bounds in-flight MMA groups; it does not by itself prove idle Tensor Cores. Account for concurrent copy work.', ['runtime_measurement'],[
      issue('tile_wait','MMA completion waited on each tile','Each mainloop tile issues a bounded MMA group and waits for completion before issuing the next tile group.','Inspect tensor-pipeline gaps at the wait; check whether dependencies permit additional in-flight MMA work.',['compiler_output','runtime_measurement'],any(attr('matrix_family','tcgen05'),attr('matrix_family','wgmma'),attr('protocol','mbarrier'))),
      good('groups','Multiple MMA groups can be in flight','The source explicitly permits additional independent MMA groups before waiting for prior completion.','Check accumulator and operand lifetime constraints alongside tensor-pipeline activity.',['compiler_output','runtime_measurement'],matrix),
      good('synchronous','Synchronous warp MMA path','The matrix path uses synchronous warp MMA rather than an asynchronous group protocol.','Inspect instruction dependencies and independent accumulator work.',['compiler_output'],o=>matrix(o)&&!o.attributes.async),
    ],['latency','utilization']),
  dim('sync-cost','Block synchronization','Pipeline','Warp State Statistics / barrier',
    'Identify why each barrier is present. Never suggest deleting a barrier without tracing cross-thread dependencies.', ['runtime_measurement'],[
      issue('per_tile','Block rendezvous on every tile','A block-wide barrier appears on each mainloop iteration, placing all participating warps at a rendezvous. It may be necessary.','Inspect arrival imbalance and whether producer/consumer scope can be narrowed while preserving dependencies.',['runtime_measurement'],op('block_barrier')),
      issue('reduction_stages','Barrier at every reduction stage','A shrinking reduction continues to use block-wide barriers in late stages.','Check whether the final warp can finish with warp-scoped collectives.',['launch_config'],op('block_barrier')),
      good('handoff','Barrier orders cooperative handoff','The visible barrier separates cross-thread producers and consumers; no redundant barrier claim is established.','Check barrier stalls and producer arrival skew before changing synchronization.',['runtime_measurement'],op('block_barrier')),
      good('warp','Warp-scoped synchronization','The visible communication uses warp-scoped synchronization or shuffle operations.','Check participation masks and whether all dependencies remain within the warp.',['launch_config'],any(op('warp_barrier'),attr('algorithm','warp_shuffle'))),
    ],['latency','utilization']),
  dim('async-protocol','Asynchronous ordering','Pipeline','Source Counters / asynchronous waits',
    'Describe visible protocols, not a correctness proof. Missing helper definitions require unknown, not an invented missing wait.', ['compiler_output'],[
      good('copy_and_matrix','Separate copy and MMA completion protocols','Async copies and asynchronous MMA use distinct visible completion mechanisms before their consumers/reuse.','Trace each buffer lifetime through both protocols, including prologue and tail.',['compiler_output'],any(asyncCopy,attr('protocol','mbarrier'),attr('protocol','matrix'))),
      good('copy','Copy commit/wait protocol expressed','Recognized async-copy issue/completion is visible without a separate asynchronous MMA protocol. This does not verify every dependency.','Trace the copied tile from issue through completion to its first consumer.',['compiler_output'],asyncCopy),
      good('matrix','Matrix completion protocol expressed','Recognized asynchronous matrix work uses a completion mechanism without a separate asynchronous copy protocol.','Check phase/group progression and accumulator/operand lifetime across iterations.',['compiler_output'],matrix),
    ]),
  dim('accumulator','Accumulator placement','Compute','Memory Workload Analysis / output traffic',
    'Identify actual accumulator storage. TMEM accumulation is not a global output update. Require repeated writes to the same global output before choosing global_rmw.', ['compiler_output'],[
      good('tensor_memory','Accumulator held in tensor memory','Resolved tcgen05 MMA accumulates into TMEM and the epilogue reads it before output stores.','Inspect TMEM allocation lifetime and epilogue transfer cost.',['runtime_measurement'],attr('matrix_family','tcgen05'),true),
      good('register','Accumulator retained locally','A local scalar/vector/fragment accumulates repeated work before the final output store.','Inspect register pressure and whether accumulators spill.',['compiler_output'],any(compute,local)),
      issue('global_rmw','Repeated global output accumulation','The same global output address is repeatedly updated inside a loop. Shared-memory updates do not qualify.','Check whether a private accumulator and one final store preserve output ownership.',['compiler_output'],o=>o.attributes.memory==='global'&&o.attributes.read_modify_write===true),
    ],['memory','throughput']),
  dim('registers','Register pressure','Resources','Launch Statistics / registers and spills',
    'Source can expose live arrays, accumulators and unrolling. It cannot determine allocated registers or spills without compiler evidence.', ['compiler_output'],[
      issue('arrays','Per-thread array pressure candidate','A sizable per-thread array or many simultaneously live accumulator fragments can compete for registers. A small fixed temporary alone is insufficient.','Inspect compiled registers per thread and local-memory spill traffic.',['compiler_output'],local),
      issue('unrolling','Unrolled live-state pressure candidate','Aggressive explicit unrolling creates overlapping per-thread live values. Loop length alone is insufficient.','Inspect register/spill changes across unroll factors.',['compiler_output'],any(local,loops)),
    ],['memory','utilization']),
  dim('resources','Shared-memory footprint','Resources','Launch Statistics / occupancy limiters',
    'Use all array dimensions and dtype widths, not just the first dimension. Symbolic BM/BN/BK are unresolved unless actual definitions are supplied; comments are not configuration.', ['shapes','compiler_output'],[
      issue('multibuffer','Multiple shared tiles consume residency budget','Multiple staged operand tiles reserve per-block shared storage. This is a resource tradeoff, not proof of low occupancy.','Resolve symbolic tile bytes, then compare compiled shared allocation with resident-block limits.',['shapes','compiler_output'],o=>o.op==='allocate'&&o.attributes.memory==='shared'),
      good('bounded','Shared tile footprint explicitly bounded','Literal dimensions or supplied compiler metadata establish the shared-storage allocation. Bounded does not mean optimal.','Compare this allocation with launch shape and register limits on the selected GPU.',['compiler_output','launch_config'],o=>o.op==='allocate'&&typeof o.attributes.bytes==='number'&&o.attributes.memory==='shared'),
    ],['memory','utilization']),
  dim('parallelism','Launch parallelism','Resources','Launch Statistics / waves per SM',
    'Kernel index expressions do not establish the actual grid, problem dimensions or occupancy. B200 alone does not resolve launch size.', ['launch_config','shapes'],[
      issue('small_grid','Launch may underfill the GPU','Explicit grid and workload dimensions establish very few independent blocks relative to the supplied GPU.','Compare grid blocks with SM count and resource-limited resident blocks.',['hardware','compiler_output'],source(/blockIdx|program_id/)),
      issue('imbalanced','Uneven block work candidate','Visible data-dependent work or edge-heavy tiling creates a concrete block-work imbalance.','Inspect block work distribution and tail-wave duration for representative shapes.',['shapes','runtime_measurement'],any(loops,branches)),
    ],['utilization','throughput']),
  dim('instruction-cost','Arithmetic and address work','Compute','Instruction Statistics / math and integer pipelines',
    'Focus on repeated variable division, transcendental work and address construction. Do not assume compile-time divisions/modulos survive optimization.', ['compiler_output'],[
      issue('expensive','Repeated expensive arithmetic candidate','Repeated data-path division, sqrt or transcendental calls are visible on dependent work.','Inspect generated math instructions and allowed numerical approximations.',['compiler_output'],attr('expensive_math')),
      issue('addressing','Repeated runtime address construction','Hot-loop address generation repeatedly uses runtime division/modulo or substantial integer arithmetic. Constant specialization may remove it.','Inspect generated integer instructions and whether invariant address work is hoisted.',['compiler_output'],any(loops,globalMemory)),
      good('specialized','Compile-time address specialization expressed','Tile and lane address arithmetic uses explicit compile-time constants suitable for specialization. This does not prove generated instruction count.','Inspect generated address arithmetic for the actual specialization.',['compiler_output'],any(loops,globalMemory)),
    ]),
  dim('dependencies','Instruction dependencies','Compute','Scheduler Statistics / short scoreboard and math dependency',
    'Identify serial dependency chains separately from available thread-level parallelism. Do not invent a stall percentage.', ['compiler_output','runtime_measurement'],[
      issue('chain','Long dependent accumulation chain','A loop repeatedly updates the same scalar accumulator with little visible independent work in that thread.','Inspect dependency stalls and whether independent accumulators fit the register budget.',['compiler_output','runtime_measurement'],attr('scalar_accumulation')),
      good('independent','Independent compute streams expressed','Multiple independent accumulators or compute chains are visible before combination.','Inspect compiler scheduling and register pressure.',['compiler_output'],compute),
    ],['latency','utilization']),
  dim('reduction','Reduction organization','Compute','Scheduler Statistics / reduction and barrier work',
    'Classify reduction organization, including warp shuffle, shared tree and narrow serial tail. Matrix compute is not automatically a scalar reduction loop.', ['launch_config'],[
      good('warp','Warp shuffle reduction','A recognized warp shuffle participates in an accumulation reduction.','Check masks, active lanes and the final reduction extent.',['launch_config'],attr('algorithm','warp_shuffle'),true),
      good('tree','Cooperative reduction tree','A halving/shared or recognized block reduction distributes the reduction across threads.','Inspect the final stages for avoidable block synchronization.',['runtime_measurement'],reduction),
      issue('serial','Serial reduction','A single lane or narrow branch sums multiple elements serially.','Compare a cooperative tail when its size and participation permit.',['launch_config'],reduction),
    ]),
  dim('epilogue','Epilogue / output path','Memory','Memory Workload Analysis / stores and shared or TMEM reads',
    'Inspect output movement and waits separately from mainloop compute. Vector stores alone do not establish a fast epilogue.', ['compiler_output','runtime_measurement'],[
      issue('tmem_wait','TMEM chunks loaded and waited on repeatedly','The output loop loads a TMEM chunk, waits for it, then stores it before issuing the next chunk.','Inspect epilogue wait/store cost and whether independent loads can be grouped within register limits.',['compiler_output','runtime_measurement'],o=>o.attributes.memory==='tensor'&&o.op==='load'),
      good('vector_store','Vector output stores expressed','The epilogue groups multiple output elements into explicit vector stores.','Verify alignment, lane ownership and tail handling in generated stores.',['compiler_output','shapes'],source(/float[24]|\.v[24]\./)),
      issue('extra_staging','Output crosses an extra staging buffer','Computed results are materialized into another buffer before final output without an established consumer requirement.','Trace output ownership and check whether that staging is required.',['compiler_output'],any(shared,op('store'))),
    ],['memory','latency']),
  dim('atomics','Atomic contention','Memory','Memory Workload Analysis / atomic traffic',
    'Require actual atomic operations; neither barriers nor TMEM issue are atomics. Contention depends on destination distribution.', ['runtime_measurement'],[
      issue('contended','Many threads target shared atomic destinations','Visible address mapping makes many participating threads update the same small set of atomic destinations.','Inspect destination contention and whether hierarchical aggregation preserves semantics.',['shapes','runtime_measurement'],attr('atomic')),
      good('distributed','Atomic destinations distributed','Visible atomic addressing distributes updates over different destinations; workload-dependent contention remains unknown.','Inspect the destination histogram and atomic throughput.',['runtime_measurement'],attr('atomic')),
    ],['throughput','memory']),
  dim('control-flow','Control-flow balance','Compute','Source Counters / branch efficiency',
    'Separate uniform tile conditions, lane predicates and data-dependent branches. An elected hardware issuer is not evidence of expensive divergence.', ['launch_config','runtime_measurement'],[
      issue('data_dependent','Data-dependent lane paths','Lanes take different substantial execution paths based on per-element data.','Inspect branch efficiency and work balance for representative inputs.',['runtime_measurement'],branches),
      good('predicated_edges','Predicates handle boundary work','Lane predicates guard loads/stores or zero-fill edge tiles; inactive-lane cost depends on actual shapes.','Inspect edge-tile fraction and active-lane efficiency.',['shapes'],branches),
      good('uniform','Uniform control structure expressed','Loop/tile branches are uniform across participating lanes, apart from required elected issue.','Check generated branches and workload-dependent trip counts.',['compiler_output'],any(loops,branches)),
    ],['utilization','throughput']),
  dim('intensity','Compute / traffic balance','Resources','Speed Of Light / roofline',
    'Classify source structure, not the measured bottleneck. A tiled GEMM may still be memory-limited for some shapes. Runtime data is needed for a bottleneck claim.', ['shapes','compiler_output'],[
      issue('low_reuse','Low-reuse contraction traffic risk','Contraction operands are repeatedly fetched globally with little visible reuse, increasing traffic relative to useful arithmetic.','Estimate useful FLOPs per operand byte for the actual tile and compare with the selected hardware roofline.',['shapes','compiler_output'],globalMemory),
      good('reuse_rich','Reuse-rich tiled computation','Cooperative tiles or matrix primitives expose reuse across many arithmetic operations. The limiting resource is not established.','Resolve tile shapes and traffic, then compare compute, HBM and shared-memory ceilings.',['shapes','compiler_output'],any(matrix,shared)),
      good('streaming','Streaming traffic structure','A one-use streaming operation performs little arithmetic per input/output element.','Estimate required bytes and inspect achieved memory throughput.',['shapes','runtime_measurement'],globalMemory),
    ],['memory','throughput']),
  dim('shape-contract','Tile and launch assumptions','Assumptions','Launch Statistics / active lanes',
    'Report unresolved divisibility, vector alignment, participating-warps and tile-tail assumptions separately from performance. Do not assert correctness or an out-of-bounds defect.', ['shapes','launch_config'],[
      b('contract','Tile / launch contract needs confirmation','unknown','Integer tile counts, vector-width guards or warp-tied addressing depend on externally supplied dimensions and launch parameters.','Confirm tile divisibility, partial-vector handling, block size and output ownership for supported inputs.',['shapes','launch_config'],any(branches,globalMemory,shared),false),
    ],['correctness_assumption']),
];

export function isKernelReview(a:Analysis):boolean {
  return a.ir.frameworks.some(f=>f==='cuda'||f==='triton')&&['auto','general','cuda','triton'].includes(a.lens);
}
export function kernelPlans(a:Analysis) {
  if(!isKernelReview(a))return [];
  const ops=a.ir.operations.filter(o=>!o.attributes.unreachable);
  return kernelDimensions.map(d=>({dimension:d,id:d.id,buckets:d.buckets.map(bucket=>{
    let matches=ops.filter(bucket.select);
    const parents=(o:SystemsOp)=>ops.filter(p=>o.controls.includes(p.id)&&p.op==='loop');
    if(d.id==='review.cuda-mma-cadence'&&bucket.key==='tile_wait') {
      const mma=ops.filter(o=>matrix(o)&&o.attributes.async);
      matches=ops.filter(o=>o.attributes.action==='wait'&&['mbarrier','matrix'].includes(String(o.attributes.protocol))
        &&parents(o).some(loop=>mma.some(m=>m.controls.includes(loop.id))));
    }
    if(d.id==='review.cuda-sync-cost'&&bucket.key==='per_tile')matches=matches.filter(o=>parents(o).some(l=>!l.attributes.halving));
    if(d.id==='review.cuda-sync-cost'&&bucket.key==='reduction_stages')matches=matches.filter(o=>parents(o).some(l=>l.attributes.halving&&l.attributes.reduction));
    if(d.id==='review.cuda-async-protocol'&&bucket.key==='copy_and_matrix'&&!(ops.some(asyncCopy)&&ops.some(o=>matrix(o)&&o.attributes.async)))matches=[];
    if(bucket.key==='global_rmw'&&!a.candidates.some(c=>c.finding.id.startsWith('cuda.global_accumulation')))matches=[];
    if(d.id==='review.cuda-parallelism'&&bucket.key==='small_grid'&&!a.ir.enrichment?.launch?.grid)matches=[];
    if(d.id==='review.cuda-pipeline-depth'&&bucket.key==='double')matches=matches.filter(o=>Array.isArray(o.attributes.dimensions)&&o.attributes.dimensions[0]==='2');
    if(d.id==='review.cuda-registers')matches=matches.filter(o=>Number(o.attributes.bytes??0)>=128||Array.isArray(o.attributes.dimensions)&&o.attributes.dimensions.some(d=>!/^\d+$/.test(String(d))));
    // Use all semantic matches for eligibility; send a distributed sample of
    // exact call/statement anchors, not the first four large enclosing loops.
    const unique=[...new Map(matches.map(o=>[o.evidence.id,o])).values()];
    const selected=unique.length<=8?unique:Array.from({length:8},(_,i)=>unique[Math.round(i*(unique.length-1)/7)]!);
    return {bucket,operations:selected,evidence:selected.map(o=>o.evidence)};
  })}));
}
export function kernelQuestions(a:Analysis) {
  return Object.fromEntries(kernelPlans(a).filter(p=>p.buckets.some(b=>b.evidence.length)).map(p=>[p.id,{type:'choice' as const,
    instructions:`Classify ${p.dimension.label} independently. ${p.dimension.question} Use the full function and resolved helper instructions. Select unknown for missing relationships/configuration; not_applicable for absent work. Relevant lines: ${[...new Set(p.buckets.flatMap(b=>b.evidence.map(e=>e.location.startLine)))].join(',')}.`,
    criteria:{unknown:'The distinguishing source relationship or required configuration is unavailable.',not_applicable:'This execution path does not contain work relevant to this axis.',
      ...Object.fromEntries(p.buckets.filter(b=>b.evidence.length).map(({bucket})=>[bucket.key,bucket.criteria]))}
  }]));
}
export function kernelFindings(a:Analysis,answers:Record<string,Choice>,threshold:number):JevFinding[] {
  const findings:JevFinding[]=[];
  for(const p of kernelPlans(a)) {
    const answer=answers[p.id],selected=p.buckets.find(b=>b.bucket.key===answer?.choice);
    if(!selected?.evidence.length||!answer)continue;
    const probability=answer.probabilities[answer.choice]??0;
    if(probability<threshold)continue;
    const {bucket,evidence,operations}=selected,primary=evidence[0]!;
    const related=operations.flatMap(o=>(o.attributes.helper_evidence as string[]|undefined)??[]);
    const f:JevFinding={id:p.id,title:bucket.label,scope:'kernel',category:p.id.replace('review.cuda-','kernel_'),assessment:bucket.assessment,
      confidence:bucket.fact?'high':'medium',evidence:{source:primary.source,location:primary.location,explanation:bucket.criteria},
      impact:p.dimension.impact,next_check:bucket.next,requires:bucket.requires,runtime_impact_measured:false,
      evidence_ids:[...new Set([...evidence.map(e=>e.id),...related])],evidence_level:0,basis:bucket.fact?'static':'inferred',assumptions:[],
      section:p.dimension.group==='Assumptions'?'assumptions':'performance',model_probability:probability,model_choice:answer.choice,model_probabilities:answer.probabilities,
      supporting_evidence:[...evidence.slice(1),...related.flatMap(id=>a.ir.sourceAnchors?.filter(e=>e.id===id)??[])].slice(0,8).map(e=>({source:e.source,location:e.location})),
      model_concern_probability:p.buckets.filter(b=>b.bucket.assessment==='possible_issue').reduce((s,b)=>s+(answer.probabilities[b.bucket.key]??0),0)};
    const gated=gateFinding(f,a.ir);if(gated)findings.push(gated);
  }
  return findings;
}
export function kernelDimensionReports(a:Analysis,answers:Record<string,Choice>,findings:JevFinding[]):NonNullable<AssessmentReport['dimensions']> {
  return kernelPlans(a).map(p=>{
    const f=findings.find(f=>f.id===p.id),answer=answers[p.id],hasEvidence=p.buckets.some(b=>b.evidence.length);
    const selected=p.buckets.find(b=>b.bucket.key===answer?.choice);
    const optional=['review.cuda-atomics','review.cuda-reduction','review.cuda-mma-cadence','review.cuda-async-protocol','review.cuda-pipeline-depth'];
    const applicable=!(answer?.choice==='not_applicable'||!hasEvidence&&optional.includes(p.id));
    return {id:p.id,label:p.dimension.label,group:p.dimension.group,metricFamily:p.dimension.metric,assessment:f?.assessment??'unknown',findingIds:f?[f.id]:[],
      bucket:f?.title??(!applicable?'Not applicable':answer&&selected?'Uncertain classification':'Needs evidence'),
      explanation:f?.evidence.explanation??(answer&&selected?`Jev did not distinguish the available buckets confidently. ${p.dimension.missing}`:p.dimension.missing),
      nextCheck:f?.next_check??`Inspect ${p.dimension.metric} after supplying the missing context.`,requires:f?.requires??p.dimension.requires,
      confidence:f?.confidence??'low',probability:answer?.probabilities[answer.choice],applicable};
  });
}
export function kernelExecution(a:Analysis) {
  if(!isKernelReview(a))return undefined;
  const all=a.ir.operations.filter(o=>!o.attributes.unreachable),semantic=all.filter(o=>o.kind!=='control'||o.op==='loop'||o.op==='branch'||o.attributes.unknown_call&&!o.attributes.semantic_helper);
  const cap=180,operations=semantic.length<=cap?semantic:Array.from({length:cap},(_,i)=>semantic[Math.round(i*(semantic.length-1)/(cap-1))]!);
  const ids=new Map(all.map((o,i)=>[o.id,i]));
  return {contract:'Operations are in source order, not a runtime timeline. Control IDs identify enclosing loops/branches. A helper effect is proven to exist in its supplied definition, not guaranteed to execute on every path. Unknown calls may have other effects. Copy and MMA completion are separate protocols. No measured bottleneck is inferred.',
    operations:operations.sort((a,b)=>Number(a.attributes.source_order)-Number(b.attributes.source_order)).map(o=>({id:ids.get(o.id),kind:o.kind,op:o.op,controls:o.controls.map(id=>ids.get(id)),
      attributes:Object.fromEntries(Object.entries(o.attributes).filter(([k])=>!['source_order','source_end','mutated_names','opaque_memory','helper_evidence'].includes(k))),line:o.evidence.location.startLine})),
    omittedOperations:Math.max(0,semantic.length-cap),
    helperInstructions:a.ir.sourceAnchors?.filter(e=>e.role==='HelperInstruction').map(e=>({id:e.id,source:e.source,location:e.location})),
    allocations:a.ir.operations.filter(o=>o.op==='allocate').map(o=>({source:o.evidence.source,location:o.evidence.location,...o.attributes})),
    requiredContext:{launch:a.ir.enrichment?.launch??'Not supplied; do not infer grid/block dimensions from kernel indexing.',compiler:a.ir.enrichment?.compiler??'Not supplied; register allocation, spills and occupancy are unmeasured.'}};
}
