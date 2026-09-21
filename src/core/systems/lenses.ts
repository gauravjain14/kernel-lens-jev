import type { FindingCandidate, JevFinding, SystemsIR, SystemsOp, Scope, Impact, Requirement, Assessment, Confidence } from './types';

export class Findings {
  readonly candidates: FindingCandidate[] = [];
  constructor(readonly ir: SystemsIR) {}
  add(rule: string, ops: SystemsOp[], data: { title: string; category: string; scope: Scope; explanation: string; next: string;
    requires?: Requirement[]; impact?: Impact[]; assessment?: Assessment; confidence?: Confidence; assumptions?: string[];
    quantities?: JevFinding['quantities']; level?: JevFinding['evidence_level']; model?: boolean }) {
    if (!ops.length) return;
    const primary = ops[0]!.evidence, assessment = data.assessment ?? 'possible_issue';
    const section = data.impact?.includes('correctness_assumption') ? 'assumptions' : 'performance';
    const finding: JevFinding = {
      id: `${rule}:${this.candidates.filter(c => c.finding.id.startsWith(`${rule}:`)).length}`, title: data.title,
      scope: data.scope, category: data.category, assessment, confidence: data.confidence ?? (ops.every(o => o.evidence.direct) ? 'high' : 'medium'),
      evidence: { source: primary.source, location: primary.location, explanation: data.explanation }, evidence_ids: [...new Set(ops.map(o => o.evidence.id))],
      impact: data.impact ?? ['latency', 'utilization'], next_check: data.next, requires: data.requires ?? ['runtime_measurement'],
      runtime_impact_measured: false, evidence_level: data.level ?? 0, basis: ops.every(o => o.evidence.direct) && data.confidence !== 'medium' && !data.assumptions?.length ? 'static' : 'inferred',
      assumptions: data.assumptions ?? [], section, ...(data.quantities ? { quantities: data.quantities } : {}),
    };
    this.candidates.push({ finding, question: `Given the supplied operations and source context, assess this specific hypothesis: ${data.explanation} Missing information: ${finding.requires.join(', ')}.`,
      modelMayAssess: data.model ?? (assessment === 'possible_issue') });
  }
  parents(op: SystemsOp) { return this.ir.operations.filter(p => op.controls.includes(p.id)); }
  loops(op: SystemsOp) { return this.parents(op).filter(p => p.op === 'loop'); }
  producer(consumer: SystemsOp, input: string): SystemsOp | undefined {
    const latest = this.ir.operations.filter(o => o.region === consumer.region && o.outputs.includes(input) && Number(o.attributes.source_order) < Number(consumer.attributes.source_order))
      .sort((a, b) => Number(b.attributes.source_order) - Number(a.attributes.source_order))[0];
    return latest && latest.controls.join() === consumer.controls.join() ? latest : undefined;
  }
}

export function kernelLens(b: Findings) {
  const ops = b.ir.operations;
  if (b.ir.frameworks.includes('triton')) for (const op of ops.filter(o => o.attributes.api === 'triton.language.dot' || o.attributes.api === 'triton.language.sum')) b.add('triton.compute_primitive', [op], {
    title: op.op === 'gemm' ? 'Dot-product primitive expressed' : 'Parallel reduction primitive expressed', category: op.op === 'gemm' ? 'compute_efficiency' : 'reduction_collectives', scope: 'kernel', assessment: 'good',
    explanation: 'The kernel expresses this operation through a recognized Triton compute primitive. Instruction selection and efficiency depend on shapes, dtype and compiler output.',
    next: 'Inspect the generated instructions for the intended tile shape and dtype.', requires: ['shapes', 'compiler_output'], model: false });
  if (!b.ir.frameworks.includes('cuda')) return;
  for (const branch of ops.filter(o => o.op === 'branch' && /threadIdx\.x\s*(?:==\s*0|<\s*(?:1|2|4|8|16|32))\b/.test(String(o.attributes.header)))) {
    const loops = ops.filter(o => o.op === 'loop' && o.controls.includes(branch.id) && o.attributes.reduction === true && (Number(o.attributes.iterations ?? o.attributes.bound) > 1 || o.attributes.halving));
    for (const loop of loops) {
      const loads = ops.filter(o => o.op === 'load' && o.controls.includes(loop.id) && ['shared', 'global'].includes(String(o.attributes.memory)));
      if (!loads.length) continue;
      b.add('cuda.serial_reduction', [branch, loop, ...loads], { title: 'Serial reduction tail', category: 'reduction_collectives', scope: 'kernel',
        explanation: 'A narrow thread-x branch performs a multi-element accumulation in a loop. The reduction work within each participating thread is serial; the number of participating threads depends on the other block dimensions.',
        next: 'Compare a warp-level reduction for this tail using the actual block shape and input size.', requires: ['none'], model: false });
    }
  }
  for (const shuffle of ops.filter(o => o.op === 'reduction' && o.attributes.algorithm === 'warp_shuffle' && o.attributes.accumulates === true)) {
    const loop = b.loops(shuffle).find(o => o.attributes.halving);
    if (!loop || ops.some(o => o.op === 'block_barrier' && o.controls.includes(loop.id))) continue;
    b.add('cuda.warp_reduction', [shuffle, loop], { title: 'Warp shuffle reduction', category: 'reduction_collectives', scope: 'kernel', assessment: 'good',
      explanation: 'The reduction loop accumulates values exchanged by warp shuffle, without a block barrier inside that loop.',
      next: 'Check that the active-lane mask and participating lanes match the intended reduction.', requires: ['none'], model: false });
  }
  for (const loop of ops.filter(o => o.op === 'loop' && o.attributes.halving && o.attributes.reduction)) {
    const barriers = ops.filter(o => o.op === 'block_barrier' && o.controls.includes(loop.id));
    if (barriers.length) b.add('cuda.reduction_barriers', [loop, ...barriers], { title: 'Block barrier at every reduction stage', category: 'synchronization', scope: 'kernel',
      explanation: 'The halving reduction loop contains a block-wide barrier, including its final stages.', next: 'Inspect whether the final participating warp can finish without further block-wide barriers.' });
    if (/blockDim\.x/.test(String(loop.attributes.header))) b.add('cuda.power_of_two', [loop], { title: 'Reduction shape assumption', category: 'launch_shape_assumptions', scope: 'kernel', assessment: 'unknown',
      explanation: 'The reduction begins at half blockDim.x and repeatedly halves its stride. Handling of an unmatched element for a non-power-of-two width is not established here.',
      next: 'Check the supported block widths and how the reduction handles non-power-of-two widths.', requires: ['launch_config'], impact: ['correctness_assumption'], model: false });
  }
  const memory = ops.filter(o => o.kind === 'memory' && o.attributes.address);
  for (let i = 0; i < memory.length; i++) {
    const op = memory[i]!;
    if (op.op === 'load' && op.attributes.memory === 'global') {
      const prev = memory.slice(0, i).reverse().find(o => o.region === op.region && o.op === 'load' && o.attributes.address === op.attributes.address && o.controls.join() === op.controls.join());
      const interveningStore = prev && ops.some(o => (o.op === 'store' || o.attributes.unknown_call || typeof o.attributes.writes_symbol === 'string' && new RegExp(`\\b${o.attributes.writes_symbol}\\b`).test(String(op.attributes.address))) && Number(o.attributes.source_order) > Number(prev.attributes.source_order) && Number(o.attributes.source_order) < Number(op.attributes.source_order));
      if (prev && !interveningStore) b.add('cuda.redundant_load', [op, prev], { title: 'Repeated global address expression', category: 'global_memory', scope: 'kernel',
        explanation: `The same global address expression (${op.attributes.address}) is loaded more than once in this control region. The compiler may already reuse the value.`,
        next: 'Inspect generated loads to see whether the compiler eliminates the repeated source access.', requires: ['compiler_output'], impact: ['memory', 'latency'] });
    }
    if (op.op === 'store' && op.attributes.memory === 'global' && op.attributes.read_modify_write) {
      const loop = b.loops(op).find(l => l.attributes.variable && (l.attributes.iterations === undefined || Number(l.attributes.iterations) > 1) && !new RegExp(`\\b${l.attributes.variable}\\b`).test(String(op.attributes.address)) && !(l.attributes.mutated_names as string[] ?? []).some(name => new RegExp(`\\b${name}\\b`).test(String(op.attributes.address)))
        && !ops.some(o => o.attributes.unknown_call && o.controls.includes(l.id)));
      if (loop) b.add('cuda.global_accumulation', [op, loop], { title: 'Global accumulator inside a loop', category: 'global_memory', scope: 'kernel',
        explanation: `The loop repeatedly updates the same global output address (${op.attributes.address}) using a read-modify-write expression.`,
        next: 'Check whether a thread-local accumulator and one final store preserve the output ownership and dependencies.', requires: ['compiler_output'], impact: ['memory', 'latency'] });
    }
    const prior = memory.slice(0, i).find(o => o.region === op.region && o.attributes.address === op.attributes.address && b.parents(o).some(p => p.op === 'branch') && !b.parents(op).some(p => p.op === 'branch'));
    if (prior) b.add('cuda.guard_contract', [op, prior], { title: 'Access outside the earlier guard', category: 'launch_shape_assumptions', scope: 'kernel', assessment: 'unknown', confidence: 'medium',
      explanation: `The address ${op.attributes.address} is accessed outside the branch that guarded an earlier access. The supported launch and initialization contract are not established.`,
      next: 'Check the launch dimensions and whether every participating thread has a valid initialized element.', requires: ['launch_config', 'shapes'], impact: ['correctness_assumption'], model: false });
  }
  for (const op of ops.filter(o => o.op === 'allocate' && o.attributes.memory === 'local' && Number(o.attributes.bytes) >= 512)) b.add('cuda.local_storage', [op], {
    title: 'Large thread-local array', category: 'register_pressure', scope: 'kernel', explanation: 'A sizable array is declared per thread. Its placement in registers or local memory depends on compilation and indexing.',
    next: 'Inspect compiler register, local-memory and spill reports for this kernel.', requires: ['compiler_output'], impact: ['memory', 'utilization'] });
  const contiguous = ops.find(o => o.op === 'load' && o.attributes.memory === 'global' && o.attributes.index === 'threadIdx.x');
  if (contiguous) b.add('cuda.contiguous_index', [contiguous], { title: 'Adjacent thread-x indices read adjacent elements', category: 'global_memory', scope: 'kernel', assessment: 'good',
    explanation: 'This global load uses threadIdx.x directly as its element index. Actual memory transactions still depend on active lanes, alignment and launch shape.',
    next: 'Check the launch shape and alignment for the intended warp access pattern.', requires: ['launch_config'], model: false });
}

export function graphLens(b: Findings) {
  const ops = b.ir.operations;
  for(const op of ops.filter(o=>o.op==='batch'&&o.attributes.iterator_recreated&&b.loops(o).length)) {
    const loader=ops.find(o=>o.evidence.id===op.attributes.loader_evidence);
    if(loader)b.add('training.iterator_recreated',[op,loader],{title:'DataLoader iterator recreated each step',scope:'input_pipeline',category:'input_pipeline',
      explanation:'The loop calls next(iter(loader)) on a visibly constructed DataLoader, creating a fresh iterator for each step. Iterator setup is repeated and iteration does not advance through one persistent loader iterator.',
      next:'Iterate directly over the loader, or create one iterator outside this loop and advance it with next(iterator).',
      requires:['none'],impact:['throughput'],model:false});
  }
  for (const capture of ops.filter(o => o.op === 'graph_boundary' && o.attributes.capture_from !== undefined)) {
    const body = ops.filter(o => o.id !== capture.id && o.region === capture.region && Number(o.attributes.source_order) > Number(capture.attributes.source_end) && Number(o.attributes.source_end) <= Number(capture.attributes.capture_to));
    const sync = body.find(o => o.op === 'device_to_host' && o.attributes.input_device === 'gpu' || o.op === 'host_sync' && !o.attributes.paired);
    if (sync) b.add('graph.capture_host_read', [sync, capture], { title: 'Host read inside CUDA graph capture', category: 'cuda_graph_compatibility', scope: 'graph',
      explanation: 'A device-to-host read or host wait is inside the explicit CUDA graph capture region. This operation cannot be replayed as an ordinary captured GPU operation.',
      next: 'Move the host read outside capture and inspect graph capture and replay behavior.', requires: ['none'], impact: ['compilation', 'latency'] });
    else if (b.ir.coverage.complete && body.some(o => o.kind === 'compute') && body.filter(o => o.kind === 'compute').every(o => o.attributes.input_shape && o.attributes.input_device === 'gpu') && !body.some(o => o.attributes.unknown_call || o.kind === 'control' && ['branch', 'dynamic_shape'].includes(o.op) || o.op === 'device_to_host')) b.add('graph.capture_static', [capture, ...body.filter(o => o.kind === 'compute')], {
      title: 'Capture region has explicit fixed input shapes', category: 'cuda_graph_compatibility', scope: 'graph', assessment: 'good',
      explanation: 'The visible capture region uses recognized compute on GPU inputs with statically known shapes and has no observed host read or dynamic branch. Actual capture success and replay reuse remain unmeasured.',
      next: 'Confirm successful capture and reuse for the intended tensor addresses and shapes.', requires: ['runtime_measurement'], impact: ['compilation', 'latency'], level: 1, model: false });
  }
  for (const op of ops.filter(o => o.op === 'device_to_host' || o.op === 'host_sync' && !o.attributes.paired)) {
    const loops = b.loops(op); if (!loops.length) continue;
    const known = op.evidence.direct;
    b.add('graph.host_sync', [op, loops[0]!], { title: known ? 'Host read on the loop path' : 'Possible device-to-host boundary', category: 'host_device_synchronization', scope: 'graph',
      explanation: known ? 'The loop reads a device result on the host or explicitly waits for device work. This creates a host/device boundary on each executed iteration.' : 'The loop calls a host-transfer API. A device transfer or wait depends on the value’s actual device placement.',
      next: 'Check whether the host read can be batched or moved outside the critical loop.', requires: known ? ['none'] : ['hardware'], confidence: known ? 'high' : 'medium',
      assumptions: known ? [] : ['The value must reside on an accelerator for a device-to-host transfer to occur.'] });
  }
  for (const op of ops.filter(o => o.op === 'jit_boundary')) {
    if (b.loops(op).length && op.attributes.recreates_callable) b.add('graph.jit_lifetime', [op, b.loops(op)[0]!], { title: 'Compiled callable recreated inside a loop', category: 'jit_boundaries', scope: 'graph',
      explanation: 'The loop constructs a fresh callable and passes it to a compilation wrapper. Cache reuse depends on the framework’s callable and specialization rules.',
      next: 'Move the stable callable and compilation wrapper outside the loop, then inspect compilation logs.', requires: ['compiler_output'], impact: ['compilation', 'latency'] });
    if (op.attributes.body_region && op.attributes.api !== 'triton.jit') {
      const body = ops.filter(o => o.region === op.attributes.body_region);
      if (b.ir.coverage.complete && body.some(o => o.kind === 'compute') && !body.some(o => ['host_sync', 'device_to_host', 'graph_boundary'].includes(o.op) || o.attributes.unknown_call)) b.add('graph.no_host_reads', [op, ...body.filter(o => o.kind === 'compute')], {
        title: 'Compiled step has no observed host reads', category: 'host_device_synchronization', scope: 'graph', assessment: 'good',
        explanation: 'The analyzed compiled body uses recognized operations and contains no explicit host-read or host-wait operation. This statement covers this visible region only.',
        next: 'Confirm the compiler captures this region as intended.', requires: ['compiler_output'], model: false });
      for (const control of body.filter(o => o.op === 'dynamic_shape' || o.op === 'conditional_execution' && o.attributes.tensor_value)) b.add(control.op === 'dynamic_shape' ? 'graph.shape_specialization' : 'graph.graph_break', [control, op], {
        title: control.op === 'dynamic_shape' ? 'Shape-dependent compiled branch' : 'Tensor value controls Python execution', category: control.op === 'dynamic_shape' ? 'recompilation_risk' : 'graph_breaks', scope: 'graph', confidence: 'medium',
        explanation: 'A Python condition in the compiled region depends on a tensor shape or value. Capture, guarding and specialization depend on framework configuration.',
        next: 'Inspect graph-break and recompilation logs with representative shapes.', requires: ['shapes', 'compiler_output'], impact: ['compilation', 'latency'] });
    }
  }
  for (const op of ops.filter(o => o.op === 'materialize')) {
    const producer = op.inputs.map(x => b.producer(op, x)).find(p => p?.kind === 'compute');
    if (producer) b.add('graph.materialization', [op, producer], { title: 'Intermediate is explicitly materialized', category: 'intermediate_materialization', scope: 'graph',
      explanation: 'A recognized compute result feeds an explicit materialization operation. Whether a copy or separate kernel remains depends on compilation and layout.',
      next: 'Check whether the consumer needs this materialized representation or can consume the producer result directly.', requires: ['compiler_output'], impact: ['memory', 'latency'] });
    if (b.loops(op).length && op.inputs.some(x => op.outputs.includes(x)) && /(?:\.cat|\.concat|\.concatenate)$/.test(String(op.attributes.api))) b.add('graph.growing_materialization', [op, b.loops(op)[0]!], {
      title: 'Loop concatenates back into its own input', category: 'memory_lifetime', scope: 'graph',
      explanation: 'A concatenation includes the previous value and reassigns its result to that value inside a loop. Each executed concatenation expresses a new combined tensor.',
      next: 'Check whether fixed storage, a list of chunks or an existing cache can avoid copying the accumulated prefix each iteration.', requires: ['shapes'], impact: ['memory', 'latency'] });
  }
  for (const end of ops.filter(o => o.kind === 'compute')) {
    const middle = end.inputs.map(x => b.producer(end, x)).find(o => o?.kind === 'compute');
    const start = middle?.inputs.map(x => b.producer(middle, x)).find(o => o?.kind === 'compute');
    if (start && middle && new Set([start.op, middle.op, end.op]).size >= 2) b.add('graph.fusion_chain', [end, middle, start], {
      title: 'Producer-consumer chain worth checking for fusion', category: 'fusion_opportunities', scope: 'graph',
      explanation: 'Three recognized compute operations form a direct producer-consumer chain. The compiler may already fuse some of these operations; source syntax does not establish a kernel count.',
      next: 'Inspect the generated kernels and intermediate allocations before considering a fused implementation.', requires: ['compiler_output'], impact: ['memory', 'latency'] });
  }
  for (const op of ops.filter(o => o.op === 'batch' && o.attributes.api === 'torch.utils.data.DataLoader')) {
    if (op.attributes.num_workers === 0) b.add('training.input_pipeline', [op], { title: 'Input loading uses the calling process', category: 'input_pipeline', scope: 'input_pipeline',
      explanation: 'The DataLoader explicitly sets num_workers=0. Loading and transforms execute in the calling process; whether they limit GPU submission requires measurement.',
      next: 'Measure time spent waiting for the next batch before changing worker or prefetch settings.', requires: ['runtime_measurement'], impact: ['throughput', 'utilization'] });
    if (Number(op.attributes.num_workers) > 0 && op.attributes.persistent_workers === true) b.add('training.persistent_workers', [op], { title: 'DataLoader workers persist across passes', category: 'input_pipeline', scope: 'input_pipeline', assessment: 'good',
      explanation: 'The loader explicitly enables worker processes and retains them across passes.', next: 'Check worker memory usage and epoch-boundary behavior for the actual dataset.', requires: ['runtime_measurement'], impact: ['throughput'], model: false });
  }
}

export function distributedLens(b: Findings) {
  const ops = b.ir.operations, collectives = ops.filter(o => o.kind === 'collective');
  for (const op of collectives.filter(o => !['reshard', 'replication'].includes(o.op))) {
    const loops = b.loops(op), dependent = ops.find(o => o.kind === 'compute' && o.inputs.some(x => b.producer(o, x)?.id === op.id));
    const bytes = typeof op.attributes.tensor_bytes === 'number' ? op.attributes.tensor_bytes : undefined;
    b.add(`distributed.${op.op}`, [op, ...loops.slice(0, 1), ...(dependent ? [dependent] : [])], { title: `${op.op.replaceAll('_', ' ')}${loops.length ? ' inside a loop' : ' on the execution path'}`, category: 'collective_placement', scope: 'distributed',
      explanation: `The source invokes ${op.op.replaceAll('_', ' ')}${loops.length ? ' inside a loop' : ''}${dependent ? ' before an operator that consumes its output' : ''}. Payload size and runtime communication cost are separate questions.`,
      next: 'Check collective placement and payload relative to useful compute at the expected shapes and rank count.', requires: bytes === undefined ? ['shapes'] : ['runtime_measurement'], impact: ['communication', 'memory'],
      ...(bytes === undefined ? {} : { level: 1 as const, quantities: [{ name: 'Input tensor bytes', value: bytes, unit: 'bytes', kind: 'static_estimate' as const, formula: 'product(shape) × dtype bytes; this is the input payload, not network traffic' }] }) });
    const group = ops.filter(o => o.attributes.api === 'torch.distributed.init_process_group' && Number(o.attributes.world_size) > 0);
    if (bytes !== undefined && group.length === 1 && !op.attributes.explicit_group) {
      const ranks = Number(group[0]!.attributes.world_size), f = b.candidates.at(-1)!.finding;
      if (Number.isInteger(ranks) && ranks > 1) {
        f.evidence_ids.push(group[0]!.evidence.id);
        if (op.op === 'all_gather') f.quantities!.push({ name: 'Gathered bytes per rank', value: bytes * ranks, unit: 'bytes', kind: 'static_estimate', formula: 'equal-size input tensor bytes × default-group rank count' });
        if (op.op === 'reduce_scatter' && bytes % ranks === 0) f.quantities!.push({ name: 'Reduced output bytes per rank', value: bytes / ranks, unit: 'bytes', kind: 'static_estimate', formula: 'input tensor bytes / default-group rank count' });
        if (op.op === 'all_reduce') {
          f.quantities!.push({ name: 'Theoretical ring bytes sent per rank', value: 2 * (ranks - 1) / ranks * bytes, unit: 'bytes', kind: 'static_estimate', formula: '2 × (ranks − 1) / ranks × tensor bytes; assumes ring reduce-scatter + all-gather, excludes protocol overhead' });
          f.assumptions.push('Ring algorithm is a theoretical traffic model; the backend may select a different algorithm.');
        }
      }
    }
    if (op.op === 'all_gather') {
      const copy = ops.find(o => o.op === 'materialize' && o.inputs.some(x => b.producer(o, x)?.id === op.id));
      if (copy) b.add('distributed.gather_materialization', [copy, op], { title: 'Gathered activation is materialized again', category: 'collective_placement', scope: 'distributed',
        explanation: 'The AllGather output directly feeds an explicit materialization operation.', next: 'Check whether the next operator can consume the gathered value directly or retain its sharded representation.', requires: ['shapes'], impact: ['communication', 'memory'] });
    }
    if (op.attributes.blocking === true && dependent) {
      const independent = ops.find(o => o.kind === 'compute' && o.region === op.region && o.evidence.location.startLine > op.evidence.location.startLine && o.evidence.location.startLine < dependent.evidence.location.startLine && o.inputs.length && !o.inputs.some(x => [...op.inputs, ...op.outputs].includes(x)));
      if (independent) b.add('distributed.overlap', [op, independent, dependent], { title: 'Independent compute follows a blocking collective', category: 'communication_compute_overlap', scope: 'distributed', confidence: 'medium',
        explanation: 'A blocking collective precedes recognized compute on different named inputs, followed by a consumer of the collective result. Aliasing and hidden dependencies are not established.',
        next: 'Check whether an asynchronous collective can overlap that compute, with an explicit wait before its first dependent consumer.', requires: ['runtime_measurement'], impact: ['communication', 'latency'], assumptions: ['Distinct input names may alias; verify ownership before reordering.'] });
    }
  }
  for (const last of collectives.filter(o => o.op === 'reshard')) {
    const middle = last.inputs.map(x => b.producer(last, x)).find(o => o?.op === 'reshard');
    const first = middle?.inputs.map(x => b.producer(middle, x)).find(o => o?.op === 'reshard');
    if (first && middle && first.attributes.sharding === last.attributes.sharding && first.attributes.sharding !== middle.attributes.sharding) b.add('distributed.reshard_roundtrip', [last, middle, first], {
      title: 'Sharding constraint returns to an earlier layout', category: 'sharding_transitions', scope: 'distributed', confidence: 'medium',
      explanation: 'A directly linked value passes through A → B → A sharding constraints. The compiler determines whether these become physical transfers.',
      next: 'Inspect compiled sharding and collective output to see whether the intermediate transition survives.', requires: ['compiler_output', 'shapes'], impact: ['communication', 'memory'] });
  }
}

export function servingLens(b: Findings) {
  for (const op of b.ir.operations.filter(o => o.op === 'configuration' && /^(vllm|sglang)\./.test(String(o.attributes.api)))) {
    const a = op.attributes, tp = Number(a.tensor_parallel_size ?? a.tp_size), batch = Number(a.max_num_seqs ?? a.max_running_requests);
    if (tp >= 8 && batch > 0 && batch <= 4) b.add('serving.tp_decode', [op], { title: 'High TP with a small request limit', category: 'tensor_parallelism', scope: 'serving', confidence: 'medium',
      explanation: `The engine configures ${tp} tensor-parallel ranks and at most ${batch} concurrent sequences. Small decode work per rank may make collective overhead more significant; capacity may require this TP setting.`,
      next: 'Compare useful per-rank decode work with collective payload at the expected shapes and hardware topology.', requires: ['shapes', 'hardware'], impact: ['latency', 'utilization', 'communication'],
      assumptions: ['The workload includes decode requests reaching this engine configuration.'] });
    const chunked = a.enable_chunked_prefill === true || Number(a.chunked_prefill_size) > 0;
    const disabled = a.enable_chunked_prefill === false || a.chunked_prefill_size === -1;
    if (chunked) b.add('serving.chunked_prefill', [op], { title: 'Chunked prefill is enabled', category: 'chunked_prefill', scope: 'serving', assessment: 'good',
      explanation: 'The engine explicitly enables splitting prefill into chunks. The setting provides a mechanism to interleave prefill and decode; it does not establish a latency improvement.',
      next: 'Check chunk size against the actual prompt-length and decode workload.', requires: ['runtime_measurement'], model: false });
    if (disabled && Number(a.max_model_len ?? a.context_length) >= 8192 && batch > 1 && !a.disaggregation_mode) b.add('serving.prefill_decode', [op], { title: 'Long prefill may share the decode scheduler', category: 'prefill_decode_interference', scope: 'serving', confidence: 'medium',
      explanation: 'The engine permits long sequences and multiple concurrent requests while chunked prefill is explicitly disabled. Interference depends on whether long prefills and active decodes arrive together.',
      next: 'Inspect mixed prefill/decode scheduling at representative prompt lengths and arrival patterns.', requires: ['runtime_measurement'], assumptions: ['The configured engine serves a mixed prefill/decode workload.'] });
    const graphsOff = a.enforce_eager === true || a.disable_cuda_graph === true;
    const graphsOn = a.enforce_eager === false || a.disable_cuda_graph === false;
    if (graphsOff || graphsOn) b.add('serving.cuda_graphs', [op], { title: graphsOff ? 'CUDA graph execution explicitly disabled' : 'CUDA graph execution permitted', category: 'cuda_graph_compatibility', scope: 'serving', assessment: graphsOff ? 'possible_issue' : 'good',
      explanation: graphsOff ? 'The engine configuration explicitly disables CUDA graph execution. Whether graph replay would help depends on the executed shapes and operations.' : 'The engine configuration permits CUDA graphs. Actual capture and reuse still depend on supported shapes and execution paths.',
      next: 'Inspect capture and replay logs for the expected decode shapes.', requires: ['compiler_output', 'runtime_measurement'], impact: ['latency', 'compilation'], model: graphsOff });
    if (a.enable_prefix_caching === true || a.disable_radix_cache === false) b.add('serving.prefix_cache', [op], { title: 'Prefix reuse is enabled', category: 'prefix_caching', scope: 'serving', assessment: 'good',
      explanation: 'The engine explicitly permits prefix-cache reuse. Cache hit rate depends on actual requests.', next: 'Inspect reuse and eviction behavior with representative request prefixes.', requires: ['runtime_measurement'], impact: ['memory', 'throughput'], model: false });
    if (typeof a.disaggregation_mode === 'string' && ['prefill', 'decode'].includes(a.disaggregation_mode)) b.add('serving.pd_transfer', [op], { title: 'Prefill/decode disaggregation boundary', category: 'pd_disaggregation', scope: 'serving',
      explanation: `The server is configured for the ${a.disaggregation_mode} side of a disaggregated execution policy. KV ownership and transfer are part of that deployment contract.`,
      next: 'Check KV transfer size and the configured transport between prefill and decode workers.', requires: ['shapes', 'hardware'], impact: ['communication', 'memory'] });
    const dims = ['num_hidden_layers', 'num_key_value_heads', 'head_dim', 'kv_cache_dtype_bytes'].map(name => Number(a[name]));
    const length = Number(a.max_model_len ?? a.context_length);
    if (dims.every(d => Number.isSafeInteger(d) && d > 0) && length > 0 && batch > 0) {
      const bytes = dims.reduce((x, y) => x * y, 2 * length * batch);
      if (Number.isSafeInteger(bytes)) b.add('serving.kv_capacity', [op], { title: 'Logical KV-cache size at configured limits', category: 'kv_cache_capacity', scope: 'serving', assessment: 'unknown',
        explanation: 'Explicit model dimensions and sequence limits give a logical K/V tensor estimate for conventional dense attention. Allocator overhead, model weights, KV replication and per-rank placement are not included.',
        next: 'Compare the logical KV requirement with usable per-rank memory after accounting for sharding, replication and model weights.', requires: ['hardware'], impact: ['memory'], level: 1, model: false,
        assumptions: ['Assumes dense K and V storage at every layer and sequence position; sliding-window, hybrid and latent-attention layouts can differ.'],
        quantities: [{ name: 'Logical KV bytes', value: bytes, unit: 'bytes', kind: 'static_estimate', formula: '2 × layers × KV heads × head dimension × dtype bytes × max sequence length × concurrent sequences' }] });
    }
  }
}
