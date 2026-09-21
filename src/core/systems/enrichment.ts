import type { Enrichment, SystemsIR, SystemsOp } from './types';
import { Findings } from './lenses';

/** Imported facts are data only. Unknown fields and invalid units are rejected. */
export function parseEnrichment(raw: unknown): Enrichment {
  const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);
  if (!object(raw)) throw new Error('Evidence must be a JSON object.');
  const fields: Record<string, string[]> = {
    compiler: ['registers_per_thread', 'shared_memory_per_block', 'spill_loads', 'spill_stores', 'stack_frame', 'generated_kernel_count'],
    hardware: ['name', 'memory_capacity_bytes', 'sm_count', 'hbm_bandwidth_bytes_per_second', 'interconnect_bandwidth_bytes_per_second'],
    runtime: ['latency_ms', 'throughput_per_second', 'achieved_bandwidth_bytes_per_second', 'achieved_flops', 'stall_percent', 'cache_hit_percent'],
  };
  const dimensions = (v: unknown) => Array.isArray(v) && v.length > 0 && v.length <= 16 && v.every(n => typeof n === 'number' && Number.isSafeInteger(n) && n > 0);
  for (const [section, value] of Object.entries(raw)) {
    if (!['tensors', 'launch', ...Object.keys(fields)].includes(section) || !object(value)) throw new Error(`Unsupported evidence section: ${section.slice(0, 50)}.`);
    if (section === 'tensors') {
      if (Object.keys(value).length > 64) throw new Error('Evidence supports at most 64 tensor descriptions.');
      for (const [name, tensor] of Object.entries(value)) {
        if (!/^[A-Za-z_]\w*$/.test(name) || !object(tensor) || Object.keys(tensor).some(k => !['shape', 'dtype_bytes', 'device'].includes(k))
          || !dimensions(tensor.shape) || ![1, 2, 4, 8, 16].includes(Number(tensor.dtype_bytes)) || typeof tensor.dtype_bytes !== 'number'
          || tensor.device !== undefined && !['gpu', 'cpu'].includes(String(tensor.device))) throw new Error('Each tensor needs a numeric shape, dtype_bytes, and optional gpu/cpu device.');
      }
    } else if (section === 'launch') {
      if (Object.keys(value).some(k => !['block', 'grid'].includes(k)) || !dimensions(value.block) || (value.block as number[]).length > 3 || value.grid !== undefined && (!dimensions(value.grid) || (value.grid as number[]).length > 3)) throw new Error('Launch evidence needs one to three positive dimensions in block and optional grid.');
    } else for (const [name, val] of Object.entries(value)) {
      if (!fields[section]!.includes(name)) throw new Error(`Unsupported ${section} field: ${name.slice(0, 60)}.`);
      if (name === 'name') { if (typeof val !== 'string' || val.length > 120) throw new Error('Hardware name must be a short string.'); continue; }
      if (typeof val !== 'number' || !Number.isFinite(val) || val < 0 || name.endsWith('_percent') && val > 100 || section === 'compiler' && !Number.isSafeInteger(val)) throw new Error(`Invalid numeric value for ${name}.`);
    }
  }
  return structuredClone(raw) as Enrichment;
}

export function enrichmentLens(b: Findings) {
  const e = b.ir.enrichment, anchor = b.ir.operations.find(o => o.kind !== 'control') ?? b.ir.operations[0];
  if (!e || !anchor) return;
  const attach = (name: string, origin: 'compiler' | 'runtime' | 'config', values: object): SystemsOp => {
    const op = { ...anchor, id: `imported_${name}`, evidence: { ...anchor.evidence, id: `imported_${name}`, origin, direct: true, explanation: `Imported ${name}: ${JSON.stringify(values)}` } };
    b.ir.operations.push(op); return op;
  };
  const scope = b.ir.frameworks.includes('cuda') ? 'kernel' : 'graph';
  if (e.launch) {
    const op = attach('launch', 'config', e.launch), width = e.launch.block[0]!;
    for (const candidate of b.candidates.filter(c => c.finding.id.startsWith('cuda.power_of_two:'))) {
      const f = candidate.finding, powerOfTwo = Number.isInteger(Math.log2(width));
      f.evidence_ids.push(op.evidence.id); f.evidence_level = 1; f.requires = ['none'];
      f.evidence.explanation += ` The supplied launch has blockDim.x=${width}, which ${powerOfTwo ? 'is' : 'is not'} a power of two.`;
      if (powerOfTwo) { f.assessment = 'good'; f.title = 'Supplied launch satisfies the power-of-two width assumption'; }
    }
  }
  if (e.compiler && Object.keys(e.compiler).length) {
    const op = attach('compiler', 'compiler', e.compiler);
    const spills = (e.compiler.spill_loads ?? 0) + (e.compiler.spill_stores ?? 0);
    b.add('evidence.compiler', [op], { title: spills > 0 ? 'Compiler reports spills' : 'Compiler resource report supplied', scope, category: 'register_pressure',
      assessment: spills > 0 ? 'likely_issue' : 'unknown', explanation: spills > 0 ? 'The supplied compiler report contains spill loads or stores for this source revision. Their runtime cost remains unmeasured.' : 'Compiler resource quantities are available for this revision. These quantities alone do not establish achieved occupancy or runtime efficiency.',
      next: spills > 0 ? 'Inspect the spilled live ranges and measure whether their traffic affects this workload.' : 'Compare these resources with the launch shape and hardware limits.',
      requires: spills > 0 ? ['runtime_measurement'] : ['launch_config', 'hardware'], impact: ['memory', 'utilization'], level: 2, model: false,
      quantities: Object.entries(e.compiler).map(([name, value]) => ({ name, value: value!, unit: ['registers_per_thread', 'generated_kernel_count'].includes(name) ? 'count' : 'bytes', kind: 'compiler_reported' })) });
    b.candidates.at(-1)!.finding.basis = 'compiler';
  }
  if (e.hardware?.memory_capacity_bytes && b.candidates.some(c => c.finding.id.startsWith('serving.kv_capacity:'))) {
    const f = b.candidates.find(c => c.finding.id.startsWith('serving.kv_capacity:'))!.finding;
    const op = attach('hardware', 'config', e.hardware);
    f.evidence_ids.push(op.evidence.id); f.evidence_level = 3;
    f.quantities!.push({ name: 'Supplied device memory capacity', value: e.hardware.memory_capacity_bytes, unit: 'bytes', kind: 'static_estimate' });
    f.evidence.explanation += ' A device memory capacity is supplied; logical model-wide KV bytes must not be treated as per-rank allocation.';
    f.requires = ['shapes', 'runtime_measurement'];
  }
  if (e.runtime && Object.keys(e.runtime).length) {
    const op = attach('runtime', 'runtime', e.runtime);
    b.add('evidence.runtime', [op], { title: 'Supplied runtime measurements', scope, category: 'runtime_evidence', assessment: 'unknown',
      explanation: 'These measurements were supplied for this source revision. They describe that measured workload and do not establish the causal impact of individual static findings.',
      next: 'Confirm the measurement workload, warmup and configuration match the code being investigated.', requires: ['none'], impact: ['latency', 'throughput'], level: 4, model: false,
      quantities: Object.entries(e.runtime).map(([name, value]) => ({ name, value: value!, unit: name.endsWith('_percent') ? '%' : name.endsWith('_ms') ? 'ms' : name.includes('bandwidth') ? 'bytes/s' : name === 'achieved_flops' ? 'FLOP/s' : 'items/s', kind: 'measured' })) });
    Object.assign(b.candidates.at(-1)!.finding, { runtime_impact_measured: true, basis: 'runtime' });
  }
}
