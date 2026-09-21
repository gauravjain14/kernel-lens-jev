/** Product findings are independent of Jev's wire format and model probabilities. */
export type Assessment = 'good' | 'possible_issue' | 'likely_issue' | 'unknown';
export type Confidence = 'high' | 'medium' | 'low';
export type Scope = 'kernel' | 'operator' | 'graph' | 'distributed' | 'training' | 'serving' | 'input_pipeline';
export type Impact = 'latency' | 'throughput' | 'memory' | 'communication' | 'utilization' | 'compilation' | 'correctness_assumption';
export type Requirement = 'none' | 'shapes' | 'launch_config' | 'hardware' | 'compiler_output' | 'runtime_measurement';
export type EvidenceLevel = 0 | 1 | 2 | 3 | 4;
export interface SourceLocation { file: string; startLine: number; endLine: number; startColumn?: number; endColumn?: number }
export interface JevFinding {
  scope: Scope; category: string; assessment: Assessment; confidence: Confidence;
  evidence: { source?: string; location?: SourceLocation; explanation: string };
  impact: Impact[]; next_check: string; requires: Requirement[]; runtime_impact_measured: boolean;
  // Provenance and stable UI identity supplement, rather than replace, the universal schema.
  id: string; title: string; evidence_ids: string[]; evidence_level: EvidenceLevel;
  basis: 'static' | 'inferred' | 'compiler' | 'runtime'; assumptions: string[];
  model_probability?: number; model_choice?: string; model_probabilities?: Record<string, number>; model_concern_probability?: number;
  section: 'performance' | 'assumptions';
  supporting_evidence?: { source:string; location:SourceLocation }[];
  quantities?: { name: string; value: number; unit: string; kind: 'static_estimate' | 'compiler_reported' | 'measured'; formula?: string }[];
}
export type Framework = 'cuda' | 'triton' | 'pytorch' | 'jax' | 'vllm' | 'sglang';
export interface Evidence {
  id: string; source: string; location: SourceLocation; explanation: string;
  origin: 'source' | 'config' | 'compiler' | 'runtime'; direct: boolean;
}
export type Attribute = string | number | boolean | number[] | string[];
interface OpBase {
  id: string; evidence: Evidence; inputs: string[]; outputs: string[];
  controls: string[]; region: string; attributes: Record<string, Attribute>;
}
export interface ComputeOp extends OpBase { kind: 'compute'; op: 'gemm' | 'attention' | 'convolution' | 'reduction' | 'elementwise' | 'normalization' | 'routing' }
export interface MemoryOp extends OpBase { kind: 'memory'; op: 'load' | 'store' | 'allocate' | 'materialize' | 'copy' | 'host_to_device' | 'device_to_host' | 'kv_read' | 'kv_write' }
export interface CollectiveOp extends OpBase { kind: 'collective'; op: 'all_reduce' | 'reduce_scatter' | 'all_gather' | 'all_to_all' | 'broadcast' | 'send' | 'recv' | 'reshard' | 'replication' }
export interface SynchronizationOp extends OpBase { kind: 'synchronization'; op: 'block_barrier' | 'warp_barrier' | 'stream_wait' | 'event_wait' | 'host_sync' }
export interface ControlOp extends OpBase { kind: 'control'; op: 'branch' | 'loop' | 'dynamic_shape' | 'conditional_execution' }
export interface CompilationOp extends OpBase { kind: 'compilation'; op: 'graph_boundary' | 'jit_boundary' | 'graph_break' | 'specialization' }
export interface SchedulingOp extends OpBase { kind: 'scheduling'; op: 'batch' | 'queue' | 'prefill' | 'decode' | 'route' | 'pipeline_stage' | 'configuration' }
export type SystemsOp = ComputeOp | MemoryOp | CollectiveOp | SynchronizationOp | ControlOp | CompilationOp | SchedulingOp;
export interface TensorFact {
  name: string; region: string; shape?: number[]; dtypeBytes?: number; dtype?: string; device: 'gpu' | 'cpu' | 'unknown';
  evidenceId: string; sharding?: string; memory?: 'global' | 'shared' | 'local';
}
export interface SystemsIR {
  version: 1; frameworks: Framework[]; operations: SystemsOp[]; tensors: TensorFact[];
  sourceAnchors?: (Evidence & { role: string })[];
  coverage: { complete: boolean; limitations: string[] }; enrichment?: Enrichment;
}
/** Small, source-backed contracts for reachable same-file CUDA helpers. */
export interface KernelHelper {
  name: string;
  effects: { kind: SystemsOp['kind']; op: SystemsOp['op']; attributes: Record<string, Attribute>; evidence: Evidence }[];
  partial: boolean;
}
export interface Enrichment {
  tensors?: Record<string, { shape: number[]; dtype_bytes: number; device?: 'gpu' | 'cpu' }>;
  launch?: { block: number[]; grid?: number[] };
  compiler?: { registers_per_thread?: number; shared_memory_per_block?: number; spill_loads?: number; spill_stores?: number; stack_frame?: number; generated_kernel_count?: number };
  hardware?: { name?: string; memory_capacity_bytes?: number; sm_count?: number; hbm_bandwidth_bytes_per_second?: number; interconnect_bandwidth_bytes_per_second?: number };
  runtime?: { latency_ms?: number; throughput_per_second?: number; achieved_bandwidth_bytes_per_second?: number; achieved_flops?: number; stall_percent?: number; cache_hit_percent?: number };
}
/** A candidate carries the actual operation IDs and the strongest permitted conclusion. */
export interface FindingCandidate {
  finding: JevFinding; question: string; modelMayAssess: boolean;
  alternatives?: { good?: string; issue?: string; nextGood?: string };
}
