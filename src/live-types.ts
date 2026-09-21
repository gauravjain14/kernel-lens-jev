export type Pack = 'cuda' | 'triton' | 'pytorch' | 'jax' | 'distributed' | 'serving' | 'training' | 'inference' | 'data' | 'general';
export type PackSetting = Pack | 'auto';
export type Signal = 'concern' | 'supported' | 'unknown' | 'not_applicable';
export interface SyntaxIssue { from: number; to: number; line: number; message: string }
export interface SourceUnit {
  name: string; kind: string; from: number; to: number; startLine: number; endLine: number;
  ready: boolean; syntax: SyntaxIssue[]; identity?: string;
}
export interface Reference { name: string; startLine: number; code: string; reason: string }
export interface CodeContext {
  file: string; language: string; unit: SourceUnit; code: string; enclosing: string;
  preamble: string; references: Reference[]; intent: string; hardware: string;
  truncated: boolean; characters: number; routingKey: string;
  recentEdit?: { before: string; after: string; startLine: number; endLine: number };
  hardwareProfile?: import('./core/systems/hardware').HardwareId;
  enrichment?: import('./core/systems/types').Enrichment;
  kernelHelpers?: import('./core/systems/types').KernelHelper[];
}
export interface Outcome { label: string; criteria: string; signal: Signal }
export interface Metric {
  id: string; pack: Pack; label: string; category: 'correctness' | 'performance' | 'numerics' | 'memory';
  question: string; outcomes: Record<string, Outcome>; reference: string;
}
export interface Choice { type: 'choice'; choice: string; probabilities: Record<string, number> }
export interface Route {
  technology: string; activity: string; packs: Pack[]; uncertain: boolean;
  probabilities: { technology: number; activity: number };
}
export interface Assessment {
  id: string; pack: Pack; label: string; category: Metric['category'];
  outcome: string; bucket: string; signal: Signal; probability: number;
  tentative: boolean; probabilities: Record<string, number>; reference: string;
  contextNeeded?: string;
}
export interface Usage { latencyMs: number; inputTokens?: number; outputTokens?: number; cost?: number }
export interface AssessmentReport extends Usage {
  route: Route; assessments: Assessment[]; changes: string[]; fingerprint: string;
  scope: { name: string; startLine: number; endLine: number; identity?: string };
  anchors?: Record<string, SourceAnchor>; detailStatus?: 'locating' | 'ready' | 'unavailable';
  visibleAfterMs?: number;
  findings?: import('./core/systems/types').JevFinding[];
  hardware?: import('./core/systems/hardware').HardwareReview;
  dimensions?: { id:string; label:string; assessment:import('./core/systems/types').Assessment; metricFamily:string; findingIds:string[];
    bucket?:string; explanation?:string; nextCheck?:string; requires?:import('./core/systems/types').Requirement[];
    confidence?:import('./core/systems/types').Confidence; probability?:number; applicable?:boolean; group?:string }[];
  coverage?: import('./core/systems/types').SystemsIR['coverage'];
}
export interface SourceAnchor { startLine: number; endLine: number; code: string; probability: number }
export interface Insight {
  id: string; relatedIds: string[]; title: string; consequence: string; nextCheck: string;
  kind: 'correctness' | 'performance' | 'tentative' | 'improved';
  change: 'new' | 'ongoing' | 'improved'; probability: number;
  startLine: number; endLine: number; code: string; anchored: boolean;
  finding?: import('./core/systems/types').JevFinding;
}
export interface AdviceItem {
  metricId: string; verdict: 'confirmed' | 'conditional' | 'dismissed';
  title: string; explanation: string; action: string; line: number | null; evidence: string;
}
export interface Advice extends Usage { model: string; summary: string; items: AdviceItem[] }
export interface LiveState {
  version: string; enabled: boolean; configured: boolean; consented: boolean;
  phase: 'disabled' | 'waiting' | 'classifying' | 'assessing' | 'ready' | 'paused' | 'limited' | 'error' | 'unsupported';
  hardwareProfile?: import('./core/systems/hardware').HardwareId;
  message: string; domain: PackSetting; intent: string; file?: string; unit?: SourceUnit;
  stale: boolean; report?: AssessmentReport; context?: { characters: number; truncated: boolean; references: { name: string; reason: string }[] };
  advisor: { mode: 'off' | 'onDemand' | 'auto'; model: string; phase: 'idle' | 'running' | 'ready' | 'error'; message: string; result?: Advice };
  references: { id: string; name: string; startLine: number; endLine: number }[];
  requests: number; advisorRequests: number; totalTokens: number; totalCost: number;
  insights?: Insight[]; improvements?: Insight[]; selectedFinding?: string;
}
