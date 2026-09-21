import type { CodeContext, PackSetting } from '../../live-types';
import { SourceView } from './source';
import { adaptCuda } from './cuda';
import { adaptPython } from './python';
import { Findings, kernelLens, graphLens, distributedLens, servingLens } from './lenses';
import { enrichmentLens, parseEnrichment } from './enrichment';
import { gateFinding } from './gate';
import type { JevFinding } from './types';

export const systemsVersion = '1.2.0';
export function analyzeSystems(context: CodeContext, lens: PackSetting = 'auto') {
  const view = new SourceView(context);
  if (context.enrichment) view.ir.enrichment = parseEnrichment(context.enrichment);
  if (view.parsed.python) adaptPython(view); else adaptCuda(view);
  const builder = new Findings(view.ir);
  kernelLens(builder); graphLens(builder); distributedLens(builder); servingLens(builder); enrichmentLens(builder);
  // Findings outside the focused block are kept only if they establish a relationship
  // touching it. Unrelated functions in the context never become current findings.
  const inLens = (f: JevFinding) => {
    if (lens === 'auto' || lens === 'general') return true;
    if (['cuda', 'triton', 'pytorch', 'jax'].includes(lens)) return view.ir.frameworks.includes(lens as 'cuda' | 'triton' | 'pytorch' | 'jax');
    if (lens === 'distributed' || lens === 'serving') return f.scope === lens;
    if (lens === 'data') return f.scope === 'input_pipeline';
    return ['graph', 'operator', 'distributed', 'training', 'input_pipeline', ...(lens === 'inference' ? ['serving'] : [])].includes(f.scope);
  };
  const candidates = builder.candidates.filter(c => inLens(c.finding) && !view.ir.operations.find(o => o.evidence.id === c.finding.evidence_ids[0])?.attributes.unreachable && c.finding.evidence_ids.some(id => {
    const e = view.ir.operations.find(o => o.evidence.id === id)?.evidence;
    return e && e.location.startLine <= context.unit.endLine && e.location.endLine >= context.unit.startLine;
  }));
  const findings = candidates.map(c => gateFinding(c.finding, view.ir)).filter((f): f is JevFinding => !!f);
  return { ir: view.ir, candidates: candidates.filter(c => findings.includes(c.finding)), findings, lens };
}
