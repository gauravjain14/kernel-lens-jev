import type { JevFinding, SystemsIR } from './types';

/** No model string or probability can create evidence, provenance or a measurement. */
export function gateFinding(finding: JevFinding, ir: SystemsIR): JevFinding | undefined {
  if (finding.confidence === 'low' || !finding.evidence_ids.length || !finding.evidence.explanation.trim() || !finding.next_check.trim()) return;
  const all = new Map([...ir.operations.map(op => op.evidence), ...ir.sourceAnchors ?? []].map(e => [e.id, e]));
  const evidence = finding.evidence_ids.map(id => all.get(id));
  if (evidence.some(e => !e)) return;
  const primary = evidence[0]!;
  if (!finding.evidence.source || finding.evidence.source !== primary.source || JSON.stringify(finding.evidence.location) !== JSON.stringify(primary.location)) return;
  if (finding.assessment === 'likely_issue' && (!primary.direct || finding.confidence !== 'high')) return;
  if (!finding.requires.length || finding.requires.includes('none') && finding.requires.length > 1) return;
  const runtime = evidence.some(e => e!.origin === 'runtime');
  if (finding.runtime_impact_measured && !runtime || finding.basis === 'runtime' && !runtime) return;
  if (finding.quantities?.some(q => !Number.isFinite(q.value) || q.value < 0 || q.kind === 'measured' && !runtime)) return;
  if (finding.section === 'assumptions' && !finding.impact.includes('correctness_assumption')) return;
  if (finding.evidence_level === 4 && !runtime || finding.evidence_level === 2 && !evidence.some(e => e!.origin === 'compiler')) return;
  return finding;
}
