import type { Assessment, AssessmentReport, CodeContext, Insight } from '../live-types';
import { guidance } from './guidance';

const groups: Record<string, string> = { 'train-loader': 'loader', 'data-iterator': 'loader', 'tensor-transfer': 'transfer', 'data-transfer': 'transfer' };
const groupKey = (report: AssessmentReport, id: string) => {
  // Module eval and functional dropout can be distinct bugs. Merge only when
  // Jev independently points both hypotheses to the same current operation.
  if (['infer-mode', 'infer-attention-dropout'].includes(id)) {
    const mode = report.anchors?.['infer-mode'], dropout = report.anchors?.['infer-attention-dropout'];
    if (mode && dropout && mode.startLine === dropout.startLine && mode.code === dropout.code) return 'same-eval-operation';
  }
  return groups[id] ?? id;
};
const severity = (a: Assessment) => a.tentative ? 0 : a.category === 'correctness' ? 4 : a.category === 'numerics' ? 3 : a.category === 'memory' ? 2 : 1;
export function kernelSummary(report:AssessmentReport):string|undefined {
  const axes=report.dimensions?.filter(d=>d.group&&d.group!=='Assumptions');if(!axes?.length)return;
  const concerns=axes.filter(d=>d.assessment==='possible_issue'||d.assessment==='likely_issue');
  const priority=['review.cuda-tensor-cores','review.cuda-overlap','review.cuda-pipeline-depth'];
  const structure=priority.flatMap(id=>axes.filter(d=>d.id===id&&d.assessment==='good').map(d=>d.bucket!)).slice(0,2);
  return [...structure,concerns.length?`${concerns.length} performance checks`:axes.some(d=>d.assessment==='unknown'&&d.applicable!==false)?'Open performance questions':'Source classifications ready'].join(' · ');
}
export function performanceAssessments(report: AssessmentReport): Assessment[] {
  return report.findings ? report.assessments.filter(a => report.findings!.some(f => f.id === a.id && f.section === 'performance')) : report.assessments;
}
export function selectedConcerns(report: AssessmentReport): Assessment[] {
  const selected: Assessment[] = [], seen = new Set<string>();
  for (const a of report.assessments.filter(a => a.signal === 'concern').sort((a, b) => severity(b) - severity(a) || b.probability - a.probability || a.id.localeCompare(b.id))) {
    const group = groupKey(report, a.id);
    if (!seen.has(group)) { selected.push(a); seen.add(group); }
  }
  return selected;
}
export function insightPresentation(report: AssessmentReport, previous?: AssessmentReport, context?: CodeContext): { insights: Insight[]; improvements: Insight[] } {
  if (report.findings) {
    const make = (f: NonNullable<AssessmentReport['findings']>[number], improved = false): Insight => ({
      id: f.id, relatedIds: [f.id], title: f.title, consequence: f.evidence.explanation, nextCheck: f.next_check,
      kind: improved ? 'improved' : 'performance', change: improved ? 'improved' : previous?.findings?.some(p => p.id === f.id && p.assessment === f.assessment) ? 'ongoing' : 'new',
      probability: f.model_probability ?? 0, startLine: f.evidence.location!.startLine, endLine: f.evidence.location!.endLine,
      code: f.evidence.source!, anchored: true, finding: f,
    });
    const risks = report.findings.filter(f => f.section === 'performance' && ['possible_issue', 'likely_issue'].includes(f.assessment));
    risks.sort((a, b) => Number(b.assessment === 'likely_issue') - Number(a.assessment === 'likely_issue') || Number(b.confidence === 'high') - Number(a.confidence === 'high'));
    const sameScope = previous && previous.scope.identity === report.scope.identity && previous.scope.name === report.scope.name;
    const improved = sameScope ? report.findings.filter(f => f.assessment === 'good' && f.section === 'performance' && !risks.some(r => r.category === f.category)
      && previous.findings?.some(p => p.category === f.category && ['possible_issue', 'likely_issue'].includes(p.assessment))) : [];
    return { insights: risks.map(f => make(f)), improvements: improved.map(f => make(f, true)) };
  }
  const make = (a: Assessment, improved = false): Insight => {
    const prior = previous?.assessments.find(p => p.id === a.id), anchor = report.anchors?.[a.id];
    const guide = guidance[a.id]!;
    return { id: a.id, relatedIds: report.assessments.filter(p => p.signal === 'concern' && groupKey(report, p.id) === groupKey(report, a.id)).map(p => p.id),
      title: a.bucket, consequence: improved ? `Jev now finds: ${a.bucket.toLowerCase()}.` : guide[0], nextCheck: improved ? '' : guide[1],
      kind: improved ? 'improved' : a.tentative ? 'tentative' : ['correctness', 'numerics'].includes(a.category) ? 'correctness' : 'performance',
      change: improved ? 'improved' : prior?.signal === 'concern' ? 'ongoing' : 'new', probability: a.probability,
      startLine: anchor?.startLine ?? report.scope.startLine, endLine: anchor?.endLine ?? report.scope.endLine, code: anchor?.code ?? '', anchored: !!anchor };
  };
  const insights = selectedConcerns(report).map(a => make(a));
  const near = (i: Insight) => !!context?.recentEdit && i.anchored && i.startLine <= context.recentEdit.endLine && i.endLine >= context.recentEdit.startLine;
  insights.sort((a, b) => Number(a.kind === 'tentative') - Number(b.kind === 'tentative')
    || Number(b.kind === 'correctness') - Number(a.kind === 'correctness')
    || Number(b.change === 'new') - Number(a.change === 'new') || Number(near(b)) - Number(near(a)));
  const improvements = report.assessments.filter(a => a.signal === 'supported' && !a.tentative
    && previous?.assessments.some(p => p.id === a.id && p.signal === 'concern' && !p.tentative)).map(a => make(a, true));
  return { insights, improvements };
}
export const insightLabel = (i: Insight) => i.kind === 'improved' ? 'Improved' : i.finding ? ({ good: 'Good structure', possible_issue: 'Possible issue', likely_issue: 'Likely issue', unknown: 'Unknown' }[i.finding.assessment]) : i.kind === 'tentative' ? 'Possible issue' : i.kind === 'correctness' ? 'Likely correctness issue' : 'Performance risk';
export const insightIcon = (i: Insight) => i.kind === 'improved' ? '✓' : i.kind === 'tentative' ? '◇' : i.kind === 'correctness' ? '⚠' : '↗';

/** Present model outcomes without promoting tentative predictions to warnings. */
export function inlineInsight(report: AssessmentReport): string {
  const first = selectedConcerns(report)[0];
  return first ? `◉ Code insights · ${first.tentative ? 'Possible: ' : ''}${first.bucket}` : '◉ Code insights';
}
