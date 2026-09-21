(() => {
  'use strict';
  const api = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : window.kernelLensPreview;
  if (!api) return;
  const $ = id => document.getElementById(id);
  const send = (type, extra = {}) => api.postMessage({ type, ...extra });
  const text = (tag, cls, value) => { const el = document.createElement(tag); el.className = cls; el.textContent = value; return el; };
  const button = (cls, label, action) => { const el = text('button', cls, label); el.type = 'button'; el.addEventListener('click', action); return el; };
  let savedIntent = '', dirty = false, selectedFinding = '';
  const labels = { disabled: 'CONNECT JEV', waiting: 'WATCHING', classifying: 'CLASSIFYING', assessing: 'ASSESSING', ready: 'LIVE', paused: 'PAUSED', limited: 'PACING', error: 'CONNECTION', unsupported: 'STANDING BY' };
  const packLabels = { cuda: 'CUDA', triton: 'Triton', pytorch: 'PyTorch', jax: 'JAX', distributed: 'Distributed', serving: 'Serving', training: 'Training', inference: 'Inference', data: 'Input pipeline', general: 'General code' };
  const assessmentLabels = {good:'Good structure', possible_issue:'Possible issue', likely_issue:'Likely issue', unknown:'Unknown'};
  const words = value => value.replaceAll('_', ' ');
  const fromFinding = f => ({id:f.id, relatedIds:[f.id], title:f.title, code:f.evidence.source, consequence:f.evidence.explanation, nextCheck:f.next_check, kind:'performance', anchored:true, startLine:f.evidence.location?.startLine, endLine:f.evidence.location?.endLine, finding:f});
  const statusLabel = a => a.tentative ? 'Tentative' : a.signal === 'concern' ? 'Concern' : a.signal === 'supported' ? 'Supported' : a.signal === 'not_applicable' ? 'Not applicable' : 'Needs context';
  function assessmentRow(a, stale) {
    const row = text('details', 'assessment', '');
    row.dataset.id = a.id;
    row.dataset.signal = a.tentative ? 'unknown' : a.signal;
    const header = text('summary', '', '');
    const name = text('span', 'assessment-name', a.label);
    const bucket = text('span', 'assessment-bucket', `${a.tentative && a.signal === 'concern' ? 'Possible: ' : ''}${a.bucket}`);
    header.append(text('span', 'assessment-dot', ''), name, bucket);
    row.append(header);
    const distribution = text('div', 'distribution', '');
    distribution.append(text('p', 'subtle', `${statusLabel(a)} · ${Math.round(a.probability * 100)}% Jev probability. This is a model output, not a measured accuracy rate.`));
    if (a.contextNeeded) distribution.append(text('p', 'context-gap', a.contextNeeded));
    Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]).forEach(([outcome, p]) => {
      const line = text('div', 'probability-row', '');
      line.append(text('span', '', outcome.replaceAll('_', ' ')), text('span', 'mono', `${Math.round(p * 100)}%`)); distribution.append(line);
    });
    const actions = text('div', 'finding-bottom', '');
    const jump = button('text-button', 'View assessed block ↗', () => send('jump', { id: a.id })); jump.disabled = stale;
    actions.append(jump, button('text-button', 'Technical reference ↗', () => send('reference', { id: a.id })));
    row.append(distribution, actions); return row;
  }
  function findingCard(finding, state) {
    const card = text('article', 'insight-card', ''); card.dataset.kind = finding.kind; card.dataset.id = finding.id;
    const f = finding.finding;
    if(f) card.dataset.assessment = f.assessment;
    const label = f ? assessmentLabels[f.assessment].toUpperCase() : finding.kind === 'tentative' ? '◇ POSSIBLE ISSUE' : finding.kind === 'correctness' ? '⚠ LIKELY CORRECTNESS ISSUE' : '↗ PERFORMANCE RISK';
    const top = text('div', 'insight-top', ''); top.append(text('span', 'insight-kind', label));
    if (finding.change === 'new') top.append(text('span', 'new-tag', 'NEW'));
    card.append(top, text('h3', 'insight-title', finding.title));
    if (f) card.append(text('p', 'finding-scope', `${words(f.scope)} lens · ${words(f.category)}`));
    card.append(text('p', 'insight-consequence', finding.consequence));
    if (finding.code) { const evidence = text('details','source-disclosure',''); evidence.open=!f; evidence.append(text('summary','section-label',`Evidence · L${finding.startLine}`),text('code', 'source-evidence', finding.code.trim()));
      for(const e of f?.supporting_evidence||[]) evidence.append(text('p','subtle',`L${e.location.startLine}`),text('code','source-evidence',e.source));
      card.append(evidence); }
    if (f) {
      const confidence = text('div','finding-confidence',''); confidence.append(text('span','','Structural confidence'),text('strong','',words(f.confidence))); card.append(confidence);
      if (f.quantities?.length) for(const q of f.quantities) {
        const quantity = text('p','quantity',`${q.name}: ${q.value.toLocaleString()} ${q.unit} · ${words(q.kind)}`);
        if(q.formula) quantity.title=q.formula; card.append(quantity);
      }
    }
    const next = text('div', 'insight-next', ''); next.append(text('span', 'section-label', 'NEXT CHECK'), text('p', '', finding.nextCheck)); card.append(next);
    const actions = text('div', 'insight-actions', '');
    const jump = button('text-button', finding.anchored ? `L${finding.startLine} · Show source ↗` : `L${finding.startLine}–${finding.endLine} · Show block ↗`, () => send('jump', { id: finding.id })); jump.disabled = state.stale;
    actions.append(jump);
    if(!f) actions.append(button('text-button', 'Reference ↗', () => send('reference', { id: finding.id })));
    card.append(actions);
    if(f) {
      const need = f.requires.filter(r=>r!=='none');
      card.append(text('p','requires',need.length ? `Requires: ${need.map(words).join(', ')}.` : 'No additional context needed to identify this structure.'));
      if(f.assumptions.length) card.append(text('p','finding-assumptions',f.assumptions.join(' ')));
      card.append(text('p','runtime-impact',f.runtime_impact_measured ? 'Runtime evidence: supplied for this revision.' : 'Runtime impact: Unmeasured.'));
      const provenance=text('details','finding-provenance',''); provenance.append(text('summary','',`Level ${f.evidence_level} · ${f.basis} evidence`));
      provenance.append(text('p','subtle',f.model_probability === undefined ? 'Established by the bounded static analysis or supplied metadata.' : `${Math.round(f.model_probability*100)}% Jev probability for this classification. Structural confidence describes evidence strength, not measured impact.`));
      if(f.model_probabilities) for(const [choice,p] of Object.entries(f.model_probabilities)) provenance.append(text('p','probability-row',`${choice}: ${Math.round(p*100)}%`));
      card.append(provenance);
    }
    if (!finding.anchored) card.append(text('p', 'anchor-note', state.report?.detailStatus === 'locating' ? 'Locating the relevant source…' : 'This prediction applies to the block; a precise source location is not established.'));
    if (finding.relatedIds.length > 1) card.append(text('p', 'anchor-note', 'Related signals grouped into one finding.'));
    return card;
  }
  function adviceCard(item, stale) {
    const card = text('article', 'finding advice-card', ''); card.dataset.verdict = item.verdict;
    const top = text('div', 'finding-top', ''); top.append(text('span', 'category', `REVIEW · ${item.verdict}`));
    if (item.line !== null) { const jump = button('line-link', `L${item.line} ↗`, () => send('jump', { id: item.metricId })); jump.disabled = stale; top.append(jump); }
    card.append(top, text('h3', '', item.title));
    if (item.evidence) card.append(text('code', 'excerpt', item.evidence));
    card.append(text('p', '', item.explanation));
    if (item.action) { const next = text('div', 'next-step', ''); next.append(text('span', '', 'NEXT STEP'), text('p', '', item.action)); card.append(next); }
    return card;
  }
  function render(state) {
    if (!state || typeof state !== 'object') return;
    $('version').textContent = `v${state.version} · BETA`;
    $('status').dataset.phase = state.phase; $('status-text').textContent = labels[state.phase] || 'WATCHING'; $('message').textContent = state.message;
    const setup = !state.enabled || !state.configured || !state.consented || state.phase === 'disabled';
    $('onboarding').hidden = !setup; $('workspace').hidden = setup;
    $('enable').textContent = !state.enabled ? 'Resume live insights ↗' : 'Connect live insights ↗';
    $('domain').value = state.domain;
    $('hardware').value = state.hardwareProfile || state.report?.hardware?.profile?.id || 'b200';
    const reportHardware=state.report?.hardware?.profile;
    const hw=!state.hardwareProfile||reportHardware?.id===state.hardwareProfile?reportHardware:undefined;
    $('hardware-ceiling').textContent=hw ? `${hw.bandwidthTBs} TB/s HBM · ${hw.fp32Tflops} TFLOP/s FP32 · published ceilings` : 'Hardware assumption applies to predictions; runtime impact is unmeasured.';
    $('file-name').textContent = state.file?.split(/[\\/]/).pop() || 'Open a source file'; $('file-name').title = state.file || '';
    $('line-range').textContent = state.unit ? `L${state.unit.startLine}–${state.unit.endLine}` : '';
    const busy = ['classifying', 'assessing'].includes(state.phase);
    $('activity').hidden = !busy; $('activity-label').textContent = state.message; $('analyze').disabled = busy;
    $('workspace').classList.toggle('stale', !!state.stale);
    const report = state.report;
    $('profile').hidden = !report;
    $('technical').hidden = !report;
    if (report) {
      const insights = state.insights || [], improvements = state.stale ? [] : state.improvements || [];
      const canonical = report.findings || [];
      const grouped = new Map(); const order={likely_issue:3,possible_issue:2,unknown:1,good:0};
      canonical.filter(f=>f.section==='performance').forEach(f=>{if(!grouped.has(f.category)||order[f.assessment]>order[grouped.get(f.category).assessment]) grouped.set(f.category,f);});
      $('physics-estimates').replaceChildren(...(report.hardware?.estimates||[]).map(e=>{const card=text('details','disclosure-panel','');card.append(text('summary','',`L${e.line} · Ideal FP32 roofline: ${e.classification==='memory_ceiling'?'memory':'compute'} ceiling`),text('p','subtle',`${e.idealIntensity.toFixed(1)} FLOP/byte vs ${e.fp32Ridge.toFixed(1)} hardware ridge. ${e.explanation}`));return card;}));
      const kernelAxes=(report.dimensions||[]).filter(d=>d.group);
      const kernel=kernelAxes.length>0;
      $('lens-summary').classList.toggle('kernel-summary',kernel);
      if(kernel) {
        const expanded=new Set([...$('lens-summary').querySelectorAll('details[open]')].map(e=>e.dataset.id));
        const content=[];
        for(const group of ['Pipeline','Compute','Memory','Resources']) {
          const dimensions=kernelAxes.filter(d=>d.group===group&&d.applicable!==false);
          if(!dimensions.length)continue;
          content.push(text('p','kernel-group',group));
          for(const d of dimensions) {
            const row=text('details','kernel-axis','');row.dataset.id=d.id;row.dataset.assessment=d.assessment;row.open=expanded.has(d.id);
            const summary=text('summary','','');summary.append(text('span','axis-name',d.label),text('span','axis-bucket',d.bucket),text('span','axis-status',assessmentLabels[d.assessment]));row.append(summary);
            row.append(text('p','',d.explanation),text('p','axis-next',`Next check: ${d.nextCheck}`),text('p','subtle',`Requires: ${(d.requires||[]).map(words).join(', ')} · Runtime impact is unmeasured.`));
            const f=canonical.find(f=>f.id===d.findingIds[0]);
            if(f){row.append(text('p','subtle',`Structural confidence: ${words(f.confidence)} · L${f.evidence.location.startLine}`),text('code','source-evidence',f.evidence.source));
              for(const e of (f.supporting_evidence||[]).slice(-3))row.append(text('p','subtle',`Related evidence · L${e.location.startLine}`),text('code','source-evidence',e.source));
              const jump=button('text-button','Show source ↗',()=>send('jump',{id:f.id}));jump.disabled=state.stale;row.append(jump);}
            row.append(text('p','subtle',`Inspect in Nsight Compute: ${d.metricFamily}`));content.push(row);
          }
        }
        $('lens-summary').replaceChildren(...content);
      } else $('lens-summary').replaceChildren(...[...grouped.values()].map(f=>{const row=text('div','lens-row','');row.dataset.assessment=f.assessment;row.append(text('span','',words(f.category)),text('span','',assessmentLabels[f.assessment]));return row;}));
      $('unresolved-items').replaceChildren();
      $('unresolved-dimensions').hidden=kernel||!(report.dimensions||[]).some(d=>!d.findingIds.length);
      (report.dimensions||[]).filter(d=>!d.findingIds.length).forEach(d=>{const row=text('div','lens-row','');row.dataset.assessment='unknown';row.title=`${d.metricFamily} · no supported prediction`;row.append(text('span','',d.label),text('span','','Unknown'));$('unresolved-items').append(row);});
      $('lens-summary').hidden=!canonical.length&&!report.dimensions?.length;
      for(const [name,items] of [['positive',canonical.filter(f=>f.assessment==='good'&&f.section==='performance')],['assumption',canonical.filter(f=>f.section==='assumptions')],['unknown',canonical.filter(f=>f.assessment==='unknown'&&f.section==='performance')]]) {
        $(name+'-findings').hidden=!items.length; $(name+'-items').replaceChildren(...items.map(f=>findingCard(fromFinding(f),state)));
      }
      $('positive-label').textContent=`Observed good structure · ${canonical.filter(f=>f.assessment==='good'&&f.section==='performance').length}`;
      $('coverage-note').textContent=report.coverage?.limitations.join(' ') || '';
      $('revision-note').hidden = !state.stale;
      $('revision-note').textContent = state.phase === 'error' ? 'Previous edit · update failed' : 'Previous edit · updating';
      $('improvements').replaceChildren(...improvements.map(i => {
        const row = text('div', 'improvement', ''); row.append(text('span', 'improvement-label', '✓ UPDATED'), text('span', '', i.title)); return row;
      }));
      $('findings').replaceChildren(...insights.slice(0, 2).map(i => findingCard(i, state)));
      $('more-findings').hidden = insights.length <= 2;
      $('more-label').textContent = `${Math.max(0, insights.length - 2)} other findings`;
      $('other-findings').replaceChildren(...insights.slice(2).map(i => findingCard(i, state)));
      $('quiet-state').hidden = kernel || insights.length > 0 || improvements.length > 0;
      const performance = report.findings ? report.assessments.filter(a=>canonical.some(f=>f.id===a.id&&f.section==='performance')) : report.assessments;
      const supported = performance.find(a => a.signal === 'supported' && !a.tentative);
      const gap = performance.find(a => a.signal === 'unknown');
      $('quiet-state').replaceChildren(text('span', 'quiet-label', supported ? 'OBSERVED PRACTICE' : 'CURRENT BLOCK'),
        text('h3', '', supported ? supported.bucket : gap ? 'More evidence needed' : 'No supported structural finding'),
        text('p', '', report.findings ? canonical.find(f=>f.id===(supported||gap)?.id)?.evidence.explanation || 'No finding in the recognized operations. Unsupported calls and omitted code remain unassessed.' : supported ? 'Supported by Jev in the available context. This does not establish that the whole block is correct.' : gap?.contextNeeded || 'This assessment does not establish that the block is correct.'));
      $('route').textContent = `${report.route.packs.map(p => packLabels[p] || p).join(' + ')}${report.route.uncertain ? ' · provisional' : ''}`;
      $('unit-name').textContent = report.scope.name === 'module' ? 'Current block' : report.scope.name;
      const assessments = performance;
      $('concern-total').textContent = assessments.filter(a => a.signal === 'concern' && !a.tentative).length;
      $('supported-total').textContent = assessments.filter(a => a.signal === 'supported' && !a.tentative).length;
      $('unknown-total').textContent = assessments.filter(a => a.signal === 'unknown' || a.tentative).length;
      const priority = a => a.tentative ? 2 : ({ concern: 0, supported: 1, unknown: 2, not_applicable: 3 }[a.signal] ?? 4);
      const expanded = new Set([...document.querySelectorAll('.assessment[open]')].map(el => el.dataset.id));
      $('assessments').replaceChildren(...(report.findings ? [] : [...assessments]).sort((a, b) => priority(a) - priority(b)).map(a => {
        const row = assessmentRow(a, state.stale); row.open = expanded.has(a.id); return row;
      }));
      $('changes').replaceChildren(...report.changes.map(change => text('p', '', `↳ ${change}`)));
      $('report-meta').textContent = `${report.visibleAfterMs !== undefined ? `${report.visibleAfterMs} ms edit-to-result · ` : ''}${report.latencyMs} ms assessment${report.inputTokens !== undefined ? ` · ${report.inputTokens.toLocaleString()} input tokens` : ''}${state.stale ? ' · previous revision' : ''}`;
      if (state.selectedFinding && !state.stale) {
        const target = [...document.querySelectorAll('.insight-card')].find(el => el.dataset.id === state.selectedFinding);
        if (target) { target.classList.add('selected'); if (target.closest('#other-findings')) $('more-findings').open = true; if (selectedFinding !== state.selectedFinding) target.scrollIntoView({ block: 'nearest' }); }
      }
      selectedFinding = state.selectedFinding || '';
    }
    const advisor = state.advisor || {};
    $('advisor').hidden = !report;
    $('advisor-heading').textContent = advisor.model?.includes('astra') ? 'Astra review' : 'Model review';
    $('advisor-message').textContent = advisor.message || '';
    $('review').disabled = state.stale || advisor.mode === 'off' || advisor.phase === 'running' || state.phase !== 'ready';
    $('review').textContent = advisor.phase === 'running' ? 'Reviewing…' : 'Review this block ↗';
    $('handoff').disabled = state.stale || state.phase !== 'ready';
    $('review-summary').hidden = !advisor.result; $('review-summary').textContent = advisor.result?.summary || '';
    $('advice-items').replaceChildren(...(advisor.result?.items || []).map(item => adviceCard(item, state.stale)));
    if (!dirty) { $('intent').value = state.intent || ''; savedIntent = state.intent || ''; }
    $('context-budget').textContent = state.context ? `${state.context.characters.toLocaleString()} characters · ${state.context.truncated ? 'context truncated; some answers may need more evidence' : 'focused block + relevant context'}` : 'Context is gathered automatically. No task comment required.';
    $('auto-context').replaceChildren(...(state.context?.references || []).map(ref => text('p', 'auto-ref', `${ref.reason} · ${ref.name}`)));
    $('references').replaceChildren(...(state.references || []).map(ref => {
      const row = text('div', 'reference-row', ''); const remove = button('remove-button', '×', () => send('removeReference', { id: ref.id }));
      remove.setAttribute('aria-label', `Remove reference ${ref.name}`);
      row.append(text('span', '', ref.name.split(/[\\/]/).pop()), text('span', 'range', `L${ref.startLine}–${ref.endLine}`), remove); return row;
    }));
    $('session-count').textContent = `${state.requests} Jev · ${state.advisorRequests} reviews`;
    $('session-count').title = `${state.totalTokens.toLocaleString()} reported tokens · $${state.totalCost.toFixed(6)} reported cost. Canceled requests may still be billed; some providers omit cost.`;
  }
  for (const id of ['enable', 'analyze', 'pause', 'key', 'attach', 'settings', 'review', 'handoff', 'importEvidence']) $(id).addEventListener('click', () => send(id));
  $('hardware').addEventListener('change', event => send('hardware', {value:event.target.value}));
  $('domain').addEventListener('change', event => send('domain', { value: event.target.value }));
  $('intent').addEventListener('input', () => { dirty = $('intent').value !== savedIntent; $('intent-status').textContent = dirty ? 'Unsaved context' : 'Relevant setup and helpers are included automatically.'; });
  $('intent-form').addEventListener('submit', event => { event.preventDefault(); dirty = false; savedIntent = $('intent').value; send('intent', { value: savedIntent }); $('intent-status').textContent = 'Context saved'; });
  window.addEventListener('message', event => { if (event.data?.type === 'state') render(event.data.state); });
  if (window.kernelLensPreview) $('preview-label').hidden = false;
  send('ready');
})();
