(() => {
  const state = __SYSTEMS_SAMPLE__;
  const emit = () => window.postMessage({type:'state',state:structuredClone(state)},'*');
  window.kernelLensSampleState=state;window.kernelLensMessages=[];
  window.kernelLensPreview={postMessage(message){
    window.kernelLensMessages.push(message);
    if(message.type==='pause'){state.enabled=false;state.phase='paused';state.message='Paused. No code is being sent.';}
    if(message.type==='enable'){state.enabled=true;state.phase='ready';state.message='Insights updated. Keep writing.';}
    if(message.type==='intent')state.intent=message.value;
    if(message.type==='hardware')state.hardwareProfile=message.value;
    if(message.type==='domain')state.domain=message.value;
    if(message.type==='removeReference')state.references=state.references.filter(r=>r.id!==message.id);
    if(message.type==='attach')state.references.push({id:'extra',name:'caller.py',startLine:1,endLine:10});
    if(message.type==='analyze'){state.phase='assessing';emit();setTimeout(()=>{state.phase='ready';state.requests++;emit();},300);return;}
    if(message.type==='review'){const f=state.report.findings[0];state.advisor.phase='ready';state.advisorRequests++;state.advisor.result={summary:'Inspect the observed reduction structure with the actual launch shape.',items:[{metricId:f.id,verdict:'conditional',title:f.title,explanation:f.evidence.explanation,action:f.next_check,line:f.evidence.location.startLine,evidence:f.evidence.source}]};}
    emit();
  }};
})();
