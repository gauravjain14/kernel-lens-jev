import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { SourceParser, normalizeCuda } from '../src/core/syntax';
import { BlockReader, contextHash } from '../src/core/blocks';
import { analyzeSystems } from '../src/core/systems/analyze';
import { systemsPayload } from '../src/core/systems/evaluate';

const source=readFileSync('test/fixtures/phase1_cpasync_2stage.cu','utf8');
const read=(code:string,line=0)=>new BlockReader().read({source:code,file:'phase.cu',language:'cpp',cursorLine:line,assessmentScope:'function'});

test('downloaded cp.async kernel and all helpers are ready at every cursor line',()=>{
  const parser=new SourceParser(),p=parser.parse('phase.cu',source,false);
  assert.equal(p.nodes.filter(n=>n.type.isError).length,0);
  assert.equal(p.nodes.filter(n=>n.name==='FunctionDefinition').length,11);
  const reader=new BlockReader(),initial=read(source,101),hash=contextHash(initial);
  assert.equal(initial.unit.name,'gemm_tcgen5_v0');assert.equal(initial.unit.startLine,102);assert.equal(initial.unit.endLine,334);
  assert.match(initial.code,/L334: }$/);assert.match(initial.code,/tcgen05\.ld\.sync/);assert.equal(initial.truncated,false);
  assert.ok(initial.characters<=16000);
  for(let cursorLine=0;cursorLine<334;cursorLine++) {
    const c=reader.read({source,file:'phase.cu',language:'cpp',cursorLine,assessmentScope:'function'});
    assert.equal(c.unit.ready,true,`L${cursorLine+1}: ${c.unit.name}`);
    assert.equal(c.unit.syntax.length,0,`L${cursorLine+1}`);
    if(cursorLine>=101)assert.equal(contextHash(c),hash,`L${cursorLine+1}: function scope changed`);
  }
  assert.equal(read(source).unit.name,'gemm_tcgen5_v0','opening comments select the kernel');
});
test('inline assembly remains exact source evidence and an opaque execution boundary',()=>{
  const c=read(source,311),a=analyzeSystems(c),payload=systemsPayload(c,a);
  assert.ok(Object.keys(payload.questions).some(q=>q.startsWith('review.')));
  assert.equal(a.ir.coverage.complete,true);
  const asm=a.ir.operations.find(o=>o.attributes.call==='asm'&&o.attributes.unknown_call&&o.evidence.source.includes('tcgen05.ld.sync'))!;
  assert.ok(asm);assert.equal(asm.attributes.unknown_call,true);assert.equal(asm.evidence.location.startLine,311);
  assert.ok(source.includes(asm.evidence.source));assert.match(payload.state.current.code,/tcgen05\.ld\.sync/);
  const f=analyzeSystems(read('__global__ void f(const float* x,float* y){float a=x[0];asm volatile("st.global.f32 [%0], 0;"::"l"(x):"memory");float b=x[0];y[0]=a+b;}')).findings;
  assert.ok(!f.some(f=>f.id.startsWith('cuda.redundant_load')),'opaque asm can change memory');
});
test('asm normalization preserves offsets, qualifiers, literals and constraints',()=>{
  const cases=[
    'asm volatile("{\\n\\t.reg .pred p; }" : "=r"(out[0]) : "r"((x + 1)) : "memory");',
    '__asm__ __volatile__("mov.u32 %0, %1;" : "=r"(out) : "r"(x));',
    'asm goto ("bra %l0;" : : : : label); label: ;',
    'asm("nop;");',
    'const char* text = R"ptx({ \" ) __global__ )ptx"; char brace=\'}\'; int n=1\'024; asm volatile("nop;");',
    'const char* text = "asm volatile(\\\"{\\\");"; // asm volatile("}")\nasm volatile("nop;");',
  ];
  for(const body of cases) {
    const code=`__device__ void f(){\n${body}\n}`,normalized=normalizeCuda(code);
    assert.equal(normalized.length,code.length);
    assert.deepEqual([...normalized.matchAll(/\n/g)].map(m=>m.index),[...code.matchAll(/\n/g)].map(m=>m.index));
    const p=new SourceParser().parse('asm.cu',code,false);
    assert.equal(p.nodes.filter(n=>n.type.isError).length,0,body);
    assert.equal(p.nodes.filter(n=>n.name==='FunctionDefinition').length,1,body);
    assert.equal(read(code,1).unit.ready,true,body);
  }
});
test('unfinished strings, comments, asm constraints and braces still wait',()=>{
  for(const code of [
    '__global__ void f(){ asm volatile("nop;"',
    '__global__ void f(){ asm volatile("nop;" : "=r"(x); }',
    '__global__ void f(){ asm volatile("unterminated); }',
    '__global__ void f(){ const char* text=R"ptx(unterminated; }',
    '__global__ void f(){ /* unclosed }',
    '__global__ void f(){ asm volatile("nop;");',
  ])assert.equal(read(code).unit.ready,false,code);
});
test('incremental PTX edits invalidate context without creating false function boundaries',()=>{
  const reader=new BlockReader();
  for(const instruction of ['add.f32','mul.f32','fma.rn.f32']) {
    const code=`__global__ void f(float x){asm volatile("${instruction} %0, %0, %0;" : "+f"(x));}`;
    const c=reader.read({source:code,file:'edit.cu',language:'cpp',cursorLine:0,assessmentScope:'function'});
    assert.equal(c.unit.ready,true);assert.equal(c.unit.name,'f');assert.ok(c.code.includes(instruction));
  }
});
