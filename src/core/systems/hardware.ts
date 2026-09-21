import type { CodeContext } from '../../live-types';
import type { SystemsIR } from './types';

export type HardwareId = 'b200' | 'h100' | 'a100' | 'unspecified';
export interface HardwareProfile {
  id: HardwareId; name: string; architecture: string; memoryGB: number;
  bandwidthTBs: number; fp32Tflops: number; denseBf16Tflops: number; denseTf32Tflops: number;
  sources: string[];
}
// Published per-GPU SXM specifications, decimal units. HGX B200 compute
// numbers are divided by eight and sparse tensor peaks by a further two.
export const hardwareProfiles: Record<Exclude<HardwareId, 'unspecified'>, HardwareProfile> = {
  b200: { id:'b200', name:'B200 · HGX/DGX', architecture:'Blackwell', memoryGB:180, bandwidthTBs:8,
    fp32Tflops:75, denseBf16Tflops:2250, denseTf32Tflops:1125,
    sources:['https://www.nvidia.com/en-us/data-center/hgx/', 'https://www.nvidia.com/en-us/data-center/dgx-b200/'] },
  h100: { id:'h100', name:'H100 SXM 80GB', architecture:'Hopper', memoryGB:80, bandwidthTBs:3.35,
    fp32Tflops:67, denseBf16Tflops:989.5, denseTf32Tflops:494.5,
    sources:['https://www.nvidia.com/en-us/data-center/h100/'] },
  a100: { id:'a100', name:'A100 SXM 80GB', architecture:'Ampere', memoryGB:80, bandwidthTBs:2.039,
    fp32Tflops:19.5, denseBf16Tflops:312, denseTf32Tflops:156,
    sources:['https://www.nvidia.com/en-us/data-center/a100/'] },
};
export const hardwareId = (value: unknown): HardwareId => typeof value === 'string' && ['b200','h100','a100','unspecified'].includes(value) ? value as HardwareId : 'b200';
export interface HardwareReview {
  profile?: HardwareProfile; assumed: true; note: string;
  estimates: { line:number; source:string; flops:number; minimumTensorBytes:number; idealIntensity:number; fp32Ridge:number;
    classification:'memory_ceiling'|'compute_ceiling'; explanation:string }[];
}
export function hardwareReview(context: CodeContext, ir: SystemsIR): HardwareReview {
  const id=hardwareId(context.hardwareProfile), profile=id==='unspecified'?undefined:hardwareProfiles[id];
  const result:HardwareReview={profile,assumed:true,estimates:[],note:profile
    ? 'Assumed per-GPU hardware; published ceilings, not achieved performance. Runtime impact is unmeasured.'
    : 'No hardware ceiling assumed. Supply hardware context for hardware-dependent predictions.'};
  // Imported hardware may describe a different device. Never silently combine
  // its bandwidth with this profile's compute peak.
  if(context.enrichment?.hardware) {
    result.note='Imported hardware context is present. Built-in ceilings are not combined with it; runtime impact remains unmeasured.';
    result.profile=undefined;return result;
  }
  if(!profile)return result;
  for(const op of ir.operations) {
    const {gemm_flops:flops,gemm_minimum_bytes:bytes,gemm_fp32:fp32}=op.attributes;
    if(op.attributes.unreachable||!fp32||typeof flops!=='number'||typeof bytes!=='number'||flops<=0||bytes<=0)continue;
    const intensity=flops/bytes, ridge=profile.fp32Tflops/profile.bandwidthTBs;
    result.estimates.push({line:op.evidence.location.startLine,source:op.evidence.source,flops,minimumTensorBytes:bytes,idealIntensity:intensity,fp32Ridge:ridge,
      classification:intensity<ridge?'memory_ceiling':'compute_ceiling',
      explanation:'2MNK FLOPs / ((MK + KN + MN) × 4 bytes), assuming each FP32 input is read once and the output written once. The FP32 roofline assumes no TF32 conversion. Caches, extra traffic, library algorithm and overlap can change the limiting resource; this is an ideal operator estimate, not a measured bottleneck.'});
  }
  return result;
}
