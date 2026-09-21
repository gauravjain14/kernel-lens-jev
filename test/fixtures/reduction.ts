export const shuffleReduction = `    if (threadIdx.x < 32) {
        float val = arr_cpy[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) {
            arr_cpy[0] = val;
        }
    }
    __syncthreads();`;

// The user's replacement, with no task intent or hardware hint supplied to Jev.
export const serialReduction = `    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            sum += arr_cpy[i];
        }
        arr_cpy[0] = sum;
    }
    __syncthreads();`;

export function reductionKernel(body: string, padding = ''): string {
  return `__global__ void reduce(const float* input, float* output) {
    __shared__ float arr_cpy[32];
    if (threadIdx.x < 32) arr_cpy[threadIdx.x] = input[threadIdx.x];
    __syncthreads();
${padding}${body}
    if (threadIdx.x == 0) output[blockIdx.x] = arr_cpy[0];
}
`;
}
