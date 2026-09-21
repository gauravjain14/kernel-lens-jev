// Inspection example. Launch with one block of 128 threads.
// Move __syncthreads() above the if to give it uniform participation.
__global__ void reverse_half(const float* input, float* output) {
    __shared__ float scratch[128];
    scratch[threadIdx.x] = input[threadIdx.x];
    if (threadIdx.x < 64) {
        __syncthreads();
        output[threadIdx.x] = scratch[127 - threadIdx.x];
    }
}
