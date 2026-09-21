// Offline validation harness. Never executed or shipped by the extension.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } } while (0)

// Both kernels launch exactly 32 threads per block and reduce 32 floats per block.
__global__ void serial_reduce(const float* input, float* output) {
    __shared__ float arr_cpy[32];
    arr_cpy[threadIdx.x] = input[blockIdx.x * 32 + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            sum += arr_cpy[i];
        }
        arr_cpy[0] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) output[blockIdx.x] = arr_cpy[0];
}

__global__ void warp_reduce(const float* input, float* output) {
    __shared__ float arr_cpy[32];
    arr_cpy[threadIdx.x] = input[blockIdx.x * 32 + threadIdx.x];
    __syncthreads();
    if (threadIdx.x < 32) {
        float val = arr_cpy[threadIdx.x];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) arr_cpy[0] = val;
    }
    __syncthreads();
    if (threadIdx.x == 0) output[blockIdx.x] = arr_cpy[0];
}

int main() {
    cudaDeviceProp gpu{}; CUDA(cudaGetDeviceProperties(&gpu, 0));
    int runtime = 0, driver = 0; CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
    std::printf("{\"gpu\":\"%s\",\"runtime\":%d,\"driver\":%d,\"method\":\"CUDA events; 21 alternating rounds of 100 launches; 100 warmup launches per variant; resident FP32 input; 32 threads/block\",\"results\":[", gpu.name, runtime, driver);
    bool first = true;
    for (int blocks : {1, 256, 32768}) {
        std::vector<float> input(blocks * 32), result(blocks), expected(blocks);
        for (int i = 0; i < blocks * 32; ++i) input[i] = float((i * 17) % 97 - 48) / 64.0f;
        for (int b = 0; b < blocks; ++b) for (int i = 0; i < 32; ++i) expected[b] += input[b * 32 + i];
        float *x, *y; CUDA(cudaMalloc(&x, input.size() * sizeof(float))); CUDA(cudaMalloc(&y, result.size() * sizeof(float)));
        CUDA(cudaMemcpy(x, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        auto launch = [&](int variant) { if (variant) warp_reduce<<<blocks, 32>>>(x, y); else serial_reduce<<<blocks, 32>>>(x, y); };
        float max_error = 0;
        for (int v = 0; v < 2; ++v) {
            launch(v); CUDA(cudaGetLastError()); CUDA(cudaMemcpy(result.data(), y, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int b = 0; b < blocks; ++b) max_error = std::max(max_error, std::abs(result[b] - expected[b]));
        }
        if (max_error > 1e-5f) { std::fprintf(stderr, "Output check failed\n"); return 1; }
        for (int i = 0; i < 100; ++i) { launch(0); launch(1); } CUDA(cudaDeviceSynchronize());
        cudaEvent_t begin, end; CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
        std::vector<float> timings[2];
        for (int round = 0; round < 21; ++round) {
            for (int slot = 0; slot < 2; ++slot) {
                int v = (round + slot) % 2;
                CUDA(cudaEventRecord(begin));
                for (int repeat = 0; repeat < 100; ++repeat) launch(v);
                CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
                float ms; CUDA(cudaEventElapsedTime(&ms, begin, end)); timings[v].push_back(ms * 10);
            }
        }
        CUDA(cudaGetLastError());
        for (auto& times : timings) std::sort(times.begin(), times.end());
        std::printf("%s{\"blocks\":%d,\"serial_median_us\":%.5f,\"warp_median_us\":%.5f,\"serial_over_warp\":%.4f,\"serial_p95_us\":%.5f,\"warp_p95_us\":%.5f,\"max_abs_error\":%.8f}", first ? "" : ",", blocks, timings[0][10], timings[1][10], timings[0][10] / timings[1][10], timings[0][19], timings[1][19], max_error); first = false;
        CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end)); CUDA(cudaFree(x)); CUDA(cudaFree(y));
    }
    std::puts("]}");
}
