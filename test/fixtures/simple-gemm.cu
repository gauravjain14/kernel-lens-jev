#include <cuda.h>
#include <cuda_runtime.h>


__global__ simpleGEMM(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    const float threadId = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = 0; i < K; i += 16) {
        As[threadIdx.y][threadIdx.x] = A[threadId + i * M];
        Bs[threadIdx.y][threadIdx.x] = B[threadId + i * N];
        __syncthreads();

        for (int j = 0; j < 16; j++) {
            C[threadId] += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        }
        __syncthreads();
    }
}
