__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * 16 + ty;
    const int col = blockIdx.x * 16 + tx;
    float sum = 0.0f;
    for (int i = 0; i < K; i += 16) {
        As[ty][tx] = row < M && i + tx < K ? A[row * K + i + tx] : 0.0f;
        Bs[ty][tx] = i + ty < K && col < N ? B[(i + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int j = 0; j < 16; j++) {
            sum += As[ty][j] * Bs[j][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
