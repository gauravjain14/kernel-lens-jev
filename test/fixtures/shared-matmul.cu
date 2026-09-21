__shared__ float As[16][16];
    __shared__ float Bs[16][16];

    const float threadId = threadIdx.y * blockDim.x + threadIdx.x;


    for (int i = 0; i < K; i += 16) {
        for (int j = 0; j < 16; j++) {
            As[threadIdx.y][j] = A[(blockIdx.y * 16 + threadIdx.y) * K + (i + j)];
            Bs[threadIdx.x][j] = B[(i + j) * N + (blockIdx.x * 16 + threadIdx.x)];
        }
        __syncthreads();

        for (int j = 0; j < 16; j++) {
            C[(blockIdx.y * 16 + threadIdx.y) * N + (blockIdx.x * 16 + threadIdx.x)] += As[threadIdx.y][j] * Bs[threadIdx.x][j];
        }
    }
