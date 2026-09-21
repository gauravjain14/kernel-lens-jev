__global__ void rms_norm_kernel(float *arr,int dim3, float* d_v_r){
    extern __shared__ float arr_cpy[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < dim3){
        arr_cpy[threadIdx.x] = arr[tid];
    }
    __syncthreads();

    if (threadIdx.x < dim3){
        arr_cpy[threadIdx.x] = arr_cpy[threadIdx.x]  * arr_cpy[threadIdx.x];
        // atomicAdd(&rms_acc, val);
    }
    __syncthreads();

    for(int i = dim3/2; i >= 32; i/= 2){
        if (threadIdx.x < i){
            arr_cpy[threadIdx.x] += arr_cpy[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) {
            sum += arr_cpy[i];
        }
        arr_cpy[0] = sum;
    }
    __syncthreads();

    if(threadIdx.x == 0){
        arr_cpy[0] = arr_cpy[0] / dim3;
        arr_cpy[0] = arr_cpy[0] + 1e-6;
        arr_cpy[0] = sqrt(arr_cpy[0]);
    }

    __syncthreads();

    d_v_r[tid] = arr[tid] / arr_cpy[0];
}
