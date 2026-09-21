// Phase 1 — cp.async double buffer, wait for every MMA pair.
// As pasted in conversation. Assumes BM/BN/BK/WARPS_PER_BLOCK are defined
// elsewhere (BM=128, BN=128, BK=32, WARPS_PER_BLOCK=4). In the epilogue as
// pasted, `taddr` and `out_row` were elided ("simple strides").

constexpr uint32_t A_LBO = 16;
constexpr uint32_t A_SBO = 512;
constexpr uint32_t A_SWZ = 4;
constexpr uint32_t A_K16_BYTES = 32;

constexpr uint32_t B_LBO = 1024;
constexpr uint32_t B_SBO = 2048;
constexpr uint32_t B_SWZ = 2;
constexpr uint32_t B_K16_BYTES = 4096;


__device__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// this is already defined by the Nvidia overlords
__host__ __device__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo,
    uint32_t sbo, uint32_t swz) {
    return (uint64_t)((saddr >> 4) & 0x3fff) | ((uint64_t)((lbo >> 4) & 0x3fff) << 16) | ((uint64_t)((sbo >> 4) & 0x3fff) << 32) | (1ULL << 46) | ((uint64_t)swz << 61);
}

__device__ void make_ab_descs(half A_tile[BM][BK], half B_tile[BK][BN],
                                uint64_t* a0, uint64_t *a1,
                                uint64_t* b0, uint64_t * b1) {
    // Wondering if it always starts at 0,0 inside the loop is right or not?
    // I think yes because the outside loop strides along K and we bring into
    // the shared memory
    uint32_t a_base = (uint32_t)__cvta_generic_to_shared(&A_tile[0][0]);
    uint32_t b_base = (uint32_t)__cvta_generic_to_shared(&B_tile[0][0]);
    // base, lbo, sbo, swizzle
    *a0 = make_smem_desc(a_base, A_LBO, A_SBO, A_SWZ);
    *a1 = make_smem_desc(a_base + A_K16_BYTES, A_LBO, A_SBO, A_SWZ);
    *b0 = make_smem_desc(b_base, B_LBO, B_SBO, B_SWZ);
    *b1 = make_smem_desc(b_base + B_K16_BYTES, B_LBO, B_SBO, B_SWZ);
}

// this is for the mbar wait. As MMA finishes, it resolves this flag
__device__ void mbar_wait(uint64_t* mbar, uint32_t parity) {
    uint32_t addr = smem_u32(mbar);
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n\t"
            "    .reg .pred p;\n\t"
            "    mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
            "    selp.b32 %0, 1, 0, p;\n\t"
            "    }"
            : "=r"(done)
            : "r"(addr), "r"(parity)
            : "memory");
    }
}

__device__ void tmem_alloc(uint32_t* smem_slot, uint32_t ncols) {
    uint32_t slot_addr = (uint32_t)__cvta_generic_to_shared((void*)smem_slot);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(slot_addr), "r"(ncols));
}

__device__ void tmem_dealloc(uint32_t d_tmem, uint32_t ncols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(d_tmem), "r"(ncols));
}

__device__ void mbar_init(uint64_t* mbar, uint32_t arrival_count) {
    uint32_t addr = smem_u32(mbar);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(addr), "r"(arrival_count) : "memory");
}

__device__ void tcgen05_commit(uint64_t* mbar) {
    uint32_t addr = smem_u32(mbar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(addr) : "memory");
}

__device__ void tmem_relinquish_alloc_permit() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}

__device__ void mma_f16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                    uint32_t idesc, uint32_t accumulate) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate)
    );
}

__global__ void gemm_tcgen5_v0(
    const half* A,
    const half* B,
    float* C,
    int M, int N, int K
) {
    using namespace nvcuda;

    __shared__ uint32_t tmem_slot;
    __shared__ __align__(1024) half A_shared[2][BM][BK];
    __shared__ __align__(1024) half B_shared[2][BK][BN];

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // number of elements each thread can load.
    constexpr int VEC_ELEMS = 16 / sizeof(half);
    // chunks per row is whatever we decide BN. BN is partition into VEC_ELEMS
    // BN should ideally also impact TMEM/MMA? let's see
    constexpr int chunks_per_row_B = BN / VEC_ELEMS;
    // number of rows read by all the threads in the threadblock.
    constexpr int num_rows_per_iteration_B = (WARPS_PER_BLOCK * 32) / chunks_per_row_B;
    // number of iterations to read all the BK rows for B
    constexpr int num_rd_iterations_per_block_B = BK / num_rows_per_iteration_B;

    // Repeat the same for A
    constexpr int chunks_per_row_A = BK / VEC_ELEMS;
    // number of rows read by all the threads in the threadblock.
    constexpr int num_rows_per_iteration_A = (WARPS_PER_BLOCK * 32) / chunks_per_row_A;
    // number of iterations to read all the BK rows for B
    constexpr int num_rd_iterations_per_block_A = BM / num_rows_per_iteration_A;
    // this thread's starting column (in halfs) within its A row:
    // chunk index scaled UP by the chunk width. Runtime value -> const,
    // not constexpr (threadIdx.x doesn't exist at compile time).
    const int thread_col_A = (threadIdx.x % chunks_per_row_A) * VEC_ELEMS;
    const int thread_col_B = (threadIdx.x % chunks_per_row_B) * VEC_ELEMS;

    // blocks mapped to the output
    int blocks_per_row = N / BN;
    int block_row = blockIdx.x / blocks_per_row;
    int block_col = blockIdx.x % blocks_per_row;

    int global_row_offset_A = block_row * BM + (threadIdx.x / chunks_per_row_A);
    int stage = 0;

    // now that we have the data, let's read
    for (int i = 0; i < num_rd_iterations_per_block_A; i++) {
        // this thread writes to row - shared_row_A
        int shared_next_row_a = i * num_rows_per_iteration_A + (threadIdx.x / chunks_per_row_A);
        int global_next_row_a = global_row_offset_A + i * num_rows_per_iteration_A;
        // now we the row to read from and the row to write to.
        // each read will get 16 bytes in.
        // The critical aspect about writing to the shared memory is to swizzle the data
        // such that for the core matrix (8 x 16B) read by the Tensor Core MMA, we have the
        // data arranged such as we have 128bytes filling the SMEM row and that means reads
        // from two rows of threads (i.e. 2 * chunks_per_row_A) will be stored side-by side.
        // The next row in the 8x16B chunk will be done from 64 bytes away in the same row
        // but the row after will come from the next row in the SMEM and you need to swizzle
        // there. Swizzle by 16B, i.e. 4 banks
        int slot_A = threadIdx.x % chunks_per_row_A;
        int swz_col_A = (slot_A ^ ((shared_next_row_a >> 1) & 3)) * VEC_ELEMS;
        if (global_next_row_a < M && thread_col_A + VEC_ELEMS <= K) {
            __pipeline_memcpy_async(
                &A_shared[stage][shared_next_row_a][swz_col_A],
                &A[global_next_row_a * K + thread_col_A],
                16
            );
        } else {
            *(float4*)&A_shared[stage][shared_next_row_a][swz_col_A] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    for (int i = 0; i < num_rd_iterations_per_block_B; i++) {
        int shared_next_row_b = threadIdx.x / chunks_per_row_B + i * num_rows_per_iteration_B;
        int slot_B = threadIdx.x % chunks_per_row_B;
        int swz_off_B = (shared_next_row_b / 8) * 1024 + (slot_B / 8) * 512
                      + (shared_next_row_b % 8) * 64 + ((slot_B % 8) ^ (shared_next_row_b % 8)) * 8;
        if (shared_next_row_b < K && block_col * BN + thread_col_B + VEC_ELEMS <= N) {
            __pipeline_memcpy_async(
                &B_shared[stage][swz_off_B / BN][swz_off_B % BN],
                &B[shared_next_row_b * N + block_col * BN + thread_col_B],
                16
            );
        } else {
            *(float4*)&B_shared[stage][swz_off_B / BN][swz_off_B % BN] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    __pipeline_commit();
    // ensure that these async copies actually finish - for a thread
    __pipeline_wait_prior(0);
    // pipeline wait prior only
    __syncthreads();

    // now bring the mma into the fold.
    __shared__ uint64_t mma_mbar;
    if (threadIdx.x < 32) {
        // number of rows is fixed - 128. Columns are controlled by BN
        tmem_alloc(&tmem_slot, BN);
        // relinquish the alloc permit to the next block
        tmem_relinquish_alloc_permit();
        if (threadIdx.x == 0) {
            mbar_init(&mma_mbar, 1);
        }
    }
    __syncthreads();

    // d_tmem in the TMEM
    uint32_t d_tmem = tmem_slot;
    uint32_t idesc = (1u << 4)              // D = fp32
        | (1u << 16)                        // B N-major
        | ((uint32_t)BN >> 3 << 17)         // N = 128
        | ((uint32_t)BM >> 4 << 24);        // M = 128
    uint32_t mma_parity = 0;

    // Now do this for the subsequent blocks
    for (int k = BK; k < K; k += BK) {
        int next_stage = 1 - stage;
        for (int i = 0; i < num_rd_iterations_per_block_A; i++) {
            int shared_next_row_a = threadIdx.x / chunks_per_row_A + i * num_rows_per_iteration_A;
            int global_next_row_a = global_row_offset_A + i * num_rows_per_iteration_A;
            int slot_A = threadIdx.x % chunks_per_row_A;
            int swz_col_A = (slot_A ^ ((shared_next_row_a >> 1) & 3)) * VEC_ELEMS;
            if (global_next_row_a < M && k + thread_col_A + VEC_ELEMS <= K) {
                __pipeline_memcpy_async(
                    &A_shared[next_stage][shared_next_row_a][swz_col_A],
                    &A[global_next_row_a * K + k + thread_col_A],
                    16
                );
            } else {
                *(float4*)&A_shared[next_stage][shared_next_row_a][swz_col_A] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }

        for (int i = 0; i < num_rd_iterations_per_block_B; i++) {
            int shared_next_row_b = threadIdx.x / chunks_per_row_B + i * num_rows_per_iteration_B;

            // same SW128 atom layout as the prologue (see comment there)
            int slot_B = threadIdx.x % chunks_per_row_B;
            int swz_off_B = (shared_next_row_b / 8) * 1024 + (slot_B / 8) * 512
                          + (shared_next_row_b % 8) * 64 + ((slot_B % 8) ^ (shared_next_row_b % 8)) * 8;

            // guard tests the B row actually addressed (k + shared_next_row_b)
            if (k + shared_next_row_b < K && block_col * BN + thread_col_B + VEC_ELEMS <= N) {
                __pipeline_memcpy_async(
                    &B_shared[next_stage][swz_off_B / BN][swz_off_B % BN],
                    &B[(k + shared_next_row_b) * N + block_col * BN + thread_col_B],
                    16
                );
            } else {
                *(float4*)&B_shared[next_stage][swz_off_B / BN][swz_off_B % BN] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __pipeline_commit();

        // now issue mma instructions.
        {
            uint64_t a_desc0, a_desc1, b_desc0, b_desc1;
            // make the a and b SMEM descriptors for the CURRENT stage's tile
            // (descs are out-params -> pass addresses)
            make_ab_descs(A_shared[stage], B_shared[stage],
                          &a_desc0, &a_desc1, &b_desc0, &b_desc1);
            if (threadIdx.x == 0) {
                // lowercase k (loop var): first mainloop iteration MMAs tile 0
                // -> overwrite garbage TMEM, all later ones accumulate.
                uint32_t acc = (k == BK) ? 0 : 1;
                mma_f16(d_tmem, a_desc0, b_desc0, idesc, acc);
                // always accumulate in place
                mma_f16(d_tmem, a_desc1, b_desc1, idesc, 1);
                // once you have the mmas, you say tcgen05 to commit mma_mbar
                tcgen05_commit(&mma_mbar);
            }
        }
        mbar_wait(&mma_mbar, mma_parity);
        mma_parity ^= 1;
        __pipeline_wait_prior(0);
        __syncthreads();
        stage = next_stage;
    }

    // tail completion.
    uint64_t a_desc0, a_desc1, b_desc0, b_desc1;
    make_ab_descs(A_shared[stage], B_shared[stage],
                  &a_desc0, &a_desc1, &b_desc0, &b_desc1);
    if (threadIdx.x == 0) {
        uint32_t acc0 = (K == BK) ? 0 : 1;
        mma_f16(d_tmem, a_desc0, b_desc0, idesc, acc0);
        mma_f16(d_tmem, a_desc1, b_desc1, idesc, 1);
        tcgen05_commit(&mma_mbar);
    }
    mbar_wait(&mma_mbar, mma_parity);
    mma_parity ^= 1;
    __syncthreads();

    // now create the epilogue.
    // to read out of TMEM to the shared memory/registers, there has to be a
    // fence that ensures that mmas have complete and the threads can start
    // reading the tmem out.
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Because BN = 128 and we have d_tmem as 128 x 128. And the lanes
    // of the TMEM are tied to warps such that each warp loads/stores
    // 32 lanes (only applicable during tcgen05.ld/st). 4 warps = 128 lanes
    // each instruction loads 32 x 8 32-bit values from the tensor memory.
    // - each instr get 32 x 32 bytes and we need 16 to get 32 x 512 bytes.
    for (int c = 0; c < BN; c += 8) {
        // so here we are reading 8 columns - 32 x 4bytes.
        uint32_t r[8];
        // sync.aligned ensures all threads in a warp are using this instruction.
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
              "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "r"(taddr + c));
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        // I don't care about this. This can be figured out. Simple strides.
        // column comes from the block_col * BN + c
        int out_col = block_col * BN + c;
        if (out_row < M && out_col + 8 <= N) {
            *(float4*)&C[out_row * N + out_col] = make_float4(
                __uint_as_float(r[0]), __uint_as_float(r[1]),
                __uint_as_float(r[2]), __uint_as_float(r[3]));
            *(float4*)&C[out_row * N + out_col + 4] = make_float4(
                __uint_as_float(r[4]), __uint_as_float(r[5]),
                __uint_as_float(r[6]), __uint_as_float(r[7]));
        }
    }
    __syncthreads();
    if (threadIdx.x < 32)
        tmem_dealloc(d_tmem, BN);
}
