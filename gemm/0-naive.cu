#include <cuda_runtime.h>
#include <iostream>

using namespace std;

const int M = 1024;
const int N = 1024;
const int K = 1024;

/* random init the data */
void random_init_data(float *data, size_t size) {
    for (size_t i=0;i<size;i++) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

/* check data with cpu compute */
bool check_with_cpu(float *A, float *B, float *C,
                    int m, int n, int k) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            float sum = 0.f;
            for (int h=0;h<k;h++) {
                sum += A[i*k+h] * B[h*n+j];
            }
            if (isnan(C[i * n + j])) {
                printf("C[%d][%d] is nan\n", i, j);
                return false;
            }
            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

/* kernel function */
/* naive implementation */
/* time: 1.037888ms GFLOPS: 2069.089867*/
__global__ void Naive_GEMM(float *A, float *B, float *C, int m, int n, int k) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty < m && tx < n) {
        float sum = 0.f;
        for (int i=0;i<k;i++) {
            sum += A[ty * k + i] * B[i * n + tx];
        }
        C[ty * n + tx] = sum;
    }
}

/* using shared memory */
/* a thread compute C's one element, the same as above */
/* time: 0.813152ms GFLOPS: 2640.937497*/
template <int shared_tile>
__global__ void Shared_GEMM(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float A_tile[shared_tile][shared_tile], B_tile[shared_tile][shared_tile];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;
    for (int i=0;i<k;i+=shared_tile) {
        A_tile[threadIdx.y][threadIdx.x] = A[row * k + i + threadIdx.x];
        B_tile[threadIdx.y][threadIdx.x] = B[(i+threadIdx.y) * n + col];
        __syncthreads();
        for (int j=0;j<shared_tile;j++) {
            sum += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * n + col] = sum;
}


/* using shared memory */
/* a thread compute thread_tile * thread_tile */
/*
    block_tile: a block contain block_tile thread in one dim.
    thread_tile: a thread compute thread_tile * thread_tile.
*/
/* time: 0.289536ms GFLOPS: 7416.983219 */
template <int block_tile, int thread_tile>
__global__ void Shared_GEMM_stride(float *A, float *B, float *C, int m, int n, int k) {
    // one block need to compute shared_size * shared_size.
    // one block corresponds to the shared memory.
    constexpr int shared_size = block_tile * thread_tile;
    // sum for one thread
    float sum[thread_tile][thread_tile] = {0.f};

    for (int i=0;i<k;i+=shared_size) {
        // global memory to shared memory
        __shared__ float A_tile[shared_size][shared_size], B_tile[shared_size][shared_size];
        for (int j=0;j<thread_tile;j++) {
            for (int h=0;h<thread_tile;h++) {
                A_tile[threadIdx.y*thread_tile + j][threadIdx.x*thread_tile + h] 
                    = A[
                        blockIdx.y * k * shared_size + i +
                        (threadIdx.y * thread_tile + j) * k +
                        threadIdx.x * thread_tile + h 
                    ];
                B_tile[threadIdx.y*thread_tile + j][threadIdx.x*thread_tile + h] 
                    = B[
                        i * n + blockIdx.x * shared_size +
                        (threadIdx.y * thread_tile + j) * n +
                        threadIdx.x * thread_tile + h
                    ]; 
            }
        }
        __syncthreads();
        // compute
        for (int j=0;j<thread_tile;j++) {
            for (int h=0;h<thread_tile;h++) {
                for (int kk=0;kk<shared_size;kk++) {
                    sum[j][h] += 
                        A_tile[threadIdx.y*thread_tile + j][kk] * B_tile[kk][threadIdx.x*thread_tile + h];
                }
            }
        }
        __syncthreads();
    }

    // to global
    for (int i=0;i<thread_tile;i++) {
        for (int j=0;j<thread_tile;j++) {
            C[
                (blockIdx.y * shared_size + threadIdx.y * thread_tile + i) * n + 
                blockIdx.x * shared_size + threadIdx.x * thread_tile + j
            ] = sum[i][j];
        }
    }
}

/* using register */
/* time: 0.290240ms GFLOPS: 7398.992984 */
template<int block_tile, int register_tile>
__global__ void Register_GEMM(float *A, float *B, float *C, int m, int n, int k) {
    float A_register[register_tile][register_tile], B_register[register_tile][register_tile];
    float C_register[register_tile][register_tile] = {0.f};
    constexpr int shared_size = block_tile * register_tile;
    __shared__ float A_shared[shared_size][shared_size], B_shared[shared_size][shared_size];

    // compute along k
    for (int i=0;i<k;i+=shared_size) {
        // read data from global to shared
        for (int j=0;j<register_tile;j++) {
            for (int h=0;h<register_tile;h++) {
                A_shared[threadIdx.y*register_tile+j][threadIdx.x*register_tile+h] = A[
                    blockIdx.y*k*shared_size + i +
                    (threadIdx.y*register_tile+j)*k +
                    threadIdx.x*register_tile+h
                ];
                B_shared[threadIdx.y*register_tile+j][threadIdx.x*register_tile+h] = B[
                    i*n + blockIdx.x * shared_size +
                    (threadIdx.y*register_tile+j)*n +
                    threadIdx.x*register_tile+h
                ];
            }
        }
        __syncthreads();
        // read data from shared to register
        for (int ii=0;ii<shared_size;ii+=register_tile) {
            for (int jj=0;jj<register_tile;jj++) {
                for (int hh=0;hh<register_tile;hh++) {
                    A_register[jj][hh] = A_shared[threadIdx.y*register_tile+jj][ii+hh];
                    B_register[hh][jj] = B_shared[ii+hh][threadIdx.x*register_tile+jj];
                }
            }
            // compute
            for (int jj=0;jj<register_tile;jj++) {
                for (int hh=0;hh<register_tile;hh++) {
                    for (int kk=0;kk<register_tile;kk++) {
                        C_register[jj][hh] += A_register[jj][kk] * B_register[kk][hh];
                    }
                }
            }
        }
        __syncthreads();
    }
    // write data to global
    for (int i=0;i<register_tile;i++) {
        for (int j=0;j<register_tile;j++) {
            C[
                (blockIdx.y * shared_size + threadIdx.y * register_tile + i) * n +
                blockIdx.x * shared_size + threadIdx.x * register_tile + j
            ] = C_register[i][j];
        }
    }
}

/* deal with bank conflict */
template<int block_tile, int register_tile>
__global__ void Bank_GEMM(float *A, float *B, float *C, int m, int n, int k) {
    float A_register[register_tile][register_tile], B_register[register_tile][register_tile];
    float C_register[register_tile][register_tile] = {0.f};
    constexpr int shared_size = block_tile * register_tile;
    constexpr int offset = 0;
    __shared__ float A_shared[shared_size][shared_size+offset], B_shared[shared_size][shared_size+offset];

    // compute along k
    for (int i=0;i<k;i+=shared_size) {
        // read data from global to shared
        for (int j=0;j<register_tile;j++) {
            for (int h=0;h<register_tile;h++) {
                int old_index = (threadIdx.y*register_tile+j) * shared_size + threadIdx.x*register_tile+h;
                int new_shared_index_y = old_index / (shared_size + offset);
                int new_shared_index_x = old_index % (shared_size + offset);
                A_shared[new_shared_index_y][new_shared_index_x] = A[
                    blockIdx.y*k*shared_size + i +
                    (threadIdx.y*register_tile+j)*k +
                    threadIdx.x*register_tile+h
                ];
                B_shared[new_shared_index_y][new_shared_index_x] = B[
                    i*n + blockIdx.x * shared_size +
                    (threadIdx.y*register_tile+j)*n +
                    threadIdx.x*register_tile+h
                ];
            }
        }
        __syncthreads();
        // read data from shared to register
        for (int ii=0;ii<shared_size;ii+=register_tile) {
            for (int jj=0;jj<register_tile;jj++) {
                for (int hh=0;hh<register_tile;hh++) {
                    // 16-way bank conflict
                    int old_index_A = (threadIdx.y*register_tile+jj) * shared_size + ii + hh;
                    int new_index_A_y = old_index_A / (shared_size + offset);
                    int new_index_A_x = old_index_A % (shared_size + offset);
                    A_register[jj][hh] = A_shared[new_index_A_y][new_index_A_x];
                    // 4-way bank conflict
                    int old_index_B = (ii+hh) * shared_size + threadIdx.x*register_tile+jj;
                    int new_index_B_y = old_index_B / (shared_size + offset);
                    int new_index_B_x = old_index_B % (shared_size + offset);
                    B_register[hh][jj] = B_shared[new_index_B_y][new_index_B_x];
                }
            }
            // compute
            for (int jj=0;jj<register_tile;jj++) {
                for (int hh=0;hh<register_tile;hh++) {
                    for (int kk=0;kk<register_tile;kk++) {
                        C_register[jj][hh] += A_register[jj][kk] * B_register[kk][hh];
                    }
                }
            }
        }
        __syncthreads();
    }
    // write data to global
    for (int i=0;i<register_tile;i++) {
        for (int j=0;j<register_tile;j++) {
            C[
                (blockIdx.y * shared_size + threadIdx.y * register_tile + i) * n +
                blockIdx.x * shared_size + threadIdx.x * register_tile + j
            ] = C_register[i][j];
        }
    }
}


int main() {
    /* host data */
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, M * K * sizeof(float));
    cudaMallocHost(&h_B, N * K * sizeof(float));
    cudaMallocHost(&h_C, M * N * sizeof(float));
    random_init_data(h_A, M * K);
    random_init_data(h_B, N * K);

    /* device data */
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    /* copy data from host to device */
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    /* record the time */
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    /* call the kernel */
    /* naive implementation */
    // dim3 block(32, 32);
    // dim3 grid((N-1)/block.x + 1, (M-1)/block.y + 1);
    // Naive_GEMM<<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* using shared memory implementation */
    // constexpr int shared_tile = 32;
    // dim3 block(shared_tile, shared_tile);
    // dim3 grid((N-1)/block.x + 1, (M-1)/block.y + 1);
    // Shared_GEMM<shared_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* using shared memory implementation */
    /* a thread compute thread_tile * thread_tile */
    // constexpr int thread_tile = 4;
    // constexpr int block_tile = 8;
    // dim3 block(block_tile, block_tile);
    /* one block for block_tile * thread_tile */
    // dim3 grid((N-1)/(block_tile * thread_tile)+1, (M-1)/(block_tile * thread_tile)+1);
    // Shared_GEMM_stride<block_tile, thread_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* using register implementation */
    // constexpr int register_tile = 4;
    // constexpr int shared_tile = 8;
    // dim3 block(shared_tile, shared_tile);
    // dim3 grid((N-1)/(shared_tile * register_tile)+1, (M-1)/(shared_tile * register_tile)+1);
    // Register_GEMM<shared_tile, register_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* deal with the bank conflict */
    constexpr int register_tile = 4;
    constexpr int shared_tile = 8;
    dim3 block(shared_tile, shared_tile);
    dim3 grid((N-1)/(shared_tile * register_tile)+1, (M-1)/(shared_tile * register_tile)+1);
    Bank_GEMM<shared_tile, register_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("Executing time: %f ms\n", time_ms);

    /* compute the throughput */
    long workload = long(M) * N * K * 2;
    double gflops = (double(workload) / 1e9) / (double(time_ms) / 1e3);
    printf("GFLOPS: %f\n", gflops);

    /*copy result from device to host */
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    /* check the result data */
    bool result_check = check_with_cpu(h_A, h_B, h_C, M, N, K);
    printf("Check result: %s\n", result_check ? "OK" : "Failed");

    /* release the source */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}