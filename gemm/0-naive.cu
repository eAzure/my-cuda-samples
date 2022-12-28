#include <cuda_runtime.h>
#include <iostream>

using namespace std;

const int M = 2048;
const int N = 2048;
const int K = 2048;

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

/* using register 1*/
/* time: 0.290240ms GFLOPS: 7398.992984 */
template<int block_tile, int register_tile>
__global__ void Register_GEMM_1(float *A, float *B, float *C, int m, int n, int k) {
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

/* using register 2*/
/*
    block_tile: one block corresponds
    register_tile: one thread corresponds
*/
/* time: 0.310752ms GFLOPS: 6894.982545 */
/* https://blog.csdn.net/qianqing13579/article/details/127359866 */
template<int block_tile, int block_tile_k, int register_tile>
__global__ void Register_GEMM_2(float *A, float *B, float *C, int m, int n, int k) {
    // register
    float A_register[register_tile] = {0.f};
    float B_register[register_tile] = {0.f};
    float C_register[register_tile][register_tile] = {0.f};

    for (int i=0;i<k/block_tile_k;i++) {
        __shared__ float A_shared[block_tile][block_tile_k];
        __shared__ float B_shared[block_tile_k][block_tile];

        // read data from global to shared
        int numberOfElementsPerThread = (block_tile * block_tile_k) / (blockDim.x * blockDim.y);
        // record one thread compute initial index in block
        int startIndex = numberOfElementsPerThread * (threadIdx.y * blockDim.x + threadIdx.x);
        for (int threadIndex=0; threadIndex<numberOfElementsPerThread; threadIndex++) {
            int logicalIndex = startIndex + threadIndex;
            // 每一个thread负责读入shared_mem中的一部分
            A_shared[logicalIndex / block_tile_k][logicalIndex % block_tile_k] = 
                A[blockIdx.y * block_tile * k + i * block_tile_k + 
                  logicalIndex / block_tile_k * k + logicalIndex % block_tile_k];
            B_shared[logicalIndex / block_tile][logicalIndex % block_tile] = 
                B[i*block_tile_k*n + blockIdx.x*block_tile + 
                  logicalIndex / block_tile * n + logicalIndex % block_tile];
        }
        __syncthreads();
        // load data from shared into register and compute
        for (int j=0;j<block_tile_k;j++) {
            for (int g=0;g<register_tile;g++) {
                A_register[g] = A_shared[threadIdx.y * register_tile + g][j];
            }
            for (int h=0;h<register_tile;h++) {
                B_register[h] = B_shared[j][threadIdx.x*register_tile + h];
            }
            // compute
            for (int a=0;a<register_tile;a++) {
                for (int b=0;b<register_tile;b++) {
                    C_register[a][b] += A_register[a] * B_register[b];
                }
            }
        }
        __syncthreads();
    }
    // copy from C_register to C
    for (int i=0;i<register_tile;i++) {
        for (int j=0;j<register_tile;j++) {            
            C[
                (blockIdx.y * block_tile + threadIdx.y * register_tile + i) * n +
                blockIdx.x * block_tile + threadIdx.x * register_tile + j
            ] = C_register[i][j];
        }
    }
}

/* deal with bank conflict */
// template<int block_tile, int register_tile>
// __global__ void Bank_GEMM(float *A, float *B, float *C, int m, int n, int k) {
    
// }

/* data prefetch */
/* compute without waiting for memory */
/* small improvement, when shape grows big, the improvement is more obvious */
/*refer: https://blog.csdn.net/qianqing13579/article/details/127359866*/
template<int block_tile, int block_tile_k, int register_tile>
__global__ void Prefetch_GEMM(float *A, float *B, float *C, int m, int n, int k) {
    // register
    float A_register[register_tile] = {0.f};
    float B_register[register_tile] = {0.f};
    float C_register[register_tile][register_tile] = {0.f};

    // shared mem
    __shared__ float A_shared[2][block_tile][block_tile_k];
    __shared__ float B_shared[2][block_tile_k][block_tile];

    // prefetch
    int numberOfElementsPerThread = (block_tile * block_tile_k) / (blockDim.x * blockDim.y);
    int startIndex = numberOfElementsPerThread * (threadIdx.y * blockDim.x + threadIdx.x);
    for (int threadIndex=0;threadIndex<numberOfElementsPerThread;threadIndex++) {
        int logicalIndex = startIndex + threadIndex;
        A_shared[0][logicalIndex / block_tile_k][logicalIndex % block_tile_k] = 
            A[blockIdx.y * block_tile * k + 0 * block_tile_k + 
                logicalIndex / block_tile_k * k + logicalIndex % block_tile_k];
        B_shared[0][logicalIndex / block_tile][logicalIndex % block_tile] = 
            B[0*block_tile_k*n + blockIdx.x*block_tile + 
                logicalIndex / block_tile * n + logicalIndex % block_tile];
    }
    __syncthreads();

    int indexOfRead, indexOfWrite;
    bool indexFlag = false;
    for (int i=1;i<k/block_tile_k;i++) {
        // 交替变换indexOfRead | indexOfWrite
        indexOfRead = (int)indexFlag; // 本次读取indexOfRead(0|1) 到register中
        indexOfWrite = 1 - indexOfRead; // 预取indexOfWrite(0|1)下一次的数据到shared_mem中

        for (int j=0;j<block_tile_k;j++) {
            for (int g=0;g<register_tile;g++) {
                A_register[g] = A_shared[indexOfRead][threadIdx.y * register_tile + g][j];
            }
            for (int h=0;h<register_tile;h++) {
                B_register[h] = B_shared[indexOfRead][j][threadIdx.x*register_tile + h];
            }
            // compute
            for (int a=0;a<register_tile;a++) {
                for (int b=0;b<register_tile;b++) {
                    C_register[a][b] += A_register[a] * B_register[b];
                }
            }
        }

        // prefetch data
        int numberOfElementsPerThread = (block_tile * block_tile_k) / (blockDim.x * blockDim.y);
        int startIndex = numberOfElementsPerThread * (threadIdx.y * blockDim.x + threadIdx.x);
        for (int threadIndex=0;threadIndex<numberOfElementsPerThread;threadIndex++) {
            int logicalIndex = startIndex + threadIndex;
            A_shared[indexOfWrite][logicalIndex / block_tile_k][logicalIndex % block_tile_k] = 
                A[blockIdx.y * block_tile * k + i * block_tile_k + 
                    logicalIndex / block_tile_k * k + logicalIndex % block_tile_k];
            B_shared[indexOfWrite][logicalIndex / block_tile][logicalIndex % block_tile] = 
                B[i*block_tile_k*n + blockIdx.x*block_tile + 
                    logicalIndex / block_tile * n + logicalIndex % block_tile];
        }
        __syncthreads();
        indexFlag = !indexFlag;
    }
    {
        // 计算最后一个BK
        for (int j=0;j<block_tile_k;j++) {
            for (int g=0;g<register_tile;g++) {
                A_register[g] = A_shared[indexOfWrite][threadIdx.y * register_tile + g][j];
            }
            for (int h=0;h<register_tile;h++) {
                B_register[h] = B_shared[indexOfWrite][j][threadIdx.x*register_tile + h];
            }
            // compute
            for (int a=0;a<register_tile;a++) {
                for (int b=0;b<register_tile;b++) {
                    C_register[a][b] += A_register[a] * B_register[b];
                }
            }
        }
    }

    // store data to C
    for (int i=0;i<register_tile;i++) {
        for (int j=0;j<register_tile;j++) {            
            C[
                (blockIdx.y * block_tile + threadIdx.y * register_tile + i) * n +
                blockIdx.x * block_tile + threadIdx.x * register_tile + j
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

    /* using register implementation 1 */
    // constexpr int register_tile = 4;
    // constexpr int shared_tile = 8;
    // dim3 block(shared_tile, shared_tile);
    // dim3 grid((N-1)/(shared_tile * register_tile)+1, (M-1)/(shared_tile * register_tile)+1);
    // Register_GEMM_1<shared_tile, register_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* using register implementation 2 */
    // constexpr int block_tile = 128;
    // constexpr int register_tile = 8;
    // constexpr int block_tile_k = 8;
    // dim3 block(block_tile/register_tile, block_tile/register_tile);
    // /* Notice! */
    // dim3 grid(N/block_tile, M/block_tile);
    // Register_GEMM_2<block_tile, block_tile_k, register_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* using data prefetch */
    constexpr int block_tile = 128;
    constexpr int register_tile = 8;
    constexpr int block_tile_k = 8;
    dim3 block(block_tile/register_tile, block_tile/register_tile);
    /* Notice! */
    dim3 grid(N/block_tile, M/block_tile);
    Prefetch_GEMM<block_tile, block_tile_k, register_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    /* deal with the bank conflict */
    // constexpr int register_tile = 4;
    // constexpr int shared_tile = 8;
    // dim3 block(shared_tile, shared_tile);
    // dim3 grid((N-1)/(shared_tile * register_tile)+1, (M-1)/(shared_tile * register_tile)+1);
    // Bank_GEMM<shared_tile, register_tile><<<grid, block>>> (d_A, d_B, d_C, M, N, K);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaError_t error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));

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