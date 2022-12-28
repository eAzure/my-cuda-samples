/*
    using tensor core
    wmma api
*/
/*
    wmma api:
    template<typename Use, int M, int N, int K, typename T, typename Layout=void> class fragment;
        Use: matrix_a matrix_b accumulator
        M, N, K represent the size of the tile matrix_a's size: M * K
        T: half float int
        layout: row_major col_major
    void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
        mptr: 256bit align point to mem
        ldm: stride the multiply of 16
        layout: accumulator need declare "mem_row_major"
    void store_matrix_sync(const T* mptr, fragment<...>&a, unsigned ldm, layout_t layout);
    void fill_fragment(fragment<...> &a, const T& value);
        value to all element in a
    void mam_sync(fragment<...> &d, fragment<...> &a, fragment<...> &b, fragment<...> &c, bool staf=false);
        d = a * b + c

*/
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <functional>

using namespace std;
using namespace nvcuda;

const int M = 1024;
const int N = 1024;
const int K = 1024;

#define WARP_SIZE 32

/* random init the data */
void random_init_data(half *data, size_t size) {
    for (size_t i=0;i<size;i++) {
        data[i] = 1.0 * rand()  / RAND_MAX;
    }
}

/* check data with cpu compute */
bool check_with_cpu(half *A, half *B, float *C,
                    int m, int n, int k) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            float sum = 0.f;
            for (int h=0;h<k;h++) {
                sum += float(A[i*k+h]) * float(B[h*n+j]);
            }
            if (isnan(C[i * n + j])) {
                printf("C[%d][%d] is nan\n", i, j);
                return false;
            }
            // origin set 1e-5, but there may be some decline in accuracy.
            if (std::fabs(sum - C[i * n + j]) / std::fabs(sum) > 1e-4f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

/* record the executing time and the throughput */
void record_time_throughput(const char *kernel_tag, const function<void()> &kernel,
                            int trial) {
    float sum_time = 0.f;
    for (int i=0;i<trial;i++) {
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start);
        kernel();
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        cudaError_t error = cudaGetLastError();
        const char *error_info = cudaGetErrorString(error);
        if (strlen(error_info) != 8) {
            printf("CUDA error: %s, happens in iter: %d of kernel: %s\n", error_info, i, kernel_tag);
        }

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, end);
        sum_time += time_ms;

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    float time = 1.0 * sum_time / trial;
    printf("Executing time: %f ms\n", time);
    /* compute the throughput */
    long workload = long(M) * N * K * 2;
    double gflops = (double(workload) / 1e9) / (double(time) / 1e3);
    printf("GFLOPS: %f\n", gflops);
}

/* wmma kernel */
/* naive implement */
template <int M_tile=16, int N_tile=16, int K_tile=16>
__global__ void naive_wmma_kernel(half *A, half *B, float *C, int M, int N, int K) {
    // warp number per dim
    int warp_number_dim_n = N / N_tile;
    int warp_number_dim_k = K / K_tile;
    // thread locate warp id
    int thread_in_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    // transform thread_in_warp_id to 2-dim
    int thread_in_warp_m_id = thread_in_warp_id / warp_number_dim_n; // 将一维warp_id展为二维
    int thread_in_warp_n_id = thread_in_warp_id % warp_number_dim_n;

    wmma::fragment<wmma::matrix_a, M_tile, N_tile, K_tile, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, M_tile, N_tile, K_tile, half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, M_tile, N_tile, K_tile, float> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

    // locate the compute index in C
    float *C_store_index = C + thread_in_warp_m_id * M_tile *N + thread_in_warp_n_id * N_tile;

    // compute along k dim
    for (int kidx=0;kidx<warp_number_dim_k;kidx++) {
        half *A_load_index = A + thread_in_warp_m_id * M_tile * K + kidx * K_tile;
        half *B_load_index = B + kidx * K_tile * N + thread_in_warp_n_id * N_tile;

        wmma::load_matrix_sync(A_frag, A_load_index, K);
        wmma::load_matrix_sync(B_frag, B_load_index, N);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(C_store_index, C_frag, N, wmma::mem_row_major);
}

/* using shared memory */
template<int M_tile=16, int N_tile=16, int K_tile=16>
__global__ void shared_wmma_kernel(half *A, half *B, float *C, int M, int N, int K) {
    
}

int main() {
    /* host data */
    half *h_A, *h_B;
    float *h_C;
    cudaMallocHost(&h_A, M * K * sizeof(half));
    cudaMallocHost(&h_B, N * K * sizeof(half));
    cudaMallocHost(&h_C, M * N * sizeof(float));
    random_init_data(h_A, M * K);
    random_init_data(h_B, N * K);

    /* device data */
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, N * K * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    /* copy data from host to device */
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(half), cudaMemcpyHostToDevice);

    /* call the kernel and record the time and the throughput */

    /* call the kernel */
    constexpr int M_tile = 16;
    constexpr int N_tile = 16;
    constexpr int K_tile = 16;
    int GRID_DIM, BLOCK_DIM;
    int number_of_warp = (M/M_tile) * (N/N_tile);
    int number_of_thread = number_of_warp * WARP_SIZE;
    int BLOCK_DIM_DEFAULT = 512;
    
    if(number_of_thread < BLOCK_DIM_DEFAULT){
        GRID_DIM = 1;
        BLOCK_DIM = number_of_thread;
    }else{
        GRID_DIM = number_of_thread % BLOCK_DIM_DEFAULT ? 
            number_of_thread / BLOCK_DIM_DEFAULT + 1 : number_of_thread / BLOCK_DIM_DEFAULT ;
        BLOCK_DIM = BLOCK_DIM_DEFAULT;
    }
    record_time_throughput("wmma_naive", [&](){
        naive_wmma_kernel<M_tile, N_tile, K_tile><<<GRID_DIM, BLOCK_DIM>>>(d_A, d_B, d_C, M, N, K);
    ;}, 100);

    /* copy result from device to host */
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
