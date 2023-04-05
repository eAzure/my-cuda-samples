// 归约加实现
// 参考https://zhuanlan.zhihu.com/p/416959273
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

const int N = 4 * 1024 * 1024;

void random_init_data(float *data, size_t size) {
    for (size_t i=0;i<size;i++) {
        data[i] = float(rand()) / RAND_MAX;
    }
}

enum class ReduceType {
    Serial = 1,
    TwoPass = 2,
    TwoPassInterleaved = 3,
    TwoPassSharedOptimized = 4

};

/*** 0. serial 
 * 单线程实现
 * ***/
__global__ void SerialKernel(const float* input, float *output, size_t n) {
    float sum = 0.f;
    for (size_t i=0;i<n;i++) {
        sum += input[i];
    }
    *output = sum;
}

/*** 1. TwoPass
 * 第一个pass将原input归约到blockNum大小
 * 第二个pass得到结果
 * ***/
__global__ void TwoPassSimpleKernel(const float* input, float* output, size_t n) {
    // 划定每个block的计算范围（相对与整个n而言）
    size_t block_begin = n / gridDim.x * blockIdx.x;
    size_t block_end = n / gridDim.x *  (blockIdx.x+1);

    // 每个block 需要计算
    n = block_end - block_begin;
    // 更新input、output的起始
    input += block_begin;
    output += blockIdx.x; // 用于第一轮的pass，只保留blockNum个即可

    // 每个线程应该计算的范围
    size_t thread_begin = n / blockDim.x * threadIdx.x;
    size_t thread_end = n / blockDim.x * (threadIdx.x + 1);
    
    // 每一个线程的归约结果
    float thread_sum = 0.f;
    for (size_t i = thread_begin;i<thread_end;i++) {
        thread_sum += input[i];
    }
    // 将计算结果存到shared_memory中
    // shared_mem是针对一个block的，每一个block都有自己专属的shared_mem区域，所以这里不需要考虑block的因素
    extern __shared__ float shared_mem[];
    shared_mem[threadIdx.x] = thread_sum;
    __syncthreads();
    // 每个block里面的第一个线程完成该block的归约
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (size_t i = 0;i<blockDim.x;i++) {
            sum += shared_mem[i];
        }
        *output = sum;
    }
}

/*** 2. TwoPass改进版
 * 上面是一个线程访问同一片连续的区域，现在要同一个warp内的线程去访问同一片连续的区域
 * ***/
__global__ void TwoPassInterleavedKernel(const float* input, float* output, size_t n) {
    // 获取线程全局ID
    int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // 获取所有的线程数
    int32_t total_thread_num = gridDim.x * blockDim.x;
    float sum = 0.f;
    // 让相邻线程操作连续数据
    for (int32_t i=global_thread_id;i<n;i+=total_thread_num) {
        sum += input[i];
    }
    // 一个线程算的数量没变，相当于n/total_thread_num
    extern __shared__ float shared_mem[];
    shared_mem[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (size_t i=0;i<blockDim.x;i++) {
            sum += shared_mem[i];
        }
        output[blockIdx.x] = sum;
    } 
}

/*** 3. TwoPass改进版
 * 继续改进，将最后一步由thread 0串行归约的过程并行化，并尽可能避免bank_conflict
 * ***/
__global__ void TwoPassSharedOptimizedKernel(const float* input, float* output, size_t n) {
    int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_thread_num = gridDim.x * blockDim.x;

    float sum = 0.f;
    for (int32_t i=global_thread_id;i<n;i+=total_thread_num) {
        sum += input[i];
    }
    extern __shared__ float shared_mem[];
    shared_mem[threadIdx.x] = sum;
    __syncthreads();

    for (int32_t active_thread_num = blockDim.x/2;active_thread_num>=1;active_thread_num/=2) {
        if (threadIdx.x < active_thread_num) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + active_thread_num];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = shared_mem[0];
    }
}

void Reduce(const float* input, float* output, size_t n, ReduceType reduceType) {
    string type="";
    
    // 计时
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    // kernel
    switch (reduceType)
    {
    case ReduceType::Serial : {
        type = "Serial";
        SerialKernel<<<1, 1>>>(input, output, n);
        break;
    }
    case ReduceType::TwoPass : {
        type = "TwoPass";
        const int32_t thread_num_per_block = 1024;
        const int32_t block_num = 1024;

        size_t shared_mem_size = thread_num_per_block * sizeof(float);
        // 第一次pass
        TwoPassSimpleKernel<<<block_num, thread_num_per_block, shared_mem_size>>>(input, output, n);
        // 第二次pass
        TwoPassSimpleKernel<<<1, thread_num_per_block, shared_mem_size>>>(output, output, block_num);
        break;
    }
    case ReduceType::TwoPassInterleaved : {
        type = "TwoPassInterleaved";
        const int32_t thread_num_per_block = 1024;
        const int32_t block_num = 1024;

        size_t shared_mem_size = thread_num_per_block * sizeof(float);
        // 第一次pass
        TwoPassInterleavedKernel<<<block_num, thread_num_per_block, shared_mem_size>>>(input, output, n);
        // 第二次pass
        TwoPassInterleavedKernel<<<1, thread_num_per_block, shared_mem_size>>>(output, output, block_num);
    }
    case ReduceType::TwoPassSharedOptimized : {
        type = "TwoPassSharedOptimized";
        const int32_t thread_num_per_block = 1024;
        const int32_t block_num = 1024;

        size_t shared_mem_size = thread_num_per_block * sizeof(float);
        // 第一次pass
        TwoPassSharedOptimizedKernel<<<block_num, thread_num_per_block, shared_mem_size>>>(input, output, n);
        // 第二次pass
        TwoPassSharedOptimizedKernel<<<1, thread_num_per_block, shared_mem_size>>>(output, output, block_num);
    }
    default:
        break;
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    printf("%s Executing time: %f ms\n", type.c_str(), time_ms);
}


int main() {
    // host data
    float* h_input, * h_output;
    cudaMallocHost(&h_input, N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));

    random_init_data(h_input, N);

    // device data
    float* d_input, * d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_input, h_input, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // run

    // // reduce by serial: 178.283173ms
    // Reduce(d_input, d_output, N, ReduceType::Serial);
    // // reduce by twopass: 0.140544ms
    // Reduce(d_input, d_output, N, ReduceType::TwoPass);
    // // reduce by twopassinterleaved: 0.146688ms
    // Reduce(d_input, d_output, N, ReduceType::TwoPassInterleaved);
    // // reduce by twopasssharedoptimized: 0.068320ms
    Reduce(d_input, d_output, N, ReduceType::TwoPassSharedOptimized);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("result: %f\n", h_output[0]);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    return 0;
}
