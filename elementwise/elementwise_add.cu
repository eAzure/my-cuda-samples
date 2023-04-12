#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 32 * 1024 * 1024;

void init_data(float *data) {
    for (int i=0;i<N;i++) {
        data[i] = 1;
    }
}

// raw kernel
__global__ void elementwise_add_v0(float *input1, float *input2, float *output) {
    int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_thread_id < N) output[global_thread_id] = input1[global_thread_id] + input2[global_thread_id];
}

// float2
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
__global__ void elementwise_add_v1(float *input1, float *input2, float *output) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    float2 reg_a = FETCH_FLOAT2(input1[idx]);
    float2 reg_b = FETCH_FLOAT2(input2[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(output[idx]) = reg_c;
}

// float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
__global__ void elementwise_add_v2(float *input1, float *input2, float *output) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    input1 += idx;
    input2 += idx;
    float4 reg_a = FETCH_FLOAT4(*input1);
    float4 reg_b = FETCH_FLOAT4(*input2);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(output[idx]) = reg_c;
}

int main() {
    float *h_input1;
    float *h_input2;
    float *h_output;
    cudaMallocHost(&h_input1, N * sizeof(float));
    cudaMallocHost(&h_input2, N * sizeof(float));
    cudaMallocHost(&h_output, N * sizeof(float));

    float *d_input1;
    float *d_input2;
    float *d_output;
    cudaMalloc(&d_input1, N * sizeof(float));
    cudaMalloc(&d_input2, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    init_data(h_input1);
    init_data(h_input2);

    cudaMemcpy(d_input1, h_input1, N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, N, cudaMemcpyHostToDevice);

    const int32_t thread_per_block = 256;
    // const int32_t block_num = N / thread_per_block;

    // raw implementation
    // elementwise_add_v0<<<block_num, thread_per_block>>> (d_input1, d_input2, d_output);

    // float2
    // const int32_t block_num = N / thread_per_block / 2;
    // elementwise_add_v1<<<block_num, thread_per_block>>> (d_input1, d_input2, d_output);

    // float4
    const int32_t block_num = N / thread_per_block / 4;
    elementwise_add_v2<<<block_num, thread_per_block>>> (d_input1, d_input2, d_output);

    cudaMemcpy(h_output, d_output, N, cudaMemcpyDeviceToHost);

    cout << h_output[0] << endl;

    cudaFreeHost(h_input1);
    cudaFreeHost(h_input2);
    cudaFreeHost(h_output);
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}