/*
    main purpose: to show the Thought of the Matrix View
    refer to https://blog.csdn.net/qianqing13579/article/details/127359866
*/

#include <cuda_runtime.h>
#include <iostream>


/* check with cpu compute */
bool check_with_cpu(float *A, float *B, float *C,
                    int m, int n, int k) {
    for (int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            if (isnan(C[i* n + j])) {
                printf("C[%d][%d] is nan\n", i, j);
                return false;
            }
            float sum = 0.f;
            for (int h=0;h<k;h++) {
                sum += A[i*k+h] * B[k*n+j];
            }
            if (std::fabs(sum - C[i  * n + j]) / std::fabs(sum) > 1e-5f) {
                printf("C[%d][%d] not match, %f vs %f\n", i, j, sum, C[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

/* set the block tile and register tile */
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

/* The Matrix View Class */
template<typename T>
class Matrix{
public:
    __device__ __host__ Matrix() = default;
    // 拷贝构造函数，用于初始化
    __device__ __host__ Matrix(const Matrix &) = default;
    // 赋值运算符，用于赋值
    __device__ __host__ Matrix& operator=(const Matrix&) = default;
    __device__ __host__ Matrix(T *_data, int _rows, int _cols, int _strideOfRow, int _strideOfCol):
                               data(_data), rows(_rows), cols(_cols),
                               strideOfRow(_strideOfRow), strideOfCol(_strideOfCol) {}

    // get data bytes
    constexpr __device__ __host__ int GetNumberOfBytes() const {
        return rows * cols * sizeof(T);
    }

    // get data number
    constexpr __device__ __host__ int GetNumberOfElements() const {
        return rows * cols;
    }

    // 2-dim logical index, overload the (a, b)
    __device__ __host__ float &operator()(int rowIndex, int colIndex) {
        // get the actual memory index
        int memoryIndex = rowIndex * strideOfRow + colIndex * strideOfCol;
        return data[memoryIndex];
    }

    // 1-dim logical index, overload the (a)
    __device__ __host__ float &operator()(int index) {
        // transform to 2-dim logical index
        int rowIndex = index / cols;
        int colIndex = index % cols;
        // get the actual memory index
        int memoryIndex = rowIndex * strideOfRow + colIndex * strideOfCol;
        return data[memoryIndex];
    }

public:
    T *data = nullptr; // pointer to the actual data address
    int rows = 0; // the row of matrix view
    int cols = 0; // the column of the matrix view
    int strideOfRow = 0; // the row stride in actual data address
    int strideOfCol = 0; // the column stride in actual data address
};

/* GEMM kernel */
__global__ void GEMM(Matrix<float> A, Matrix<float> B, Matrix<float> C) {
    // register
    float A_register[TM] = {0.f};
    float B_register[TN] = {0.f};
    float C_register[TM][TN] = {0.f};

    for (int i=0;i<A.cols/BK;i++) {
        // Asub Bsub corresponds to the block tile
        // so we should set the data to the initial block pos.
        Matrix<float> ASub(A.data+blockIdx.y*BM*A.strideOfRow + i*BK,
                           BM, BK, A.strideOfRow, A.strideOfCol);
        Matrix<float> BSub(B.data+i*BK*B.strideOfRow+blockIdx.x*BN,
                           BK, BN, B.strideOfRow, B.strideOfCol);

        // load data from global to shared
        __shared__ float A_shared[BM][BK];
        __shared__ float B_shared[BK][BN];

        int numberOfElementsPerThread = (BK * BM) / (blockDim.x * blockDim.y);
        int startIndex = numberOfElementsPerThread * (threadIdx.y * blockDim.x + threadIdx.x);
        for (int threadIndex=0;threadIndex<numberOfElementsPerThread;threadIndex++) {
            int logicalIndex = startIndex + threadIndex;
            A_shared[logicalIndex/BK][logicalIndex%BK] = ASub(logicalIndex/BK, logicalIndex%BK);
            B_shared[logicalIndex/BN][logicalIndex%BN] = BSub(logicalIndex/BN, logicalIndex%BN);
        }
        __syncthreads();

        // ASub_shared BSub_shared corresponds to the thread read in the shared mem
        Matrix<float> ASub_shared((float*)A_shared + threadIdx.y * TM * BK,
                                   TM, BK, BK, 1);
        Matrix<float> BSub_shared((float*)B_shared + threadIdx.x * TN,
                                   BK, TN, BN, 1);

        for (int k=0;k<BK;k++) {
            // load data from shared to register
            for (int m=0;m<TM;m++) {
                A_register[m] = ASub_shared(m, k);
            }
            for (int n=0;n<TN;n++) {
                B_register[n] = BSub_shared(k, n);
            }
            // compute
            for (int m=0;m<TM;m++) {
                for (int n=0;n<TN;n++) {
                    C_register[m][n] += A_register[m] * B_register[n];
                }
            }
        }
        __syncthreads();
    }

    // store data from register to global
    Matrix<float> CSub(C.data+((blockIdx.y*BM+threadIdx.y*TM) * C.strideOfRow+blockIdx.x*BN+threadIdx.x*TN),
                       TM, TN, C.strideOfRow, C.strideOfCol);
    for (int m=0;m<TM;m++) {
        for (int n=0;n<TN;n++) {
            CSub(m, n) = C_register[m][n];
        }
    }
}

#define MATRIX_SIZE 1024
int main(int argc, char *argv[]) {
    // host data
    float *A_Host, *B_Host, *C_Host;
    cudaMallocHost(&A_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost(&B_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost(&C_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // device data
    float *dataOfA_Device = nullptr;
    cudaMalloc((void **)&dataOfA_Device, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMemcpy(dataOfA_Device, A_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    Matrix<float> A_Device(dataOfA_Device,  MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1);

    // 创建GPU B矩阵
    float *dataOfB_Device=nullptr;
    cudaMalloc((void **)&dataOfB_Device, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMemcpy(dataOfB_Device, B_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    Matrix<float> B_Device(dataOfB_Device, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1);

    // 创建GPU C矩阵
    float *dataOfC_Device=nullptr;
    cudaMalloc((void **)&dataOfC_Device, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    Matrix<float> C_Device(dataOfC_Device, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1);


    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    //////////////////////////////// BlockGEMM_V2 /////////////////////////////////////////////
    {
        int BLOCKX = 16;// 每个block的x方向线程数
        int BLOCKY = 16;// 每个block的y方向线程数
        dim3 block(BLOCKX,BLOCKY);
        dim3 grid(C_Device.cols/BN,C_Device.rows/BM);
        GEMM<<<grid, block>>>(A_Device,B_Device,C_Device);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time_ms;
    cudaEventElapsedTime(&time_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("Executing time: %f ms\n", time_ms);

    long workload = long(MATRIX_SIZE) * MATRIX_SIZE * MATRIX_SIZE * 2;
    double gflops = (double(workload) / 1e9) / (double(time_ms) / 1e3);
    printf("GFLOPS: %f\n", gflops);

    // 拷贝GPU结果
    float *dataOfC_DeviceToHost=nullptr;
    dataOfC_DeviceToHost=(float *)malloc(C_Device.GetNumberOfBytes());
    cudaMemcpy(dataOfC_DeviceToHost, C_Device.data, C_Device.GetNumberOfBytes(), cudaMemcpyDeviceToHost);

    // 验证结果的正确性
    bool result_check = check_with_cpu(A_Host, B_Host, dataOfC_DeviceToHost, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
    printf("Check result: %s\n", result_check ? "OK" : "Failed");

    // free
    cudaFree(dataOfA_Device);
    cudaFree(dataOfB_Device);
    cudaFree(dataOfC_Device);
    free(dataOfC_DeviceToHost);
    cudaFree(A_Host);
    cudaFree(B_Host);
    cudaFree(C_Host);

    return 0;
}