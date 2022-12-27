/* GEMM优化示例

C=αA∗B+βC

示例程序中：alpha=1,beta=0
*/

#include <sys/time.h>
#include <cuda.h>
#include <cublas.h>
#include <iostream>

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

/* 分块参数设置
*/
#define BM 128 // block子块大小
#define BN 128
#define BK 8
#define TM 8 // thread子块大小
#define TN 8

/* 矩阵类

Matrix是一个view（类似于Pytorch中的Tensor类型），与原矩阵共享数据，作为原矩阵的一个视图，主要用来访问矩阵数据
*/
template<typename T>
class Matrix
{
public:
    __device__ __host__ Matrix() = default;
    __device__ __host__ Matrix(const Matrix &) = default;
    __device__ __host__ Matrix& operator=(const Matrix &) = default;
    __device__ __host__ Matrix(T *_data,int _rows,int _cols,int _strideOfRow,int _strideOfCol):
                                    data(_data),
                                    rows(_rows),
                                    cols(_cols),
                                    strideOfRow(_strideOfRow),
                                    strideOfCol(_strideOfCol){}

    // 返回该矩阵所有字节数
    constexpr __device__ __host__ int GetNumberOfBytes() const
    {
        return rows*cols*sizeof(T);
    }

    // 返回该矩阵元素个数
    constexpr __device__ __host__ int GetNumberOfElements() const
    {
        return rows*cols;
    }

    // 访问某个元素，该元素的索引为二维逻辑索引：(rowIndex,colIndex)
    __device__ __host__ float &operator()(int rowIndex,int colIndex)
    {
        // 计算内存索引
        int memoryIndex=rowIndex*strideOfRow+colIndex*strideOfCol;

        return data[memoryIndex];
    }

    // 访问某个元素，该元素的索引为一维逻辑索引：(Index)
    __device__ __host__ float &operator()(int index)
    {
        // 转换为二维逻辑索引
        int colIndex=index%cols;
        int rowIndex=index/cols;

        // 计算内存索引
        int memoryIndex=rowIndex*strideOfRow+colIndex*strideOfCol;

        return data[memoryIndex];
    }



public:
    T *data = nullptr;// 数据指针
    int rows = 0;// 矩阵的行数
    int cols = 0;// 矩阵的列数
    int strideOfRow = 0;// 行步长
    int strideOfCol = 0;// 列步长

};

/* BlockGEMM_V2
*/
__global__ void BlockGEMM_V2(Matrix<float> A,Matrix<float> B,Matrix<float> C)
{
    // 每个线程的计算结果
    float c[TM][TN]={0.0};
    float a[TM]={0.0};
    float b[TN]={0.0};

    // 沿着K维度循环加载一个block中对应的A和B的数据到共享内存
    for(int i=0;i<A.cols/BK;++i)
    {
        // 每个block对应的全局内存中的A,B子块，即创建全局内存中A,B的view
        Matrix<float> ASub(A.data+blockIdx.y*BM*A.strideOfRow+i*BK,BM,BK,A.strideOfRow,A.strideOfCol);
        Matrix<float> BSub(B.data+i*BK*B.strideOfRow+blockIdx.x*BN,BK,BN,B.strideOfRow,B.strideOfCol);

        // 将Asub,BSub加载到共享内存
        // 以block为128，thread为8为例：由于一个block有16x16=256个线程，而ASub和BSub中一共有1024个元素，所以每个线程加载4个元素
        // 注意：这里需要将一维逻辑索引转换为多维逻辑索引：stardIndex->(stardIndex/cols,stardIndex%cols)
        __shared__ float A_Shared[BM][BK];
        __shared__ float B_Shared[BK][BN];
        int numberOfElementsPerThread=(BK*BM)/(blockDim.x*blockDim.y);// 每个线程需要读取多少数据
        int stardIndex=numberOfElementsPerThread*(threadIdx.y*blockDim.x+threadIdx.x);// stardIndex为每个线程读取的起始索引
        for(int threadIndex=0;threadIndex<numberOfElementsPerThread;++threadIndex)
        {
            int logicalIndex=stardIndex+threadIndex;
            A_Shared[logicalIndex/BK][logicalIndex%BK]=ASub(logicalIndex/BK,logicalIndex%BK);
            B_Shared[logicalIndex/BN][logicalIndex%BN]=BSub(logicalIndex/BN,logicalIndex%BN);
        }
        __syncthreads();

        // 每个thread对应的共享内存中的A_Shared,B_Shared的子块，即创建A_Shared,B_Shared的view
        Matrix<float> ASub_Shared((float *)A_Shared+threadIdx.y*TM*BK,TM,BK,BK,1);// 每个线程对应的共享内存中A和B的子块
        Matrix<float> BSub_Shared((float *)B_Shared+threadIdx.x*TN,BK,TN,BN,1);

        // 每个线程执行计算
        for(int k=0;k<BK;++k)
        {
            // 先将A的一列和B的一行加载到寄存器
            for(int m=0;m<TM;++m)
            {
                a[m]=ASub_Shared(m,k);
            }
            for(int n=0;n<TN;++n)
            {
                b[n]=BSub_Shared(k,n);
            }

            // 使用寄存器计算
            for(int m=0;m<TM;++m)
            {
                for(int n=0;n<TN;++n)
                {
                    c[m][n]+=a[m]*b[n];
                }
            }
        }
        __syncthreads();

    }

    // 将每个线程计算好的结果写回到C矩阵
    // CSub为每个线程对应的全局内存的C矩阵子块，创建C矩阵的view
    Matrix<float> CSub(C.data+((blockIdx.y*BM+threadIdx.y*TM)*C.strideOfRow+blockIdx.x*BN+threadIdx.x*TN),TM,TN,C.strideOfRow,C.strideOfCol);
    for(int m=0;m<TM;++m)
    {
        for(int n=0;n<TN;++n)
        {
            CSub(m,n)=c[m][n];
        }
    }

}

#define MATRIX_SIZE 1024
int main(int argc,char *argv[])
{
    // 创建CPU A,B矩阵
    float *A_Host, *B_Host, *C_Host;
    cudaMallocHost(&A_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost(&B_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMallocHost(&C_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // 创建GPU A矩阵
    float *dataOfA_Device=nullptr;
    cudaMalloc((void **)&dataOfA_Device, MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    cudaMemcpy(dataOfA_Device, A_Host, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    Matrix<float> A_Device(dataOfA_Device, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, 1);

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
        BlockGEMM_V2<<<grid, block>>>(A_Device,B_Device,C_Device);
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
