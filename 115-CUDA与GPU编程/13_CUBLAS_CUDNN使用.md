# cuBLAS与cuDNN使用

## cuBLAS概述

cuBLAS是NVIDIA提供的GPU线性代数库，实现了BLAS（Basic Linear Algebra Subprograms）接口。它是高性能矩阵运算的基础。

## cuBLAS基本使用

```cpp
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// cuBLAS使用列主序（column-major），与C的行主序不同
void cublas_sgemm_example() {
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.0f;

    // 分配内存
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    // 初始化（行主序）
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 执行 C = alpha * A * B + beta * C
    // 注意：cuBLAS使用列主序
    // 如果输入是行主序，计算 C = A * B 等价于 cuBLAS的 C^T = B^T * A^T
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
                N, M, K,                    // 注意维度顺序！
                &alpha,
                d_B, N,                     // B^T in row-major
                d_A, K,                     // A^T in row-major
                &beta,
                d_C, N);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("C[0][0] = %f (expected: %d)\n", h_C[0], K);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
}

int main() {
    cublas_sgemm_example();
    return 0;
}
```

## cuBLAS常用函数

```cpp
// 向量点积: result = x · y
float cublas_dot(cublasHandle_t handle, float* d_x, float* d_y, int N) {
    float result;
    cublasSdot(handle, N, d_x, 1, d_y, 1, &result);
    return result;
}

// 向量缩放: x = alpha * x
void cublas_scale(cublasHandle_t handle, float* d_x, float alpha, int N) {
    cublasSscal(handle, N, &alpha, d_x, 1);
}

// 向量加法: y = alpha * x + y
void cublas_axpy(cublasHandle_t handle, float* d_x, float* d_y,
                  float alpha, int N) {
    cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);
}

// 矩阵向量乘法: y = alpha * A * x + beta * y
void cublas_gemv(cublasHandle_t handle, float* d_A, float* d_x, float* d_y,
                  int M, int N, float alpha, float beta) {
    cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha,
                d_A, M, d_x, 1, &beta, d_y, 1);
}

// 矩阵乘法: C = alpha * op(A) * op(B) + beta * C
void cublas_gemm(cublasHandle_t handle,
                  float* d_A, float* d_B, float* d_C,
                  int M, int N, int K,
                  float alpha, float beta) {
    // 行主序输入时的列主序等价计算
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N, d_A, K, &beta, d_C, N);
}

// 批量矩阵乘法
void cublas_batched_gemm(cublasHandle_t handle,
                          float* d_Aarray[], float* d_Barray[], float* d_Carray[],
                          int M, int N, int K, int batchSize) {
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemmBatched(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K, &alpha,
                       d_Barray, N, d_Aarray, K,
                       &beta, d_Carray, N, batchSize);
}
```

## cuDNN概述

cuDNN是深度学习专用的GPU加速库，提供了高度优化的卷积、池化、归一化等操作。

## cuDNN基本使用

```cpp
#include <cstdio>
#include <cudnn.h>

// 错误检查宏
#define CUDNN_CHECK(call)                                                   \
    do {                                                                    \
        cudnnStatus_t status = call;                                        \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuDNN Error: %s\n", cudnnGetErrorString(status));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

void cudnn_convolution_example() {
    // 创建cuDNN句柄
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    // 定义张量维度 (NCHW格式)
    int n = 1, c = 3, h = 224, w = 224;         // 输入: 1x3x224x224
    int k = 64, kh = 3, kw = 3;                  // 卷积核: 64x3x3x3
    int pad = 1, stride = 1, dilation = 1;

    // 计算输出尺寸
    int out_h = (h + 2 * pad - kh) / stride + 1;  // 224
    int out_w = (w + 2 * pad - kw) / stride + 1;  // 224

    // 创建张量描述符
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 设置描述符
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, n, c, h, w));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW, k, c, kh, kw));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride,
                                                  dilation, dilation,
                                                  CUDNN_CROSS_CORRELATION,
                                                  CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT, n, k, out_h, out_w));

    // 选择最优卷积算法
    cudnnConvolutionFwdAlgoPerf_t perfResults[1];
    int returnedAlgoCount;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(cudnn,
                input_desc, filter_desc, conv_desc, output_desc,
                1, &returnedAlgoCount, perfResults));
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;

    printf("Best convolution algorithm: %d\n", algo);

    // 分配GPU内存
    size_t input_bytes = n * c * h * w * sizeof(float);
    size_t filter_bytes = k * c * kh * kw * sizeof(float);
    size_t output_bytes = n * k * out_h * out_w * sizeof(float);

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_filter, filter_bytes);
    cudaMalloc(&d_output, output_bytes);

    // 分配工作空间
    size_t workspace_bytes;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                input_desc, filter_desc, conv_desc, output_desc,
                algo, &workspace_bytes));
    void* d_workspace;
    cudaMalloc(&d_workspace, workspace_bytes);

    // 执行卷积
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn, &alpha,
                input_desc, d_input,
                filter_desc, d_filter,
                conv_desc, algo,
                d_workspace, workspace_bytes,
                &beta,
                output_desc, d_output));

    printf("Convolution completed: %dx%dx%dx%d → %dx%dx%dx%d\n",
           n, c, h, w, n, k, out_h, out_w);

    // 清理
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_output);
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
}

int main() {
    cudnn_convolution_example();
    return 0;
}
```

## cuDNN池化操作

```cpp
void cudnn_pooling_example() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    int n = 1, c = 64, h = 112, w = 112;

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnPoolingDescriptor_t pool_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreatePoolingDescriptor(&pool_desc);

    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                n, c, h, w);
    cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX,
                                 CUDNN_NOT_PROPAGATE_NAN,
                                 2, 2,  // 窗口大小
                                 0, 0,  // padding
                                 2, 2); // stride

    int out_n, out_c, out_h, out_w;
    cudnnGetPooling2dForwardOutputDim(pool_desc, in_desc,
                                       &out_n, &out_c, &out_h, &out_w);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                out_n, out_c, out_h, out_w);

    printf("Max pool 2x2: %dx%dx%dx%d → %dx%dx%dx%d\n",
           n, c, h, w, out_n, out_c, out_h, out_w);

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * c * h * w * sizeof(float));
    cudaMalloc(&d_output, out_n * out_c * out_h * out_w * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    cudnnPoolingForward(cudnn, pool_desc, &alpha,
                        in_desc, d_input, &beta,
                        out_desc, d_output);

    cudaFree(d_input); cudaFree(d_output);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroy(cudnn);
}
```

## cuDNN批归一化

```cpp
void cudnn_batchnorm_example() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    int n = 32, c = 64, h = 56, w = 56;

    cudnnTensorDescriptor_t tensor_desc, bn_desc;
    cudnnCreateTensorDescriptor(&tensor_desc);
    cudnnCreateTensorDescriptor(&bn_desc);

    cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                n, c, h, w);
    // BatchNorm参数是per-channel的
    cudnnDeriveBNTensorDescriptor(bn_desc, tensor_desc,
                                   CUDNN_BATCHNORM_SPATIAL);

    float *d_input, *d_output, *d_scale, *d_bias;
    float *d_running_mean, *d_running_var;
    size_t param_bytes = c * sizeof(float);

    cudaMalloc(&d_input, n * c * h * w * sizeof(float));
    cudaMalloc(&d_output, n * c * h * w * sizeof(float));
    cudaMalloc(&d_scale, param_bytes);
    cudaMalloc(&d_bias, param_bytes);
    cudaMalloc(&d_running_mean, param_bytes);
    cudaMalloc(&d_running_var, param_bytes);

    float alpha = 1.0f, beta = 0.0f;
    double epsilon = 1e-5;
    double exponentialAverageFactor = 0.1;

    cudnnBatchNormalizationForwardTraining(cudnn,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta,
        tensor_desc, d_input,
        tensor_desc, d_output,
        bn_desc, d_scale, d_bias,
        exponentialAverageFactor,
        d_running_mean, d_running_var,
        epsilon, nullptr, nullptr);

    printf("Batch normalization completed\n");

    // 清理
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_scale); cudaFree(d_bias);
    cudaFree(d_running_mean); cudaFree(d_running_var);
    cudnnDestroyTensorDescriptor(tensor_desc);
    cudnnDestroyTensorDescriptor(bn_desc);
    cudnnDestroy(cudnn);
}
```

## 与PyTorch集成

cuBLAS和cuDNN是PyTorch GPU后端的基础：

```python
# PyTorch内部使用cuBLAS/cuDNN
import torch

# 矩阵乘法 → cuBLAS Sgemm/Hgemm
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = torch.mm(a, b)  # 调用cublasSgemm

# 卷积 → cuDNN
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
x = torch.randn(1, 3, 224, 224, device='cuda')
y = conv(x)  # 调用cudnnConvolutionForward

# 查看PyTorch使用的cuDNN算法
torch.backends.cudnn.benchmark = True  # 自动选择最快算法
torch.backends.cudnn.deterministic = True  # 确保可复现性
```

```cpp
// 从C++调用PyTorch的CUDA操作（LibTorch C++ API）
#include <torch/torch.h>
#include <torch/script.h>

void libtorch_example() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto a = torch::randn({1024, 1024}, options);
    auto b = torch::randn({1024, 1024}, options);

    // 矩阵乘法（底层调用cuBLAS）
    auto c = torch::mm(a, b);

    // 卷积（底层调用cuDNN）
    torch::nn::Conv2d conv(torch::nn::Conv2dOptions(3, 64, 3).padding(1));
    conv->to(torch::kCUDA);
    auto x = torch::randn({1, 3, 224, 224}, options);
    auto y = conv(x);
}
```

## 小结

1. cuBLAS提供BLAS级别的GPU线性代数运算，注意列主序存储
2. cuDNN提供深度学习专用的优化操作：卷积、池化、归一化
3. 使用`cudnnFindConvolutionForwardAlgorithm`自动选择最优卷积算法
4. PyTorch/TensorFlow底层都依赖cuBLAS和cuDNN
5. 在实际项目中优先使用cuBLAS/cuDNN而非手写kernel
