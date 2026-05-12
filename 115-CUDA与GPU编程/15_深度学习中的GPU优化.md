# 深度学习中的GPU优化

## 混合精度训练

混合精度训练使用FP16（半精度）和BF16（Brain Floating Point）来加速训练，同时保持FP32的精度。

### FP16 vs BF16 vs FP32

```
FP32:  [1 sign][8 exponent][23 mantissa]  32位, 最大精度
FP16:  [1 sign][5 exponent][10 mantissa]  16位, 范围小易溢出
BF16:  [1 sign][8 exponent][ 7 mantissa]  16位, 范围与FP32相同, 精度低

       符号  指数  尾数    范围              精度
FP32   1     8     23      ±3.4e38          高
FP16   1     5     10      ±65504           低（易溢出下溢）
BF16   1     8     7       ±3.4e38          低（与FP32同范围）
```

### CUDA混合精度核心代码

```cpp
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// FP16矩阵乘法示例（利用Tensor Core）
__global__ void matmul_fp16(const __half* A, const __half* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // __half2float: FP16 → FP32 进行累加
            float a = __half2float(A[row * K + k]);
            float b = __half2float(B[k * N + col]);
            sum += a * b;
        }
        C[row * N + col] = sum;  // FP32存储结果
    }
}

// FP16向量加法
__global__ void vector_add_fp16(const __half* A, const __half* B, __half* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // __hadd2 可以同时处理两个FP16值
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

// 批量FP16转换
void convert_fp32_to_fp16(const float* src, __half* dst, int N) {
    for (int i = 0; i < N; i++) {
        dst[i] = __float2half(src[i]);
    }
}
```

## Tensor Core编程

Tensor Core是Volta架构（V100）引入的专用矩阵运算单元，能大幅加速深度学习中的矩阵乘加运算。

### WMMA API（Warp Matrix Multiply Accumulate）

```cpp
#include <mma.h>
using namespace nvcuda;

// Tensor Core矩阵乘法
// 使用WMMA API操作16x16x16的矩阵块
__global__ void tensor_core_matmul(const half* A, const half* B, float* C,
                                    int M, int N, int K) {
    // 声明WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 初始化累加器为0
    wmma::fill_fragment(c_frag, 0.0f);

    // warp的行列位置
    int warpRow = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpCol = (blockIdx.x * blockDim.x + threadIdx.x);

    // 每个warp处理一个16x16的输出块
    int row = warpRow * 16;
    int col = warpCol * 16;

    if (row < M && col < N) {
        // 遍历K维度的tile
        for (int k = 0; k < K; k += 16) {
            // 加载A和B的16x16块到fragments
            wmma::load_matrix_sync(a_frag, A + row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + col, N);

            // 矩阵乘加: c_frag += a_frag * b_frag
            // 这会在Tensor Core上执行！
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // 存储结果
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}
```

### cuBLASLt Tensor Core加速

```cpp
#include <cublasLt.h>

// 使用cuBLASLt进行Tensor Core加速的矩阵乘法
void cublaslt_tensor_core_example() {
    int M = 4096, N = 4096, K = 4096;

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M);

    float alpha = 1.0f, beta = 0.0f;

    // cuBLASLt会自动选择Tensor Core算法
    cublasLtMatmul(ltHandle, operationDesc,
                   &alpha, d_A, Adesc, d_B, Bdesc,
                   &beta, d_C, Cdesc, d_C, Cdesc,
                   nullptr, nullptr, 0, 0);

    printf("Tensor Core matrix multiplication completed\n");

    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtDestroy(ltHandle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
```

## 梯度累积

梯度累积允许在显存有限的情况下模拟更大的batch size：

```cpp
// 梯度累积：多次前向/反向传播后才更新参数
__global__ void accumulate_gradients(float* accumulated_grad,
                                      const float* current_grad,
                                      int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        accumulated_grad[idx] += current_grad[idx];
    }
}

__global__ void apply_gradients(float* weights, const float* gradients,
                                 float learning_rate, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

__global__ void zero_gradients(float* gradients, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        gradients[idx] = 0.0f;
    }
}

void gradient_accumulation_training() {
    int N = 1024 * 1024;
    int accumulation_steps = 4;
    float lr = 0.001f;

    float *d_weights, *d_accum_grads, *d_current_grads;
    cudaMalloc(&d_weights, N * sizeof(float));
    cudaMalloc(&d_accum_grads, N * sizeof(float));
    cudaMalloc(&d_current_grads, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int epoch = 0; epoch < 100; epoch++) {
        // 清零累积梯度
        zero_gradients<<<blocks, threads>>>(d_accum_grads, N);

        // 累积多个mini-batch的梯度
        for (int step = 0; step < accumulation_steps; step++) {
            // 前向传播（简化）
            // forward_pass(...)

            // 反向传播得到d_current_grads
            // backward_pass(...)

            // 累积梯度
            accumulate_gradients<<<blocks, threads>>>(
                d_accum_grads, d_current_grads, N);
        }

        // 使用累积的梯度更新参数（等效于accumulation_steps倍batch size）
        apply_gradients<<<blocks, threads>>>(
            d_weights, d_accum_grads, lr / accumulation_steps, N);
    }

    cudaFree(d_weights);
    cudaFree(d_accum_grads);
    cudaFree(d_current_grads);
}
```

## 数据并行训练

### 单机多GPU数据并行

```cpp
// 数据并行：每个GPU处理不同的数据batch
__global__ void forward_kernel(float* input, float* weights, float* output,
                                int batch_size, int in_features, int out_features) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int out_idx = idx % out_features;
    int batch_idx = idx / out_features;

    if (batch_idx < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[batch_idx * in_features + i] * weights[out_idx * in_features + i];
        }
        output[batch_idx * out_features + out_idx] = sum;
    }
}

// 梯度平均：All-Reduce的简化实现
__global__ void average_gradients(float** device_grads, float* avg_grad,
                                   int N, int num_gpus) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int g = 0; g < num_gpus; g++) {
            sum += device_grads[g][idx];
        }
        avg_grad[idx] = sum / num_gpus;
    }
}

void multi_gpu_data_parallel() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 2) {
        printf("Need at least 2 GPUs for data parallel training\n");
        return;
    }

    int N = 1024 * 1024;
    int batch_per_gpu = 32;

    // 每个GPU上的数据和梯度
    float* d_weights[num_gpus];
    float* d_grads[num_gpus];

    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_weights[g], N * sizeof(float));
        cudaMalloc(&d_grads[g], N * sizeof(float));
    }

    // 训练循环
    for (int epoch = 0; epoch < 10; epoch++) {
        // 每个GPU独立计算前向和反向
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            // forward_kernel<<<...>>>(d_input[g], d_weights[g], d_output[g], ...);
            // backward_kernel<<<...>>>(d_weights[g], d_grads[g], ...);
        }

        // 同步所有GPU
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }

        // 梯度平均（简化实现，实际应使用NCCL）
        cudaSetDevice(0);
        // average_gradients<<<...>>>(d_grads, d_avg_grad, N, num_gpus);

        // 广播平均梯度到所有GPU
        for (int g = 1; g < num_gpus; g++) {
            cudaSetDevice(g);
            // cudaMemcpyPeer(d_grads[g], g, d_avg_grad, 0, N * sizeof(float));
        }

        // 每个GPU更新自己的权重
        for (int g = 0; g < num_gpus; g++) {
            cudaSetDevice(g);
            // apply_gradients<<<...>>>(d_weights[g], d_grads[g], lr, N);
        }
    }

    // 清理
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(d_weights[g]);
        cudaFree(d_grads[g]);
    }
}
```

## 算子融合

将多个小kernel合并为一个大kernel，减少kernel启动开销和全局内存访问：

```cpp
// 融合前：3个独立的kernel，3次全局内存读写
__global__ void add_kernel(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

__global__ void relu_kernel(float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = fmaxf(c[idx], 0.0f);
}

__global__ void scale_kernel(float* c, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = c[idx] * scale;
}

// 融合后：1个kernel，1次读1次写
__global__ void fused_add_relu_scale(const float* a, const float* b,
                                      float* out, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Add + ReLU + Scale 全部在寄存器中完成
        float val = a[idx] + b[idx];  // 加法
        val = fmaxf(val, 0.0f);       // ReLU
        out[idx] = val * scale;       // 缩放，只写一次全局内存
    }
}
```

## 注意力机制优化

```cpp
// Flash Attention思想：分块计算避免大矩阵的中间存储
__global__ void flash_attention_block(const float* Q, const float* K,
                                       const float* V, float* O,
                                       int N, int d, int block_size) {
    // 使用共享内存分块处理
    extern __shared__ float s_mem[];
    float* s_K = s_mem;
    float* s_V = s_mem + block_size * d;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // 分块遍历KV
    for (int kv_start = 0; kv_start < N; kv_start += block_size) {
        // 加载K和V的块到共享内存
        int kv_idx = kv_start + threadIdx.x;
        if (threadIdx.x < block_size && kv_idx < N) {
            for (int j = 0; j < d; j++) {
                s_K[threadIdx.x * d + j] = K[kv_idx * d + j];
                s_V[threadIdx.x * d + j] = V[kv_idx * d + j];
            }
        }
        __syncthreads();

        // 计算Q与当前K块的注意力
        float local_max = -1e30f;
        float local_sum = 0.0f;

        int kv_end = min(kv_start + block_size, N);
        for (int kv = kv_start; kv < kv_end; kv++) {
            float score = 0.0f;
            int local_kv = kv - kv_start;
            for (int j = 0; j < d; j++) {
                score += Q[row * d + j] * s_K[local_kv * d + j];
            }
            score /= sqrtf((float)d);

            // Softmax在线计算
            float exp_score = expf(score - local_max);
            local_sum = local_sum * expf(local_max - score) + 1.0f;
            local_max = fmaxf(local_max, score);

            // 累加到输出
            for (int j = 0; j < d; j++) {
                // 在线更新O
            }
        }
        __syncthreads();
    }
}
```

## 小结

1. 混合精度训练（FP16/BF16 + FP32）利用Tensor Core大幅加速训练
2. 梯度累积可以在显存受限时模拟更大的batch size
3. 数据并行是多GPU训练的基本模式，需要高效的梯度同步
4. 算子融合减少kernel启动次数和全局内存访问
5. Flash Attention等新方法通过分块计算优化大模型的注意力机制
