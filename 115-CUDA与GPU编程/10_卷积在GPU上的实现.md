# 卷积在GPU上的实现

## 卷积基础

卷积是深度学习和信号处理中最核心的操作之一。在GPU上高效实现卷积需要充分利用内存层次结构。

```
2D卷积示意：
输入图像 (H x W)          卷积核 (Kh x Kw)         输出 (H' x W')
┌─────────────┐          ┌───────┐                ┌──────────┐
│ . . . . . . │          │ w  w  │                │   . .    │
│ . ┌───┐ . . │    *     │ w  w  │      =         │   . .    │
│ . │   │ . . │          │ w  w  │                │   . .    │
│ . └───┘ . . │          └───────┘                └──────────┘
│ . . . . . . │
└─────────────┘
```

## 朴素2D卷积实现

```cpp
// 朴素卷积：直接从全局内存读取
__global__ void conv2d_naive(const float* input, const float* kernel,
                              float* output, int H, int W,
                              int KH, int KW, int pad) {
    int out_col = threadIdx.x + blockIdx.x * blockDim.x;
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;

    int out_H = H + 2 * pad - KH + 1;
    int out_W = W + 2 * pad - KW + 1;

    if (out_row < out_H && out_col < out_W) {
        float sum = 0.0f;
        for (int ky = 0; ky < KH; ky++) {
            for (int kx = 0; kx < KW; kx++) {
                int in_row = out_row - pad + ky;
                int in_col = out_col - pad + kx;

                float val = 0.0f;
                // 边界检查：padding区域值为0
                if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
                    val = input[in_row * W + in_col];
                }
                sum += val * kernel[ky * KW + kx];
            }
        }
        output[out_row * out_W + out_col] = sum;
    }
}
```

## 使用常量内存存储卷积核

卷积核通常较小且只读，非常适合放在常量内存中：

```cpp
// 常量内存存储卷积核（最大7x7=49个元素）
__constant__ float d_kernel[49];

__global__ void conv2d_const_mem(const float* input, float* output,
                                  int H, int W, int KH, int KW, int pad) {
    int out_col = threadIdx.x + blockIdx.x * blockDim.x;
    int out_row = threadIdx.y + blockIdx.y * blockDim.y;

    int out_H = H + 2 * pad - KH + 1;
    int out_W = W + 2 * pad - KW + 1;

    if (out_row < out_H && out_col < out_W) {
        float sum = 0.0f;
        for (int ky = 0; ky < KH; ky++) {
            for (int kx = 0; kx < KW; kx++) {
                int in_row = out_row - pad + ky;
                int in_col = out_col - pad + kx;
                float val = 0.0f;
                if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
                    val = input[in_row * W + in_col];
                }
                // 从常量内存读取卷积核（同一warp读同一地址时有广播加速）
                sum += val * d_kernel[ky * KW + kx];
            }
        }
        output[out_row * out_W + out_col] = sum;
    }
}

void launch_conv_const(float* d_input, float* h_kernel, float* d_output,
                        int H, int W, int KH, int KW) {
    int pad = KW / 2;
    // 将卷积核拷贝到常量内存
    cudaMemcpyToSymbol(d_kernel, h_kernel, KH * KW * sizeof(float));

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16);
    conv2d_const_mem<<<blocks, threads>>>(d_input, d_output, H, W, KH, KW, pad);
}
```

## Tiled卷积（共享内存优化）

类似矩阵乘法的tiled优化，将输入数据分块加载到共享内存：

```cpp
#define TILE_SIZE 16
#define KERNEL_RADIUS 3   // 7x7卷积核，半径为3
#define KERNEL_SIZE 7

__global__ void conv2d_tiled(const float* input, const float* kernel,
                              float* output, int H, int W) {
    // 共享内存需要包含halo区域
    // 输出tile: TILE_SIZE x TILE_SIZE
    // 输入tile: (TILE_SIZE + 2*KERNEL_RADIUS) x (TILE_SIZE + 2*KERNEL_RADIUS)
    __shared__ float s_input[TILE_SIZE + 2 * KERNEL_RADIUS][TILE_SIZE + 2 * KERNEL_RADIUS];

    // 输出tile的左上角在输入中的位置（考虑padding）
    int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // 输入tile的左上角
    int in_tile_x = blockIdx.x * TILE_SIZE - KERNEL_RADIUS;
    int in_tile_y = blockIdx.y * TILE_SIZE - KERNEL_RADIUS;

    // 协作加载：每个线程加载一个或多个元素到共享内存
    // 需要加载 (TILE_SIZE + 2*KERNEL_RADIUS)^2 个元素
    int total_threads = TILE_SIZE * TILE_SIZE;  // 256
    int total_elements = (TILE_SIZE + 2 * KERNEL_RADIUS) *
                         (TILE_SIZE + 2 * KERNEL_RADIUS);  // 22*22=484

    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    for (int i = tid; i < total_elements; i += total_threads) {
        int load_y = i / (TILE_SIZE + 2 * KERNEL_RADIUS);
        int load_x = i % (TILE_SIZE + 2 * KERNEL_RADIUS);

        int global_y = in_tile_y + load_y;
        int global_x = in_tile_x + load_x;

        if (global_y >= 0 && global_y < H && global_x >= 0 && global_x < W) {
            s_input[load_y][load_x] = input[global_y * W + global_x];
        } else {
            s_input[load_y][load_x] = 0.0f;
        }
    }
    __syncthreads();

    // 计算卷积
    if (out_row < H && out_col < W) {
        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                sum += s_input[threadIdx.y + ky][threadIdx.x + kx] *
                       kernel[ky * KERNEL_SIZE + kx];
            }
        }
        output[out_row * W + out_col] = sum;
    }
}
```

## 多通道卷积（深度学习场景）

```cpp
// 多通道卷积：输入有C_in个通道，输出C_out个通道
__global__ void conv2d_multi_channel(
    const float* input,     // [N, C_in, H, W]
    const float* weights,   // [C_out, C_in, KH, KW]
    const float* bias,      // [C_out]
    float* output,          // [N, C_out, H_out, W_out]
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int H_out, int W_out, int pad)
{
    // 每个线程计算一个输出像素
    int ox = threadIdx.x + blockIdx.x * blockDim.x;  // 输出x
    int oy = threadIdx.y + blockIdx.y * blockDim.y;  // 输出y

    // blockIdx.z编码了 (batch, output_channel)
    int oc = blockIdx.z % C_out;
    int n  = blockIdx.z / C_out;

    if (ox >= W_out || oy >= H_out || n >= N) return;

    float sum = bias ? bias[oc] : 0.0f;

    // 输入: input[n, :, :, :]
    // 权重: weights[oc, :, :, :]
    int input_offset = n * C_in * H * W;
    int weight_offset = oc * C_in * KH * KW;

    for (int c = 0; c < C_in; c++) {
        for (int ky = 0; ky < KH; ky++) {
            for (int kx = 0; kx < KW; kx++) {
                int iy = oy - pad + ky;
                int ix = ox - pad + kx;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                    float in_val = input[input_offset + c * H * W + iy * W + ix];
                    float w_val = weights[weight_offset + c * KH * KW + ky * KW + kx];
                    sum += in_val * w_val;
                }
            }
        }
    }

    output[n * C_out * H_out * W_out + oc * H_out * W_out + oy * W_out + ox] = sum;
}

void launch_conv_multi(float* d_input, float* d_weights, float* d_bias,
                        float* d_output, int N, int C_in, int H, int W,
                        int C_out, int KH, int KW, int pad) {
    int H_out = H + 2 * pad - KH + 1;
    int W_out = W + 2 * pad - KW + 1;

    dim3 threads(16, 16);
    dim3 blocks(
        (W_out + 15) / 16,
        (H_out + 15) / 16,
        N * C_out  // z维度编码batch和output channel
    );
    conv2d_multi_channel<<<blocks, threads>>>(
        d_input, d_weights, d_bias, d_output,
        N, C_in, H, W, C_out, KH, KW, H_out, W_out, pad);
}
```

## 1D卷积

```cpp
// 1D卷积 - 信号处理常用
__global__ void conv1d(const float* signal, const float* kernel,
                        float* output, int N, int K) {
    extern __shared__ float s_signal[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int radius = K / 2;

    // 加载数据到共享内存（包括halo区域）
    // 主区域
    if (idx < N) {
        s_signal[tid + radius] = signal[idx];
    }
    // 左侧halo
    if (tid < radius) {
        int halo_idx = blockIdx.x * blockDim.x - radius + tid;
        s_signal[tid] = (halo_idx >= 0) ? signal[halo_idx] : 0.0f;
    }
    // 右侧halo
    int right_tid = blockDim.x + tid;
    if (tid < radius && right_tid + radius < blockDim.x + 2 * radius) {
        int halo_idx = blockIdx.x * blockDim.x + blockDim.x + tid;
        s_signal[right_tid + radius] = (halo_idx < N) ? signal[halo_idx] : 0.0f;
    }
    __syncthreads();

    // 计算卷积
    if (idx < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += s_signal[tid + k] * kernel[k];
        }
        output[idx] = sum;
    }
}
```

## 深度可分离卷积

深度可分离卷积将标准卷积分解为depthwise + pointwise，大幅减少计算量：

```cpp
// Depthwise卷积：每个通道独立卷积
__global__ void conv2d_depthwise(
    const float* input,    // [N, C, H, W]
    const float* weights,  // [C, 1, KH, KW]
    float* output,         // [N, C, H, W]
    int N, int C, int H, int W,
    int KH, int KW, int pad)
{
    int ox = threadIdx.x + blockIdx.x * blockDim.x;
    int oy = threadIdx.y + blockIdx.y * blockDim.y;
    int oc = blockIdx.z % C;
    int n  = blockIdx.z / C;

    if (ox >= W || oy >= H || n >= N) return;

    float sum = 0.0f;
    int in_offset = n * C * H * W + oc * H * W;
    int w_offset = oc * KH * KW;

    for (int ky = 0; ky < KH; ky++) {
        for (int kx = 0; kx < KW; kx++) {
            int iy = oy - pad + ky;
            int ix = ox - pad + kx;
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                sum += input[in_offset + iy * W + ix] * weights[w_offset + ky * KW + kx];
            }
        }
    }

    output[n * C * H * W + oc * H * W + oy * W + ox] = sum;
}

// Pointwise卷积：1x1卷积，跨通道混合
__global__ void conv2d_pointwise(
    const float* input,    // [N, C_in, H, W]
    const float* weights,  // [C_out, C_in, 1, 1]
    float* output,         // [N, C_out, H, W]
    int N, int C_in, int H, int W, int C_out)
{
    int ox = threadIdx.x + blockIdx.x * blockDim.x;
    int oy = threadIdx.y + blockIdx.y * blockDim.y;
    int oc = blockIdx.z % C_out;
    int n  = blockIdx.z / C_out;

    if (ox >= W || oy >= H || n >= N) return;

    float sum = 0.0f;
    int out_idx = (n * C_out + oc) * H * W + oy * W + ox;

    for (int c = 0; c < C_in; c++) {
        int in_idx = (n * C_in + c) * H * W + oy * W + ox;
        sum += input[in_idx] * weights[oc * C_in + c];
    }

    output[out_idx] = sum;
}
```

## 小结

1. 常量内存适合存储小卷积核，同一warp内广播读取非常高效
2. Tiled卷积利用共享内存减少全局内存访问，类似矩阵乘法优化
3. 多通道卷积是深度学习的核心，需要正确组织batch/channel/空间维度
4. 深度可分离卷积分解为depthwise + pointwise，大幅减少计算量
5. 实际项目中优先使用cuDNN库，手写kernel用于学习和特殊场景
