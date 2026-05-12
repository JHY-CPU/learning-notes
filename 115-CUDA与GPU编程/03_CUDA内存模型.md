# CUDA内存模型

## GPU内存层次总览

GPU拥有多种不同类型的存储器，各有不同的容量、带宽和作用域：

```
┌──────────────────────────────────────────────────┐
│  寄存器 (Register)                                │
│  ~256 KB/SM, 延迟~1周期, 每线程私有                 │
├──────────────────────────────────────────────────┤
│  共享内存 (Shared Memory)                         │
│  48-164 KB/SM, 延迟~5周期, 每Block共享             │
├──────────────────────────────────────────────────┤
│  L1缓存 / L2缓存                                  │
│  L1: 128KB/SM, L2: 40MB(on A100)                 │
├──────────────────────────────────────────────────┤
│  常量内存 (Constant Memory)                       │
│  64 KB, 缓存, 只读, 所有线程共享                    │
├──────────────────────────────────────────────────┤
│  纹理内存 (Texture Memory)                        │
│  缓存, 只读, 空间局部性优化                         │
├──────────────────────────────────────────────────┤
│  全局内存 (Global Memory / Device Memory)         │
│  4-80 GB, 延迟~400周期, 所有线程可访问              │
├──────────────────────────────────────────────────┤
│  主机内存 (Host Memory)                           │
│  通过PCIe访问, 带宽受限                            │
└──────────────────────────────────────────────────┘
```

## 寄存器（Register）

寄存器是最快的存储器，每个线程拥有自己的私有寄存器。

```cpp
// 寄存器使用示例
__global__ void register_example(float* data, int n) {
    // 这些局部标量变量通常存储在寄存器中
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0.0f;        // 寄存器
    float result = 0.0f;      // 寄存器

    if (idx < n) {
        temp = data[idx];
        result = temp * temp + 1.0f;
        data[idx] = result;
    }
}

// 寄存器溢出：当线程使用过多寄存器时，多余的值溢出到本地内存（Local Memory）
// 导致性能大幅下降
__global__ void register_spill_example(float* data, int n) {
    // 使用过多局部变量可能引起寄存器溢出
    float v0, v1, v2, v3, v4, v5, v6, v7;
    float v8, v9, v10, v11, v12, v13, v14, v15;
    // ... 更多变量
    // 编译器可能将部分变量溢出到本地内存
}
```

**限制**：每个线程最多使用255个寄存器（计算能力7.0+）。使用过多寄存器会减少SM上的并发线程数。

## 共享内存（Shared Memory）

共享内存位于芯片上，同一线程块内所有线程可以访问。延迟接近寄存器，带宽远高于全局内存。

```cpp
__global__ void shared_memory_example(const float* input, float* output, int n) {
    // 声明共享内存数组
    __shared__ float s_data[256];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // 将数据从全局内存加载到共享内存
    if (idx < n) {
        s_data[tid] = input[idx];
    }

    // 同步：确保所有线程都完成了写入
    __syncthreads();

    // 使用共享内存中的数据（比全局内存快得多）
    if (idx < n) {
        // 相邻线程读取：利用共享内存的广播特性
        int neighbor = (tid + 1) % 256;
        output[idx] = s_data[tid] + s_data[neighbor];
    }
}
```

### 动态共享内存

大小可以在运行时指定：

```cpp
// Kernel声明：使用extern关键字声明动态共享内存
__global__ void dynamic_shared_mem(float* data, int n) {
    extern __shared__ float s_data[];
    // 大小在kernel启动时指定: kernel<<<grid, block, shared_bytes>>>();

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) s_data[tid] = data[idx];
    __syncthreads();

    // 使用s_data...
}

// 启动时指定共享内存大小
int main() {
    int blockSize = 256;
    size_t sharedMemSize = blockSize * sizeof(float);
    dynamic_shared_mem<<<100, blockSize, sharedMemSize>>>(d_data, N);
    return 0;
}
```

## 全局内存（Global Memory）

全局内存是GPU上最大容量的存储器，所有线程都可以访问，但延迟最高（~400个时钟周期）。

```cpp
// 全局内存分配与访问
__global__ void global_memory_rw(float* d_in, float* d_out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // 全局内存读取 - 延迟高，需要通过合并访问优化
        float val = d_in[idx];
        // 全局内存写入
        d_out[idx] = val * 2.0f;
    }
}

int main() {
    int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    float *d_data, *d_result;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, bytes);

    global_memory_rw<<<(N + 255) / 256, 256>>>(d_data, d_result, N);

    cudaFree(d_data);
    cudaFree(d_result);
    return 0;
}
```

## 常量内存（Constant Memory）

常量内存总大小64KB，对所有线程只读，适合存储不变的配置参数或小型查找表。

```cpp
// 声明常量内存（在所有kernel之外）
__constant__ float d_coefficients[16];
__constant__ int d_config[4];

__global__ void use_constant_memory(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = data[idx];
        // 读取常量内存 - 如果同一warp内线程读同一地址，广播非常高效
        for (int i = 0; i < 16; i++) {
            val += d_coefficients[i];
        }
        data[idx] = val + d_config[0];
    }
}

int main() {
    float h_coeffs[16] = {1.0f, 2.0f, 3.0f /* ... */};
    int h_config[4] = {100, 200, 300, 400};

    // 从主机拷贝到常量内存
    cudaMemcpyToSymbol(d_coefficients, h_coeffs, 16 * sizeof(float));
    cudaMemcpyToSymbol(d_config, h_config, 4 * sizeof(int));

    use_constant_memory<<<4096, 256>>>(d_data, N);
    return 0;
}
```

## 纹理内存（Texture Memory）

纹理内存针对2D空间局部性进行了优化，提供缓存和自动边界处理。

```cpp
// 现代CUDA使用Texture Object API
__global__ void texture_kernel(cudaTextureObject_t texObj, float* output,
                                int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        // tex2D自动处理缓存和边界
        float val = tex2D<float>(texObj, x, y);
        output[y * width + x] = val * 1.5f;
    }
}

void use_texture(float* d_input, int width, int height) {
    // 创建CUDA数组
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 拷贝数据到CUDA数组
    cudaMemcpy2DToArray(cuArray, 0, 0, d_input, width * sizeof(float),
                        width * sizeof(float), height, cudaMemcpyDeviceToDevice);

    // 创建纹理对象
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // 启动kernel
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    texture_kernel<<<grid, block>>>(texObj, d_output, width, height);

    // 清理
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}
```

## 内存性能对比

| 内存类型 | 容量 | 带宽 | 延迟 | 作用域 | 缓存 |
|----------|------|------|------|--------|------|
| 寄存器 | ~256KB/SM | 极高 | ~1周期 | 线程私有 | 无 |
| 共享内存 | 48-164KB/SM | 极高 | ~5周期 | Block内共享 | 无 |
| L1缓存 | ~128KB/SM | 高 | ~30周期 | SM内 | 有 |
| L2缓存 | ~40MB | 中高 | ~200周期 | GPU全局 | 有 |
| 常量内存 | 64KB | 中 | ~100周期 | 全局只读 | 有 |
| 纹理内存 | 取决于显存 | 中 | ~100周期 | 全局只读 | 有 |
| 全局内存 | 4-80GB | 1-2TB/s | ~400周期 | 全局可读写 | L2 |
| 主机内存 | 系统内存 | ~16GB/s(PCIe) | 极高 | 主机 | 无 |

## 本地内存（Local Memory）

本地内存虽然名字中有"local"，实际存储在全局内存中。当以下情况发生时，变量会存入本地内存：

1. 寄存器溢出
2. 数组下标是变量的数组
3. 大型结构体

```cpp
__global__ void local_memory_example(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 这个数组会存储在本地内存中（下标是变量）
    float local_array[64];  // 存在本地内存，不是共享内存！

    if (idx < n) {
        for (int i = 0; i < 64; i++) {
            local_array[i] = data[idx] * i;
        }
        data[idx] = local_array[idx % 64];
    }
}
```

## 小结

1. 内存层次是CUDA性能优化的核心：寄存器 > 共享内存 > 全局内存
2. 减少全局内存访问次数，利用共享内存做数据复用
3. 常量内存适合只读小数据集，同一warp内同地址读取有广播加速
4. 理解每种内存的特性才能做出正确的优化决策
