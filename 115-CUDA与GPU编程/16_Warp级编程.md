# Warp级编程

## Warp Shuffle操作

Warp shuffle允许同一warp内的线程直接交换寄存器中的值，不经过共享内存或全局内存，延迟极低。

### __shfl_sync 系列函数

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// __shfl_sync: 从指定线程获取值
__global__ void shfl_example(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];
    int lane = threadIdx.x % 32;

    // 从lane号线程获取值
    // mask: 0xFFFFFFFF表示所有32个线程参与
    float from_lane_0 = __shfl_sync(0xFFFFFFFF, val, 0);
    // 所有线程都获得lane 0的值

    // 获取相邻线程的值
    float from_prev = __shfl_sync(0xFFFFFFFF, val, (lane - 1 + 32) % 32);

    data[idx] = val + from_lane_0 + from_prev;
}

// __shfl_up_sync: 从编号较小的线程获取值（向上取）
__global__ void shfl_up_example(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];
    int lane = threadIdx.x % 32;

    // lane=i的线程从lane=i-offset获取值
    // lane<offset的线程保持自己的值
    float shifted = __shfl_up_sync(0xFFFFFFFF, val, 1);
    // lane 0 保持自己的值
    // lane 1 获取 lane 0 的值
    // lane 2 获取 lane 1 的值
    // ...

    data[idx] = val + shifted;
}

// __shfl_down_sync: 从编号较大的线程获取值
__global__ void shfl_down_example(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];

    // lane=i的线程从lane=i+offset获取值
    float shifted = __shfl_down_sync(0xFFFFFFFF, val, 1);

    data[idx] = val + shifted;
}

// __shfl_xor_sync: 通过XOR索引交换
__global__ void shfl_xor_example(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];
    int lane = threadIdx.x % 32;

    // 与lane XOR 1的线程交换值
    float swapped = __shfl_xor_sync(0xFFFFFFFF, val, 1);
    // lane 0 ↔ lane 1, lane 2 ↔ lane 3, ...

    data[idx] = val + swapped;
}
```

## 使用Warp Shuffle进行归约

Warp shuffle归约比共享内存版本更快：

```cpp
// Warp内归约：使用__shfl_down_sync
__device__ float warp_reduce_sum(float val) {
    // 16 → 8 → 4 → 2 → 1
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;  // 只有lane 0有正确结果
}

// Warp内求最大值
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// 完整的Block归约
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_results[32];  // 最多32个warp

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // warp内归约
    val = warp_reduce_sum(val);

    // 每个warp的第一个线程写入共享内存
    if (lane == 0) warp_results[warp_id] = val;
    __syncthreads();

    // 第一个warp归约所有warp的结果
    if (warp_id == 0) {
        // 读取warp结果（如果warp数不足32，多余位置为0）
        val = (lane < (blockDim.x + 31) / 32) ? warp_results[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;  // 只有lane 0（threadIdx.x == 0）有Block的总和
}

// 使用示例
__global__ void sum_kernel(const float* input, float* output, int N) {
    float sum = 0.0f;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        sum += input[i];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}
```

## Warp Vote操作

Warp vote操作允许检查warp内所有线程的谓词条件：

```cpp
// CUDA 9.0+的同步版本
__global__ void vote_all_example(const float* data, bool* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];
    bool predicate = (val > 0.0f);

    // 检查warp内所有线程的谓词是否都为true
    bool all_positive = __all_sync(0xFFFFFFFF, predicate);
    // 如果warp内所有线程的val > 0，则all_positive为true

    if (threadIdx.x % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = all_positive;
    }
}

__global__ void vote_any_example(const float* data, bool* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];
    bool predicate = (val > 0.0f);

    // 检查warp内是否有线程的谓词为true
    bool any_positive = __any_sync(0xFFFFFFFF, predicate);

    if (threadIdx.x % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32] = any_positive;
    }
}

// ballot返回一个32位掩码，每个位对应一个线程的谓词结果
__global__ void vote_ballot_example(const float* data, unsigned int* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];
    bool predicate = (val > 0.0f);

    // 返回32位掩码：bit i = 线程i的谓词
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, predicate);

    if (threadIdx.x % 32 == 0) {
        result[blockIdx.x * (blockDim.x / 32) + ballot);  // 可以分析warp内的模式
    }
}
```

## Cooperative Groups（协作组）

Cooperative Groups（CUDA 9.0+）提供了更灵活的线程分组抽象：

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// 基本cooperative group使用
__global__ void coop_groups_example(float* data, int N) {
    // 获取当前线程块
    cg::thread_block block = cg::this_thread_block();

    // 获取当前warp
    cg::tiled_partition<32> warp = cg::tiled_partition<32>(block);

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];

    // warp内归约
    for (int i = warp.size() / 2; i > 0; i >>= 1) {
        val += cg::reduce(warp, cg::shift_group_right(warp, val, i), cg::plus<float>());
    }

    // 或者使用更简洁的方式
    float sum = cg::reduce(warp, val, cg::plus<float>());

    if (warp.thread_rank() == 0) {
        data[idx] = sum;
    }
}

// 自定义分组
__global__ void custom_groups(float* data, int N) {
    cg::thread_block block = cg::this_thread_block();

    // 将block分成4个组
    cg::thread_block_tile<64> group = cg::tiled_partition<64>(block);

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    // 组内同步
    group.sync();

    // 组内归约
    float val = data[idx];
    float group_sum = cg::reduce(group, val, cg::plus<float>());

    if (group.thread_rank() == 0) {
        data[idx] = group_sum;
    }
}

// 多级分组
__global__ void multi_level_groups(float* data, int N) {
    cg::thread_block block = cg::this_thread_block();

    // 第一级：分成8个32线程的warp
    auto warps = cg::tiled_partition<32>(block);

    // 第二级：在warp内分成4个8线程的组
    auto sub_groups = cg::tiled_partition<8>(warps);

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    float val = data[idx];

    // 子组内归约
    float sub_sum = cg::reduce(sub_groups, val, cg::plus<float>());

    // warp内归约子组结果
    float warp_sum = cg::reduce(warps, sub_sum, cg::plus<float>());

    if (block.thread_rank() == 0) {
        atomicAdd(data, warp_sum);
    }
}
```

## 跨Block协作（Cooperative Kernel）

```cpp
// 需要计算能力6.0+和cudaLaunchCooperativeKernel
__global__ void cooperative_kernel(float* data, int N) {
    cg::grid_group grid = cg::this_grid();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 网格级同步：所有Block同步
    // 注意：这是全局同步，成本很高
    grid.sync();

    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

void launch_cooperative() {
    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 使用cooperative launch
    void* args[] = {&d_data, &N};
    cudaLaunchCooperativeKernel((void*)cooperative_kernel,
                                 blocks, threads, args);

    cudaFree(d_data);
}
```

## Warp级数据排序

```cpp
// Warp内的bitonic sort
__device__ void warp_bitonic_sort(float* data) {
    int lane = threadIdx.x % 32;

    for (int k = 2; k <= 32; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int partner = lane ^ j;
            float my_val = data[lane];
            float partner_val = __shfl_sync(0xFFFFFFFF, my_val, partner);

            bool ascending = ((lane & k) == 0);

            if (ascending) {
                if (my_val > partner_val) {
                    data[lane] = partner_val;
                }
            } else {
                if (my_val < partner_val) {
                    data[lane] = partner_val;
                }
            }
        }
    }
}
```

## 小结

1. Warp shuffle（__shfl_sync系列）提供线程间寄存器直接通信，延迟极低
2. Warp vote（__all_sync, __any_sync, __ballot_sync）检查warp内谓词状态
3. Cooperative Groups提供灵活的线程分组抽象，支持多级分组
4. Warp级归约比共享内存归约更高效
5. Cooperative launch支持网格级同步，但成本很高，需谨慎使用
