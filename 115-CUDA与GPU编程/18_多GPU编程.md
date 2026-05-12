# 多GPU编程

## 多GPU架构概述

现代深度学习训练和HPC应用需要利用多块GPU的计算能力：

```
多GPU系统拓扑：

方案1：PCIe连接
GPU0 ←→ PCIe ←→ CPU ←→ PCIe ←→ GPU1
带宽受限（~16 GB/s per direction per PCIe 4.0 x16）

方案2：NVLink连接（NVIDIA高端GPU）
GPU0 ←→ NVLink (600 GB/s) ←→ GPU1
GPU2 ←→ NVLink ←→ GPU3
       ↕ NVSwitch ↕
全互联带宽可达900 GB/s（H100 + NVSwitch）

方案3：多节点
Node0: GPU0, GPU1, GPU2, GPU3
Node1: GPU4, GPU5, GPU6, GPU7
       ↕ InfiniBand (400 Gb/s) ↕
```

## 设备管理

```cpp
#include <cstdio>
#include <cuda_runtime.h>

void enumerate_devices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multi-processor Count: %d\n", prop.multiProcessorCount);

        // 检查P2P支持
        for (int j = 0; j < deviceCount; j++) {
            if (i != j) {
                int canAccess;
                cudaDeviceCanAccessPeer(&canAccess, i, j);
                printf("  P2P to Device %d: %s\n", j,
                       canAccess ? "Supported" : "Not Supported");
            }
        }
    }
}

void set_device(int dev) {
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Using Device %d: %s\n", dev, prop.name);
}
```

## P2P（Peer-to-Peer）访问

P2P允许GPU直接访问其他GPU的显存，无需经过CPU和系统内存：

```cpp
void enable_p2p(int src_dev, int dst_dev) {
    cudaSetDevice(src_dev);

    // 启用P2P访问
    cudaError_t err = cudaDeviceEnablePeerAccess(dst_dev, 0);
    if (err == cudaSuccess) {
        printf("P2P enabled: Device %d → Device %d\n", src_dev, dst_dev);
    } else if (err == cudaErrorPeerAccessAlreadyEnabled) {
        printf("P2P already enabled: Device %d → Device %d\n", src_dev, dst_dev);
    } else {
        printf("Failed to enable P2P: %s\n", cudaGetErrorString(err));
    }
}

void disable_p2p(int src_dev, int dst_dev) {
    cudaSetDevice(src_dev);
    cudaDeviceDisablePeerAccess(dst_dev);
}

// P2P内存拷贝
void p2p_copy_example() {
    int N = 1024 * 1024;
    float *d_data0, *d_data1;

    // 在GPU 0上分配
    cudaSetDevice(0);
    cudaMalloc(&d_data0, N * sizeof(float));

    // 在GPU 1上分配
    cudaSetDevice(1);
    cudaMalloc(&d_data1, N * sizeof(float));

    // 启用P2P
    enable_p2p(0, 1);

    // GPU 0 → GPU 1 的P2P拷贝（不经CPU）
    cudaSetDevice(0);
    cudaMemcpyPeer(d_data1, 1, d_data0, 0, N * sizeof(float));

    // 异步P2P拷贝
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyPeerAsync(d_data1, 1, d_data0, 0, N * sizeof(float), stream);

    cudaStreamSynchronize(stream);

    disable_p2p(0, 1);
    cudaStreamDestroy(stream);
    cudaSetDevice(0); cudaFree(d_data0);
    cudaSetDevice(1); cudaFree(d_data1);
}
```

## 多GPU内存分配与拷贝

```cpp
void multi_gpu_memory_example() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // 每个GPU上分配内存
    float* d_data[8];  // 最多8个GPU
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_data[g], bytes);
    }

    // 主机数据
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    // 分发数据到各GPU（每个GPU处理N/numGPUs个元素）
    int chunk = N / numGPUs;
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMemcpy(d_data[g], h_data + g * chunk, chunk * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    // 各GPU独立处理
    // kernel<<<...>>>(d_data[g], chunk);

    // 收集结果
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMemcpy(h_data + g * chunk, d_data[g], chunk * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    // 清理
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaFree(d_data[g]);
    }
    delete[] h_data;
}
```

## NCCL库

NCCL（NVIDIA Collective Communications Library）是多GPU/多节点通信的标准库，提供高效的集合通信操作：

```cpp
#include <nccl.h>
#include <cstdio>
#include <cuda_runtime.h>

#define NCCL_CHECK(call)                                                    \
    do {                                                                    \
        ncclResult_t res = call;                                            \
        if (res != ncclSuccess) {                                           \
            fprintf(stderr, "NCCL Error: %s\n", ncclGetErrorString(res));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

void nccl_allreduce_example() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    if (numGPUs < 2) {
        printf("Need at least 2 GPUs\n");
        return;
    }

    int N = 1024 * 1024;
    size_t bytes = N * sizeof(float);

    // 每个GPU上的数据
    float* d_data[8];
    cudaStream_t streams[8];

    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_data[g], bytes);
        cudaStreamCreate(&streams[g]);

        // 初始化为GPU编号
        cudaMemset(d_data[g], g, bytes);
    }

    // 获取所有GPU的设备ID
    int devs[8];
    for (int g = 0; g < numGPUs; g++) devs[g] = g;

    // 创建NCCL通信器
    ncclComm_t comms[8];
    NCCL_CHECK(ncclCommInitAll(comms, numGPUs, devs));

    // AllReduce: 所有GPU上d_data的值求和，结果存回每个GPU
    // 每个GPU的结果 = GPU0数据 + GPU1数据 + ... + GPU(n-1)数据
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        NCCL_CHECK(ncclAllReduce(
            d_data[g],          // 发送缓冲区
            d_data[g],          // 接收缓冲区（原地操作）
            N,                  // 元素数量
            ncclFloat,          // 数据类型
            ncclSum,            // 操作类型
            comms[g],           // NCCL通信器
            streams[g]          // CUDA流
        ));
    }

    // 同步
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaStreamSynchronize(streams[g]);
    }

    // 验证（每个GPU的结果应该相同）
    float* h_result = new float[N];
    cudaSetDevice(0);
    cudaMemcpy(h_result, d_data[0], bytes, cudaMemcpyDeviceToHost);
    printf("After AllReduce: result[0] = %f (expected: %f)\n",
           h_result[0], (float)(numGPUs * (numGPUs - 1) / 2));

    // 清理
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        ncclCommDestroy(comms[g]);
        cudaStreamDestroy(streams[g]);
        cudaFree(d_data[g]);
    }
    delete[] h_result;
}
```

## NCCL集合通信操作

```cpp
// Broadcast: 从一个GPU广播数据到所有GPU
void nccl_broadcast_example() {
    ncclComm_t comms[8];
    float* d_data[8];
    cudaStream_t streams[8];
    int numGPUs = 2;
    int N = 1024;

    // 初始化...
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_data[g], N * sizeof(float));
        cudaStreamCreate(&streams[g]);
    }

    // 从GPU 0广播到所有GPU
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        ncclBroadcast(d_data[g], d_data[g], N, ncclFloat,
                      0, comms[g], streams[g]);  // root=0
    }
}

// AllGather: 每个GPU收集所有GPU的数据
void nccl_allgather_example() {
    int numGPUs = 2;
    int N = 1024;
    int totalN = N * numGPUs;

    float* d_send[8];   // 每个GPU的发送缓冲区（N个元素）
    float* d_recv[8];   // 每个GPU的接收缓冲区（N*numGPUs个元素）

    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_send[g], N * sizeof(float));
        cudaMalloc(&d_recv[g], totalN * sizeof(float));
    }

    // AllGather后，d_recv[g]包含所有GPU的d_send数据
    // d_recv[g] = [GPU0的数据 | GPU1的数据 | ... | GPU(n-1)的数据]
    // for (int g = 0; g < numGPUs; g++) {
    //     ncclAllGather(d_send[g], d_recv[g], N, ncclFloat, comms[g], stream);
    // }
}

// Reduce-Scatter: 所有GPU数据求和，然后分散到各GPU
void nccl_reduce_scatter_example() {
    int numGPUs = 2;
    int N = 1024;

    float* d_send[8];   // 每个GPU发送N*numGPUs个元素
    float* d_recv[8];   // 每个GPU接收N个元素

    // Reduce-Scatter后，d_recv[g]包含所有GPU对应第g块数据的求和
    // for (int g = 0; g < numGPUs; g++) {
    //     ncclReduceScatter(d_send[g], d_recv[g], N, ncclFloat, ncclSum,
    //                       comms[g], stream);
    // }
}
```

## 多GPU数据并行训练

```cpp
// 完整的多GPU数据并行训练示例
__global__ void forward_kernel(float* input, float* weights, float* output,
                                int batch, int in_dim, int out_dim) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int out_idx = idx % out_dim;
    int batch_idx = idx / out_dim;

    if (batch_idx < batch) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += input[batch_idx * in_dim + i] * weights[out_idx * in_dim + i];
        }
        output[batch_idx * out_dim + out_idx] = sum;
    }
}

__global__ void update_weights(float* weights, const float* gradients,
                                float lr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        weights[idx] -= lr * gradients[idx];
    }
}

void multi_gpu_training() {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    int batch_size = 256;
    int in_dim = 1024;
    int out_dim = 512;
    int batch_per_gpu = batch_size / numGPUs;
    int weight_size = in_dim * out_dim;

    // 初始化NCCL
    ncclComm_t comms[8];
    int devs[8];
    for (int i = 0; i < numGPUs; i++) devs[i] = i;
    ncclCommInitAll(comms, numGPUs, devs);

    // 每个GPU上的权重和梯度
    float* d_weights[8];
    float* d_grads[8];
    cudaStream_t streams[8];

    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        cudaMalloc(&d_weights[g], weight_size * sizeof(float));
        cudaMalloc(&d_grads[g], weight_size * sizeof(float));
        cudaStreamCreate(&streams[g]);
        // 初始化权重
    }

    float lr = 0.001f;
    int threads = 256;

    // 训练循环
    for (int epoch = 0; epoch < 10; epoch++) {
        // 每个GPU独立前向和反向
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            int output_size = batch_per_gpu * out_dim;
            forward_kernel<<<(output_size + threads - 1) / threads, threads,
                            0, streams[g]>>>(
                /* input */, d_weights[g], /* output */,
                batch_per_gpu, in_dim, out_dim);
            // backward_kernel计算d_grads[g]
        }

        // 同步前向/反向完成
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }

        // AllReduce梯度：所有GPU的梯度求和
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            ncclAllReduce(d_grads[g], d_grads[g], weight_size,
                          ncclFloat, ncclSum, comms[g], streams[g]);
        }

        // 同步AllReduce
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }

        // 每个GPU更新权重
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            update_weights<<<(weight_size + threads - 1) / threads, threads,
                            0, streams[g]>>>(
                d_weights[g], d_grads[g], lr / numGPUs, weight_size);
        }

        // 同步更新
        for (int g = 0; g < numGPUs; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(streams[g]);
        }
    }

    // 清理
    for (int g = 0; g < numGPUs; g++) {
        cudaSetDevice(g);
        ncclCommDestroy(comms[g]);
        cudaStreamDestroy(streams[g]);
        cudaFree(d_weights[g]);
        cudaFree(d_grads[g]);
    }
}
```

## 多节点通信

```cpp
// 多节点使用MPI + NCCL
#include <mpi.h>

void multi_node_nccl() {
    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 每个进程管理1个GPU
    cudaSetDevice(rank % 4);  // 假设每节点4个GPU

    // 基于MPI初始化NCCL通信器ID
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 初始化NCCL通信器
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);

    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // AllReduce跨所有节点的所有GPU
    ncclAllReduce(d_data, d_data, N, ncclFloat, ncclSum, comm, stream);
    cudaStreamSynchronize(stream);

    printf("Node %d: AllReduce completed\n", rank);

    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    MPI_Finalize();
}
```

## 小结

1. 多GPU编程需要正确管理设备上下文，使用cudaSetDevice切换当前设备
2. P2P允许GPU间直接访问显存，NVLink提供比PCIe更高的带宽
3. NCCL提供高效的集合通信操作：AllReduce、Broadcast、AllGather等
4. 数据并行训练的核心流程：分发数据、独立计算、AllReduce梯度、更新权重
5. 多节点训练结合MPI和NCCL，实现跨节点的高效通信
