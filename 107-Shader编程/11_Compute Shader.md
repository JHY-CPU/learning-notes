# Compute Shader

## 1. 核心理论

Compute Shader是运行在GPU上的通用计算程序，**不参与传统渲染管线**，允许开发者直接利用GPU的大规模并行能力执行任意计算任务。它是现代GPGPU（General-Purpose Computing on GPU）的主要接口。

### 1.1 GPGPU概念

```
传统渲染管线 vs Compute Shader：

传统管线：
  CPU → Vertex Shader → 光栅化 → Fragment Shader → 帧缓冲
  固定流程，输入/输出受限于几何和像素

Compute Shader：
  CPU → Dispatch → [N个线程并行执行] → 任意输出
  完全自由的计算流程，输入/输出可以是任意Buffer

核心优势：
  - 百万级线程并行执行（现代GPU有数千个核心）
  - 线程间可通过共享内存通信
  - 可读写任意Buffer，不受帧缓冲限制
  - 可与渲染管线无缝交互（Compute结果供Fragment Shader使用）
```

### 1.2 线程组织层次

```
线程组织结构（以HLSL为例）：

Dispatch(groupCountX, groupCountY, groupCountZ)
  → 启动 groupCountX × groupCountY × groupCountZ 个线程组

每个线程组 [numthreads(x, y, z)]：
  → 包含 x × y × z 个线程

总线程数 = groupCount × numthreads

例如：
  [numthreads(8, 8, 1)]    → 每组64个线程
  Dispatch(240, 135, 1)    → 240×135 = 32400个线程组
  总线程 = 32400 × 64 = 2,073,600个线程

线程ID系统：
  SV_GroupID            → 线程组在Dispatch网格中的索引
  SV_GroupThreadID      → 线程在组内的索引（0-based）
  SV_GroupIndex         → 线程在组内的线性索引
  SV_DispatchThreadID   → 线程的全局索引（= GroupID × numthreads + GroupThreadID）
```

```
线程组织示意图（2D）：

Dispatch(3, 2, 1) → 6个线程组
numthreads(4, 4, 1) → 每组16个线程

+-----+-----+-----+
| G00 | G10 | G20 |   ← GroupID.y = 0
| 4×4 | 4×4 | 4×4 |
+-----+-----+-----+
| G01 | G11 | G21 |   ← GroupID.y = 1
| 4×4 | 4×4 | 4×4 |
+-----+-----+-----+

全局线程 ID (DispatchThreadID):
  (0,0) (1,0) (2,0) (3,0) | (4,0) (5,0) ... | (8,0) ...
  (0,1) (1,1) (2,1) (3,1) | (4,1) (5,1) ... | (8,1) ...
  ...
```

## 2. HLSL Compute Shader

### 2.1 基础语法

```hlsl
// 编译指令
#pragma kernel KernelName

// 线程组大小声明
[numthreads(8, 8, 1)]
void CSMain(
    uint3 groupID : SV_GroupID,                // 线程组索引
    uint3 groupThreadID : SV_GroupThreadID,    // 组内线程索引
    uint3 dispatchThreadID : SV_DispatchThreadID, // 全局线程索引
    uint groupIndex : SV_GroupIndex            // 组内线性索引
)
{
    // dispatchThreadID.xy 通常是处理2D数据的关键
    uint2 pixel = dispatchThreadID.xy;

    // 边界检查（总线程数可能大于数据尺寸）
    if (pixel.x >= screenWidth || pixel.y >= screenHeight)
        return;

    // 执行计算...
}
```

### 2.2 GLSL Compute Shader

```glsl
#version 460 core

// 工作组大小声明
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 输入输出图像
layout(binding = 0, rgba32f) uniform image2D inputImage;
layout(binding = 1, rgba32f) uniform image2D outputImage;

// 常量
uniform int screenWidth;
uniform int screenHeight;

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    if (pixel.x >= screenWidth || pixel.y >= screenHeight)
        return;

    vec4 color = imageLoad(inputImage, pixel);
    // 处理...
    imageStore(outputImage, pixel, color * 2.0);
}
```

## 3. 数据缓冲区

### 3.1 StructuredBuffer

```hlsl
// 结构化缓冲区 — 最常用的Compute Shader数据结构
struct Particle
{
    float3 position;
    float3 velocity;
    float age;
    float maxLife;
    float size;
    float mass;
    uint type;      // 0=普通, 1=爆炸, 2=拖尾
};

// 只读结构化缓冲区
StructuredBuffer<Particle> inputBuffer : register(t0);

// 可读写结构化缓冲区
RWStructuredBuffer<Particle> outputBuffer : register(u0);

// Append/Consume缓冲区
AppendStructuredBuffer<Particle> appendBuffer : register(u1);
ConsumeStructuredBuffer<Particle> consumeBuffer : register(u2);
```

### 3.2 ByteAddressBuffer

```hlsl
// 字节寻址缓冲区 — 按字节偏移访问，最灵活
ByteAddressBuffer rawData : register(t0);
RWByteAddressBuffer rwData : register(u0);

void main(uint3 id : SV_DispatchThreadID)
{
    // 按4字节偏移读取
    uint value = rawData.Load(id.x * 4);
    // 读取float3（12字节）
    float3 pos = asfloat(rawData.Load3(id.x * 16));

    // 写入
    rwData.Store(id.x * 4, value + 1);
    rwData.Store3(id.x * 16, asuint(pos + float3(1, 0, 0)));
}
```

### 3.3 共享内存（Shared Memory / LDS）

```hlsl
// 共享内存 — 组内所有线程可访问的高速内存（比全局显存快100倍）
// 大小通常限制为16KB-32KB（因硬件而异）

groupshared float sharedData[256];
groupshared float3 sharedPositions[64];

[numthreads(256, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    // 每个线程加载一个数据到共享内存
    sharedData[groupThreadID.x] = globalBuffer[id.x];

    // 同步屏障 — 确保组内所有线程都完成了写入
    GroupMemoryBarrierWithGroupSync();

    // 现在可以安全读取其他线程写入的数据
    // 例如：归约求和（Reduction）
    for (uint stride = 128; stride > 0; stride >>= 1)
    {
        if (groupThreadID.x < stride)
        {
            sharedData[groupThreadID.x] += sharedData[groupThreadID.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // 第一个线程写入最终结果
    if (groupThreadID.x == 0)
    {
        result[groupID.x] = sharedData[0];
    }
}
```

## 4. 粒子系统

### 4.1 GPU粒子更新

```hlsl
// 完整的GPU粒子更新Compute Shader

struct Particle
{
    float3 position;
    float pad1;
    float3 velocity;
    float pad2;
    float4 color;
    float age;
    float maxLife;
    float size;
    float rotation;
};

RWStructuredBuffer<Particle> Particles : register(u0);
AppendStructuredBuffer<uint> DeadList : register(u1);

cbuffer ParticleParams : register(b0)
{
    float deltaTime;
    float3 gravity;
    float3 emitterPosition;
    float emitterRadius;
    float emitRate;
    float3 windForce;
    uint maxParticles;
    uint frameIndex;
};

// 伪随机数生成
float Random(float seed)
{
    return frac(sin(seed * 12.9898 + 78.233) * 43758.5453);
}

float3 Random3(float seed)
{
    return float3(
        Random(seed),
        Random(seed + 1.0),
        Random(seed + 2.0)
    ) * 2.0 - 1.0;
}

[numthreads(256, 1, 1)]
void UpdateParticles(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= maxParticles) return;

    Particle p = Particles[id.x];

    // 跳过已死粒子
    if (p.age < 0)
    {
        // 尝试重生
        if (id.x < uint(emitRate * deltaTime))
        {
            float3 offset = Random3(float(id.x + frameIndex)) * emitterRadius;
            p.position = emitterPosition + offset;
            p.velocity = Random3(float(id.x + frameIndex + 100)) * 2.0 + float3(0, 3, 0);
            p.age = 0;
            p.maxLife = 2.0 + Random(float(id.x)) * 3.0;
            p.size = 0.1 + Random(float(id.x + 50)) * 0.2;
            p.color = float4(1, 0.8, 0.3, 1);
            p.rotation = Random(float(id.x + 200)) * 6.28;
        }
        else
        {
            Particles[id.x] = p;
            return;
        }
    }

    // 物理更新
    float3 totalForce = gravity + windForce;
    p.velocity += totalForce * deltaTime;
    p.position += p.velocity * deltaTime;

    // 生命周期
    p.age += deltaTime;

    // 颜色衰减
    float lifeRatio = p.age / p.maxLife;
    p.color.a = 1.0 - lifeRatio;  // 渐隐
    p.size *= (1.0 - deltaTime * 0.1); // 渐小

    // 死亡检测
    if (p.age >= p.maxLife)
    {
        p.age = -1; // 标记为死亡
        DeadList.Append(id.x);
    }

    Particles[id.x] = p;
}
```

### 4.2 C#端Dispatch

```csharp
// Unity C#端调用Compute Shader
public class GPUParticleSystem : MonoBehaviour
{
    public ComputeShader particleCS;
    public int maxParticles = 100000;

    private ComputeBuffer particleBuffer;
    private int updateKernel;

    void Start()
    {
        // 创建粒子缓冲区
        int stride = sizeof(float) * 16; // Particle结构体大小
        particleBuffer = new ComputeBuffer(maxParticles, stride);

        // 获取Kernel索引
        updateKernel = particleCS.FindKernel("UpdateParticles");
    }

    void Update()
    {
        // 设置参数
        particleCS.SetFloat("deltaTime", Time.deltaTime);
        particleCS.SetVector("gravity", new Vector3(0, -9.8f, 0));
        particleCS.SetInt("maxParticles", maxParticles);

        // 绑定缓冲区
        particleCS.SetBuffer(updateKernel, "Particles", particleBuffer);

        // Dispatch：计算需要的线程组数
        int threadGroups = Mathf.CeilToInt(maxParticles / 256.0f);
        particleCS.Dispatch(updateKernel, threadGroups, 1, 1);
    }

    void OnDestroy()
    {
        particleBuffer?.Release();
    }
}
```

## 5. GPU剔除（GPU Culling）

用Compute Shader在GPU端进行视锥剔除和遮挡剔除，替代CPU端逐物体判断。

```hlsl
// GPU视锥剔除
struct ObjectData
{
    float3 boundsMin;
    float3 boundsMax;
    uint drawIndex;
    uint visible;   // 输出：1=可见, 0=剔除
};

RWStructuredBuffer<ObjectData> Objects : register(u0);

cbuffer CameraData : register(b0)
{
    float4 frustumPlanes[6]; // 视锥体6个面 (normal.xyz, distance)
    uint objectCount;
};

// 点到平面距离
float PointToPlaneDistance(float4 plane, float3 point)
{
    return dot(plane.xyz, point) + plane.w;
}

// AABB与视锥体相交测试
bool IntersectFrustum(float3 boundsMin, float3 boundsMax)
{
    for (int i = 0; i < 6; i++)
    {
        // 取AABB在平面法线方向上的最远点
        float3 positiveVertex = float3(
            frustumPlanes[i].x > 0 ? boundsMax.x : boundsMin.x,
            frustumPlanes[i].y > 0 ? boundsMax.y : boundsMin.y,
            frustumPlanes[i].z > 0 ? boundsMax.z : boundsMin.z
        );

        if (PointToPlaneDistance(frustumPlanes[i], positiveVertex) < 0)
            return false; // 完全在平面外侧
    }
    return true;
}

[numthreads(64, 1, 1)]
void CullObjects(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= objectCount) return;

    ObjectData obj = Objects[id.x];
    obj.visible = IntersectFrustum(obj.boundsMin, obj.boundsMax) ? 1 : 0;
    Objects[id.x] = obj;
}
```

## 6. 图像处理

```hlsl
// 高斯模糊（使用共享内存优化）
Texture2D<float4> inputTexture : register(t0);
RWTexture2D<float4> outputTexture : register(u0);

// 共享内存存储一行像素
groupshared float4 sharedRow[264]; // 256 + 8 (halo)

[numthreads(256, 1, 1)]
void BlurHorizontal(uint3 id : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    // 加载数据到共享内存（包含边界halo）
    int loadIdx = int(id.x) - 4;
    if (loadIdx >= 0)
        sharedRow[groupThreadID.x] = inputTexture[int2(loadIdx, id.y)];
    else
        sharedRow[groupThreadID.x] = float4(0, 0, 0, 0);

    // 右侧halo
    int rightLoad = int(id.x) + 252;
    if (groupThreadID.x >= 252 && rightLoad < int(textureWidth))
        sharedRow[groupThreadID.x + 8] = inputTexture[int2(rightLoad, id.y)];

    GroupMemoryBarrierWithGroupSync();

    // 9-tap高斯模糊
    float weights[9] = {0.016, 0.054, 0.122, 0.195, 0.227, 0.195, 0.122, 0.054, 0.016};
    float4 result = 0;
    for (int i = -4; i <= 4; i++)
    {
        result += sharedRow[groupThreadID.x + 4 + i] * weights[i + 4];
    }

    outputTexture[id.xy] = result;
}
```

## 7. 线程组大小选择

```
线程组大小对性能的影响：

最优大小通常是硬件Warp/Wavefront的整数倍：
  NVIDIA GPU: 32线程/Warp → 64, 128, 256, 512
  AMD GPU: 64线程/Wavefront → 64, 128, 256, 512
  移动端Mali: 16线程 → 16, 32, 64, 128
  移动端Adreno: 64线程 → 64, 128, 256

经验法则：
  - 通用计算：[numthreads(256, 1, 1)] 最常用
  - 2D图像处理：[numthreads(8, 8, 1)] 或 [numthreads(16, 16, 1)]
  - 使用共享内存时：线程数应等于共享内存数组大小
  - 总线程数必须能被Warp/Wavefront大小整除
```

## 8. 同步与原子操作

```hlsl
// 内存屏障
GroupMemoryBarrierWithGroupSync();     // 组内共享内存屏障 + 同步
DeviceMemoryBarrier();                  // 设备内存屏障（全局）
AllMemoryBarrier();                     // 所有内存屏障

// 原子操作
uint originalValue;
InterlockedAdd(counter, 1, originalValue);  // 原子加，返回旧值
InterlockedMin(minValue, newValue);
InterlockedMax(maxValue, newValue);
InterlockedAnd/Or/Xor(mask, value);
InterlockedExchange(target, value, originalValue);
InterlockedCompareExchange(target, compare, value, originalValue); // CAS

// 典型用例：原子计数器
groupshared uint groupCounter;
void AtomicExample()
{
    // 初始化（由第一个线程执行）
    if (groupThreadID.x == 0 && groupThreadID.y == 0)
        groupCounter = 0;
    GroupMemoryBarrierWithGroupSync();

    // 每个线程原子递增
    uint localIndex;
    InterlockedAdd(groupCounter, 1, localIndex);
    GroupMemoryBarrierWithGroupSync();

    // localIndex 是当前线程在组内的唯一索引
}
```

## 9. 性能提示

- 线程组大小应是Warp大小（NVIDIA=32, AMD=64）的整数倍
- 共享内存（LDS）比全局显存快100倍以上，尽量使用
- 避免共享内存的Bank Conflict（交错访问不同Bank）
- GPU-CPU数据回读有巨大延迟，尽量让数据留在GPU
- 多个Compute Shader之间可以用UAV屏障同步
- Dispatch次数尽量少，合并多个操作到一个Dispatch中

## 10. 常见问题与调试

**问题1：结果错误但无报错**
- 检查边界条件：线程数可能大于数据尺寸
- 检查共享内存同步：所有读取共享内存的地方之前必须有Barrier

**问题2：性能不升反降**
- 检查线程组大小是否与Warp对齐
- 检查是否存在线程发散（同一组内线程走不同分支）

**问题3：GPU崩溃/TDR超时**
- 计算量过大导致GPU超过Windows TDR（超时检测恢复）的2秒限制
- 减小Dispatch规模或分多帧执行

**问题4：移动端Compute Shader性能差**
- 移动端的共享内存和原子操作开销比PC端大很多
- 优先使用简单的逐纹素计算，避免复杂同步

## 11. 实际使用案例

- Unity VFX Graph使用Compute Shader驱动大规模粒子系统（百万级粒子实时模拟）
- 《战地》系列用Compute Shader做屏幕空间反射（SSR）和环境遮蔽（SSAO）
- 《荒野大镖客2》的草地系统使用Compute Shader进行GPU剔除和LOD选择
- 机器学习推理框架（TensorRT、DirectML）使用Compute Shader加速GPU推理
- 光线追踪的BVH遍历和着色计算大量使用Compute Shader
- 排序算法（Bitonic Sort）在Compute Shader中实现高效GPU并行排序
