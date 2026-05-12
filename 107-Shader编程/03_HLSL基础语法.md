# HLSL基础语法

## 1. 核心理论

HLSL（High-Level Shading Language）是微软开发的着色器语言，主要用于DirectX平台（Direct3D 11/12）和Unity引擎。HLSL与GLSL的语法非常相似，但在语义系统（Semantics）、常量缓冲区（Constant Buffer）和类型命名上有显著差异。

### 1.1 HLSL与GLSL核心差异对比

```
差异项          GLSL                          HLSL
─────────────────────────────────────────────────────────
矩阵命名        mat4                           float4x4
向量命名        vec4                           float4
语义系统        layout(location=N)             : SEMANTIC
uniform组织     分散声明                       cbuffer { ... }
矩阵存储        列主序（默认）                  行主序（默认）
入口函数        main()                         自定义函数名
精度限定符      highp/mediump/lowp             min16float/min10float
版本声明        #version 330 core              无（由编译目标决定）
纹理采样        texture(sampler, uv)           tex.Sample(sampler, uv)
函数修饰符      无                             [numthreads(X,Y,Z)]
```

### 1.2 HLSL版本与编译目标

```
HLSL版本      Direct3D版本     Shader Model    主要特性
─────────────────────────────────────────────────────────
HLSL 2.0      D3D9             SM 2.0          基础着色器
HLSL 3.0      D3D9             SM 3.0          动态分支
HLSL 4.0      D3D10            SM 4.0          几何着色器
HLSL 5.0      D3D11            SM 5.0          计算着色器、曲面细分
HLSL 5.1      D3D12            SM 5.1          资源绑定（Descriptor Table）
HLSL 6.0      D3D12            SM 6.0          波前操作（Wave Ops）
HLSL 6.1      D3D12            SM 6.1          SV_ViewID
HLSL 6.2      D3D12            SM 6.2          16位类型
HLSL 6.3      D3D12            SM 6.3          DXR光线追踪
HLSL 6.4      D3D12            SM 6.4          网格着色器
HLSL 6.5      D3D12            SM 6.5          采样器反馈
HLSL 6.6      D3D12            SM 6.6          WaveMatrix、Work Graphs

// 编译指令
// DXBC (旧版): fxc /T vs_5_0 /E Vert shader.hlsl
// DXIL (DXC):  dxc -T vs_6_0 -E Vert shader.hlsl
// Unity: 自动编译，通过#pragma指定目标
```

## 2. 数据类型

### 2.1 标量与向量

```hlsl
// 标量
float f = 3.14f;           // 32位浮点
half h = 1.0h;             // 16位浮点（移动端/部分PC）
double d = 3.14159265;     // 64位浮点
int i = 42;                // 32位有符号整数
uint u = 100u;             // 32位无符号整数
bool b = true;             // 布尔

// 向量
float2 v2 = float2(1.0, 2.0);
float3 v3 = float3(1.0, 2.0, 3.0);
float4 v4 = float4(1.0, 2.0, 3.0, 4.0);

// 构造
float3 v = float3(1.0);              // 广播：(1.0, 1.0, 1.0)
float4 v4b = float4(v3, 1.0);        // vec3 + float → vec4
float4 v4c = float4(v2, v2);         // vec2 + vec2 → vec4

// Swizzle（与GLSL相同）
float3 pos = float3(1.0, 2.0, 3.0);
float x = pos.x;                     // 或 pos.r
float2 xy = pos.xy;                  // 或 pos.rg
float3 bgr = pos.bgr;                // 反转

// 精度类型（移动端重要）
min16float f16 = 1.0;                // 最少16位精度
min10float f10 = 1.0;                // 最少10位精度
// Unity中使用 half 代替 min16float，fixed 代替 min10float
half4 color = half4(1.0, 0.0, 0.0, 1.0);
fixed4 tint = fixed4(1.0, 1.0, 1.0, 1.0);
```

### 2.2 矩阵

```hlsl
// 矩阵声明：float行x列（与GLSL相反！）
float4x4 mvp;               // 4×4矩阵
float3x3 normalMat;         // 3×3矩阵
float2x3 m23;               // 2列3行矩阵

// 默认行主序存储
// float3x3 m = float3x3(
//     1, 2, 3,      ← 第一行
//     4, 5, 6,      ← 第二行
//     7, 8, 9       ← 第三行
// );
// 注意：GLSL是列主序，同样的数据填法在HLSL中布局不同！

// 可以用row_major或column_major显式指定
row_major float4x4 m1;     // 行主序（HLSL默认）
column_major float4x4 m2;  // 列主序（与GLSL一致）

// 全局设置列主序（推荐，与GLSL/Unity保持一致）
#pragma pack_matrix(column_major)

// 矩阵访问
float4 row = mvp[1];       // 获取第2行（注意：行主序时取的是行）
float elem = mvp[1][2];    // 第2行第3列

// 矩阵运算
float4 worldPos = mul(model, float4(pos, 1.0)); // 矩阵×向量
// 注意：mul的参数顺序很重要！
// mul(vector, matrix) — 向量在左（行向量）
// mul(matrix, vector) — 向量在右（列向量）
```

## 3. 语义系统（Semantics）

语义是HLSL最核心的特性，用于标记变量在管线中的用途和数据流向。

### 3.1 系统值语义（SV_前缀）

```hlsl
// 顶点着色器输出
SV_POSITION      // 裁剪空间位置（顶点着色器必须输出）

// 片元着色器输出
SV_TARGET0       // 渲染目标0（颜色输出）
SV_TARGET1       // 渲染目标1（MRT输出）
SV_DEPTH         // 深度输出（覆盖默认深度）

// 片元着色器输入
SV_Position      // 片元的窗口坐标（等同于gl_FragCoord）
SV_IsFrontFace   // bool，是否正面
SV_SampleIndex   // 多重采样样本索引
SV_VertexID      // 顶点索引
SV_InstanceID    // 实例索引
SV_DispatchThreadID  // Compute Shader全局线程ID
SV_GroupThreadID     // Compute Shader组内线程ID
SV_GroupID           // Compute Shader线程组ID
SV_GroupIndex        // Compute Shader组内线性索引
```

### 3.2 自定义语义

```hlsl
// 自定义语义用于顶点着色器→片元着色器的数据传递
struct Varyings
{
    float4 positionCS : SV_POSITION;   // 系统语义
    float3 worldPos   : TEXCOORD0;     // 自定义语义（最多到TEXCOORD7）
    float3 normalWS   : TEXCOORD1;
    float2 uv         : TEXCOORD2;
    float4 color      : COLOR;         // 顶点颜色
};
// TEXCOORD语义不仅用于纹理坐标，也用于传递任意插值数据
```

## 4. 常量缓冲区（Constant Buffer / CBUFFER）

Constant Buffer是HLSL管理uniform变量的方式，DirectX 11/12要求按16字节对齐。

### 4.1 对齐规则

```
16字节对齐规则：
float4 = 16字节 → 占1个槽位
float3 + float = 16字节 → 占1个槽位（打包到一行）
float + float3 = 不自动打包！float占4字节，float3占12字节但要新开一行
float2 + float2 = 16字节 → 占1个槽位

推荐：总是按4的倍数组织变量，避免跨行浪费
```

```hlsl
// DX11/12 常量缓冲区
cbuffer PerObject : register(b0)
{
    row_major float4x4 World;        // 64字节 = 4行
    row_major float4x4 ViewProj;     // 64字节 = 4行
    float4 Color;                    // 16字节
    // 总计：144字节，对齐到16的倍数 = 144字节
};

cbuffer PerFrame : register(b1)
{
    float3 LightDir;      // 12字节
    float Time;           // 4字节  → 合并为16字节一行
    float3 CameraPos;     // 12字节
    float DeltaTime;      // 4字节  → 合并为16字节一行
    // 注意：如果写成 float3 LightDir; float3 CameraPos; float Time; float DeltaTime;
    // LightDir占12字节，Time占4字节 → 一行
    // CameraPos需要新一行（12字节），DeltaTime占4字节 → 一行
    // 但如果想让CameraPos和Time在同一行，需要显式组织
};

// Unity URP中的CBUFFER写法（兼容SRP Batcher）
CBUFFER_START(UnityPerMaterial)
    half4 _BaseColor;
    half _Metallic;
    half _Smoothness;
    half _BumpScale;
    half _OcclusionStrength;
CBUFFER_END
```

### 4.2 寄存器绑定

```hlsl
// 寄存器类型：
// b0-b13  → Constant Buffer (CBV)
// t0-t127 → Shader Resource View (SRV) - 纹理、Buffer
// s0-s15  → Sampler State
// u0-u7   → Unordered Access View (UAV) - RWTexture, RWBuffer

Texture2D MainTex : register(t0);
Texture2D NormalMap : register(t1);
TextureCube EnvMap : register(t2);
SamplerState LinearClamp : register(s0);
SamplerState PointWrap : register(s1);
RWTexture2D<float4> Output : register(u0);
```

## 5. 纹理与采样

```hlsl
// 声明
Texture2D<float4> _MainTex : register(t0);
Texture2D<float> _ShadowMap : register(t1);
TextureCube<float4> _Skybox : register(t2);
Texture2DArray<float4> _TexArray : register(t3);
Texture2DMS<float4, 4> _MSAATex : register(t4);  // 4倍MSAA

// 采样器状态
SamplerState sampler_Linear : register(s0);
SamplerState sampler_Point : register(s1);
SamplerComparisonState sampler_Shadow : register(s2); // 深度比较采样

// 采样操作
float4 color = _MainTex.Sample(sampler_Linear, uv);              // 普通采样
float4 colorLOD = _MainTex.SampleLevel(sampler_Linear, uv, 2.0); // 指定LOD
float4 colorBias = _MainTex.SampleBias(sampler_Linear, uv, 1.0); // LOD偏移
float4 colorGrad = _MainTex.SampleGrad(sampler_Linear, uv, ddx, ddy); // 手动梯度
float4 texel = _MainTex.Load(int3(pixelX, pixelY, mipLevel));    // 直接加载（无过滤）

// Shadow Map采样
float shadow = _ShadowMap.SampleCmpLevelZero(sampler_Shadow, uv, depth);
// 返回0.0（在阴影中）或1.0（不在阴影中），硬件PCF

// 纹理尺寸
uint width, height, levels;
_MainTex.GetDimensions(0, width, height, levels); // 获取Mip层级0的尺寸和层级数
```

## 6. 函数与控制流

```hlsl
// 函数声明
float3 CalcBlinnPhong(float3 normal, float3 lightDir, float3 viewDir,
                      float3 lightColor, float shininess)
{
    float3 ambient = 0.1 * lightColor;
    float NdotL = max(dot(normal, lightDir), 0.0);
    float3 diffuse = NdotL * lightColor;

    float3 halfDir = normalize(lightDir + viewDir);
    float NdotH = max(dot(normal, halfDir), 0.0);
    float3 specular = pow(NdotH, shininess) * lightColor;

    return ambient + diffuse + specular;
}

// 内联提示（编译器通常自动内联）
inline float Luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// 模板函数（HLSL 2021+）
template<typename T>
T Saturate(T x)
{
    return clamp(x, T(0), T(1));
}

// 控制流
[branch] if (condition) { ... }  // 提示编译器使用分支
[flatten] if (condition) { ... }  // 提示编译器展平（无分支）

for (int i = 0; i < MAX_LIGHTS; i++) { ... }
while (condition) { ... }
do { ... } while (condition);

// 循环属性
[loop] for (int i = 0; i < count; i++) { ... }    // 强制不展开
[unroll] for (int i = 0; i < 4; i++) { ... }      // 强制展开

// discard — 丢弃当前片元
if (alpha < cutoff)
    discard;
```

## 7. 波前操作（Wave Ops，SM 6.0+）

波前（Wave/Warp）是GPU上同时执行的一组线程（通常32或64个）。波前操作允许组内线程直接通信，无需共享内存。

```hlsl
// SM 6.0+ 波前内建函数
bool WaveActiveAllTrue(bool expr);     // 组内所有线程是否都为true
bool WaveActiveAnyTrue(bool expr);     // 组内是否有线程为true
uint WaveActiveCountBits(bool expr);   // 组内为true的线程数
float WaveActiveSum(float expr);       // 组内求和
float WaveActiveProduct(float expr);   // 组内求积
float WaveActiveMin(float expr);       // 组内最小值
float WaveActiveMax(float expr);       // 组内最大值

// 波前投票（高效条件计算）
float weightedSum = WaveActiveSum(weight * contribution);
uint activeCount = WaveActiveCountBits(true);
float average = weightedSum / activeCount;

// 波前 shuffle
float WaveReadLaneAt(float expr, uint lane);  // 读取指定线程的值
float WaveReadLaneFirst(float expr);          // 读取第一个活跃线程的值
```

## 8. 完整的Unity URP Shader示例

```hlsl
Shader "Custom/URP_BlinnPhong"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _MainTex ("Main Texture", 2D) = "white" {}
        _Specular ("Specular", Range(0, 1)) = 0.5
        _Gloss ("Gloss", Range(1, 256)) = 20
    }

    SubShader
    {
        Tags {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry"
        }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            // 属性声明（必须在CBUFFER中！SRP Batcher要求）
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _MainTex_ST;
                float _Specular;
                float _Gloss;
            CBUFFER_END

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float3 viewDirWS : TEXCOORD2;
                float fogFactor : TEXCOORD3;
            };

            Varyings vert(Attributes input)
            {
                Varyings output;
                VertexPositionInputs posInputs = GetVertexPositionInputs(input.positionOS.xyz);
                VertexNormalInputs normInputs = GetVertexNormalInputs(input.normalOS);

                output.positionCS = posInputs.positionCS;
                output.uv = TRANSFORM_TEX(input.uv, _MainTex);
                output.normalWS = normInputs.normalWS;
                output.viewDirWS = GetWorldSpaceNormalizeViewDir(posInputs.positionWS);
                output.fogFactor = ComputeFogFactor(posInputs.positionCS.z);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                half4 texColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv);
                half4 baseColor = texColor * _BaseColor;

                Light mainLight = GetMainLight();
                half3 normalWS = normalize(input.normalWS);
                half3 lightDir = mainLight.direction;
                half3 viewDir = normalize(input.viewDirWS);

                // 漫反射
                half NdotL = saturate(dot(normalWS, lightDir));
                half3 diffuse = baseColor.rgb * mainLight.color * NdotL;

                // 高光（Blinn-Phong）
                half3 halfDir = normalize(lightDir + viewDir);
                half NdotH = saturate(dot(normalWS, halfDir));
                half3 specular = mainLight.color * _Specular * pow(NdotH, _Gloss);

                // 环境光
                half3 ambient = SampleSH(normalWS) * baseColor.rgb;

                half3 finalColor = ambient + diffuse + specular;
                finalColor = MixFog(finalColor, input.fogFactor);

                return half4(finalColor, baseColor.a);
            }
            ENDHLSL
        }
    }
}
```

## 9. Compute Shader（HLSL）

```hlsl
// 粒子更新Compute Shader
#pragma kernel UpdateParticles

struct Particle
{
    float3 position;
    float3 velocity;
    float age;
    float maxLife;
    float size;
    float pad;
};

RWStructuredBuffer<Particle> particles : register(u0);

cbuffer Params : register(b0)
{
    float deltaTime;
    float3 gravity;
    float3 spawnPos;
    float spawnRadius;
    uint particleCount;
};

[numthreads(256, 1, 1)]
void UpdateParticles(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= particleCount) return;

    Particle p = particles[id.x];

    // 物理更新
    p.velocity += gravity * deltaTime;
    p.position += p.velocity * deltaTime;
    p.age += deltaTime;

    // 重生
    if (p.age >= p.maxLife)
    {
        p.position = spawnPos + float3(
            frac(sin(id.x * 12.9898) * 43758.5453) * 2.0 - 1.0,
            frac(sin(id.x * 78.233) * 43758.5453),
            frac(sin(id.x * 39.346) * 43758.5453) * 2.0 - 1.0
        ) * spawnRadius;
        p.velocity = float3(0, 2, 0);
        p.age = 0;
    }

    particles[id.x] = p;
}
```

## 10. 性能提示

- 使用`half`代替`float`（移动端可节省50%的ALU周期）
- Constant Buffer按16字节对齐，避免padding浪费
- 减少纹理采样次数，打包多张单通道纹理为一张RGBA纹理
- 使用`mad`融合指令：编译器会将`a * b + c`优化为单条指令
- 避免在动态分支中进行纹理采样
- 波前操作（SM 6.0+）比共享内存更快，优先使用

## 11. 常见问题与调试

**问题1：Constant Buffer对齐错误导致数据错位**
- 使用`cbuffer`内的`// N byte`注释跟踪偏移
- 确保float3后跟随的变量填满16字节行

**问题2：Shader编译警告"implicit truncation"**
- 向量维度不匹配时自动截断会产生警告
- 显式使用`.xyz`或`(float3)v4`进行转换

**问题3：纹理采样返回全0**
- 检查register绑定是否与C++端对应
- 检查采样器状态是否正确创建

## 12. 实际使用案例

- Unity所有Shader（Surface Shader除外）底层均使用HLSL，通过HLSLcc转译为GLSL/Vulkan
- Unreal Engine的材质编辑器同时支持HLSL和自定义节点，底层编译为HLSL
- Xbox游戏《光环：无限》使用HLSL 6.x实现DirectX 12 Ultimate特性（Mesh Shader、Sampler Feedback）
- DirectX 12的DXR（光线追踪）管线使用HLSL 6.3+的RayQuery和RaytracingAccelerationStructure
- 《极限竞速：地平线5》使用HLSL实现了全动态日夜循环和天气系统的渲染管线
