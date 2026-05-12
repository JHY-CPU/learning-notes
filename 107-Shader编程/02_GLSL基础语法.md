# GLSL基础语法

## 1. 核心理论

GLSL（OpenGL Shading Language）是OpenGL的着色器语言，语法风格类似C语言，专为GPU大规模并行计算设计。每个Shader程序以`main()`函数为入口，同一份代码在GPU上同时对数千个顶点/片元并行执行。

### 1.1 版本演进

```
GLSL版本与OpenGL对应关系：
GLSL 1.10 → OpenGL 2.0  (2004)
GLSL 1.20 → OpenGL 2.1
GLSL 1.30 → OpenGL 3.0  引入in/out取代varying
GLSL 1.50 → OpenGL 3.2  引入geometry shader
GLSL 3.30 → OpenGL 3.3  最常用稳定版本
GLSL 4.00 → OpenGL 4.0  引入tessellation
GLSL 4.20 → OpenGL 4.2  引入compute shader (共享内存)
GLSL 4.30 → OpenGL 4.3  引入compute shader正式版
GLSL 4.60 → OpenGL 4.6  最新稳定版本

// 版本声明（必须在Shader第一行）
#version 330 core     // OpenGL 3.3 核心模式
#version 460 core     // OpenGL 4.6 核心模式
#version 300 es       // OpenGL ES 3.0（移动端/WebGL2）
#version 100          // OpenGL ES 2.0（旧移动端/WebGL1，无in/out）
```

### 1.2 精度限定符详解

精度限定符控制数值的存储精度，在移动端（OpenGL ES）至关重要，直接影响性能和画质。

```
精度级别与范围（OpenGL ES）：
lowp:    float 精度至少 2^-8，范围 [-2, 2]     → 适合颜色、简单标志
mediump: float 精度至少 2^-10，范围 [-2^10, 2^10] → 适合UV、一般计算
highp:   float 精度至少 2^-16，范围 [-2^16, 2^16] → 适合世界坐标、深度

PC端OpenGL忽略精度限定符（始终使用32位float）
OpenGL ES 3.0中，片元着色器的float默认无精度，必须手动指定
```

```glsl
// 全局精度设置（放在Shader顶部）
precision highp float;     // float默认高精度
precision mediump int;     // int默认中精度
precision lowp sampler2D;  // 采样器默认低精度

// 局部精度覆盖
highp vec4 worldPosition;    // 世界坐标必须高精度
mediump vec2 texCoord;       // UV坐标用中精度
lowp vec4 vertexColor;       // 颜色用低精度足够
lowp float alpha;            // Alpha值用低精度

// 精度选择对性能的影响（移动端）：
// highp → 更多ALU周期，更耗电
// lowp  → 更快的ALU运算，节省带宽和电量
// 但精度不足会导致：条纹状色带（banding）、动画抖动、坐标偏移
```

## 2. 数据类型完整参考

### 2.1 标量类型

```glsl
float f = 3.14;           // 32位浮点数
int i = 42;               // 32位有符号整数
uint u = 100u;            // 32位无符号整数（GLSL 1.30+）
bool b = true;            // 布尔值

// 类型转换
float f2 = float(i);      // int → float
int i2 = int(f);          // float → int（截断）
int i3 = int(round(f));   // float → int（四舍五入）
```

### 2.2 向量类型

```glsl
// 声明
vec2 v2 = vec2(1.0, 2.0);              // 2D float向量
vec3 v3 = vec3(1.0, 2.0, 3.0);         // 3D float向量
vec4 v4 = vec4(1.0, 2.0, 3.0, 4.0);    // 4D float向量

ivec2 iv2 = ivec2(1, 2);               // 整数向量
uvec3 uv3 = uvec3(1u, 2u, 3u);         // 无符号整数向量
bvec4 bv4 = bvec4(true, false, true, false); // 布尔向量

// 构造技巧
vec3 v = vec3(1.0);                    // 标量广播：(1.0, 1.0, 1.0)
vec4 v4b = vec4(v3, 1.0);              // 从vec3 + 标量构造vec4
vec4 v4c = vec4(v2, v2);               // 从两个vec2构造vec4

// Swizzle（分量重组）—— GLSL最强大的特性之一
vec3 pos = vec3(1.0, 2.0, 3.0);
float x = pos.x;          // 提取x分量 = 1.0
float y = pos.g;          // 也可以用颜色命名 = 2.0
vec2 xy = pos.xy;         // 提取xy分量 = (1.0, 2.0)
vec3 bgr = pos.bgr;       // 反转分量 = (3.0, 2.0, 1.0)
vec3 yzx = pos.yzx;       // 旋转分量 = (2.0, 3.0, 1.0)

// 可以混合使用xyzw和rgba，但不能混用
// pos.xrg  ← 合法
// pos.xra  ← 非法（混用了两套命名系统）
// pos.stpq ← 纹理坐标命名（第三套，很少用）

// 向量运算（逐分量运算）
vec3 a = vec3(1.0, 2.0, 3.0);
vec3 b = vec3(4.0, 5.0, 6.0);
vec3 sum = a + b;           // (5.0, 7.0, 9.0) — 逐分量加法
vec3 prod = a * b;          // (4.0, 10.0, 18.0) — 逐分量乘法
float dot_ab = dot(a, b);   // 点积 = 1*4 + 2*5 + 3*6 = 32
vec3 cross_ab = cross(a, b); // 叉积 = (-3, 6, -3)
```

### 2.3 矩阵类型

```glsl
// 声明
mat2 m2 = mat2(1.0);               // 2x2单位矩阵
mat3 m3 = mat3(1.0);               // 3x3单位矩阵
mat4 m4 = mat4(1.0);               // 4x4单位矩阵
mat2x3 m23 = mat2x3(1.0);          // 2列3行矩阵（GLSL 1.20+）
mat3x2 m32 = mat3x2(1.0);          // 3列2行矩阵

// 矩阵构造（列主序！先填第一列，再第二列...）
// | m[0][0]  m[1][0]  m[2][0] |
// | m[0][1]  m[1][1]  m[2][1] |
// | m[0][2]  m[1][2]  m[2][2] |
mat3 identity = mat3(
    1.0, 0.0, 0.0,   // 第一列
    0.0, 1.0, 0.0,   // 第二列
    0.0, 0.0, 1.0    // 第三列
);

// 矩阵访问
float elem = m3[1][2];  // 第2列第3行的元素
vec3 col = m3[1];       // 第2列（整个列向量）

// 矩阵运算
mat4 mvp = projection * view * model;  // 矩阵乘法（从右到左应用）
vec4 worldPos = model * vec4(position, 1.0);  // 矩阵 × 向量

// 矩阵函数
mat3 transposed = transpose(m3);           // 转置
float det = determinant(m4);               // 行列式
mat4 inv = inverse(m4);                    // 逆矩阵（开销大！）
```

### 2.4 采样器类型

```glsl
// 2D纹理
uniform sampler2D u_DiffuseMap;
uniform sampler2DShadow u_ShadowMap;    // 深度比较采样

// 立方体贴图
uniform samplerCube u_Skybox;

// 数组纹理
uniform sampler2DArray u_TextureArray;

// 多重采样（MSAA）
uniform sampler2DMS u_MSAATexture;

// 缓冲区对象
uniform samplerBuffer u_LightBuffer;

// 纹理采样
vec4 color = texture(u_DiffuseMap, vec2(0.5, 0.5));  // 普通采样
vec4 lodColor = textureLod(u_DiffuseMap, uv, 2.0);    // 指定Mip层级
vec4 projColor = textureProj(u_DiffuseMap, vec3(0.5, 0.5, 2.0)); // 透视除法后采样
vec4 gradColor = textureGrad(u_DiffuseMap, uv, dFdx(uv), dFdy(uv)); // 指定梯度
vec4 cubeColor = texture(u_Skybox, reflectDir);       // 立方体贴图采样

// 纹理查询
ivec2 size = textureSize(u_DiffuseMap, 0);  // 获取纹理尺寸（Mip层级0）
int levels = textureQueryLevels(u_DiffuseMap); // 获取Mip层级数（GLSL 4.30+）
```

## 3. 存储限定符

存储限定符决定了变量的数据流向和生命周期，是理解Shader间通信的关键。

```glsl
// ============ uniform ============
// CPU端每帧传入的全局常量，整个Draw Call内不变，所有着色器阶段共享
uniform mat4 u_ModelViewProjection;
uniform vec3 u_CameraPosition;
uniform float u_Time;
uniform int u_LightCount;

// ============ in（输入变量）===========
// 顶点着色器：从顶点缓冲区接收顶点属性
layout(location = 0) in vec3 a_Position;    // 位置
layout(location = 1) in vec3 a_Normal;      // 法线
layout(location = 2) in vec2 a_TexCoord;    // UV
layout(location = 3) in vec4 a_Color;       // 顶点颜色
layout(location = 4) in vec4 a_Tangent;     // 切线（xyz=方向, w=手性）

// 片元着色器：从光栅化阶段接收插值后的属性
in vec3 v_WorldPos;     // 从顶点着色器传入（经过插值）
in vec3 v_Normal;
in vec2 v_TexCoord;

// ============ out（输出变量）===========
// 顶点着色器：输出到光栅化阶段（会被插值）
out vec3 v_WorldPos;
out vec3 v_Normal;

// 片元着色器：输出到帧缓冲区
layout(location = 0) out vec4 fragColor;    // 颜色输出
layout(location = 1) out vec4 fragNormal;   // MRT：法线输出（延迟渲染）

// ============ const ============
// 编译时常量，不可修改
const float PI = 3.14159265359;
const vec3 RED = vec3(1.0, 0.0, 0.0);
const int MAX_LIGHTS = 16;

// ============ buffer（SSBO）===========
// 着色器存储缓冲区对象，可读写的大数据块（GLSL 4.30+）
layout(std430, binding = 0) buffer ParticleBuffer {
    vec4 positions[];
    vec4 velocities[];
};

// ============ shared（仅Compute Shader）===========
// 线程组内共享内存
shared vec3 sharedData[256];
```

## 4. 内置变量

```glsl
// 顶点着色器内置输出
gl_Position    // vec4 裁剪空间位置（必须写入）
gl_PointSize   // float 点精灵大小（渲染点时）
gl_ClipDistance[] // float[] 用户裁剪平面距离

// 片元着色器内置输入
gl_FragCoord   // vec4 片元的窗口坐标(x,y)和深度(z)
gl_FrontFacing // bool 是否正面朝向
gl_PointCoord  // vec2 点精灵内的坐标[0,1]
gl_SampleID    // int 当前多重采样样本索引
gl_HelperInvocation // bool 是否为辅助调用（被跳过的像素）

// 片元着色器内置输出
gl_FragDepth   // float 可选写入深度值（默认使用gl_FragCoord.z）
gl_FragColor   // vec4 (旧版，GLSL 1.30+用自定义out变量)
```

## 5. 内置函数分类

### 5.1 常用数学函数

```glsl
// 绝对值、符号
abs(x)          // |x|
sign(x)         // -1, 0, 1

// 取整
floor(x)        // 向下取整
ceil(x)         // 向上取整
round(x)        // 四舍五入
trunc(x)        // 截断小数部分
fract(x)        // 小数部分：x - floor(x)
mod(x, y)       // 取模：x - y * floor(x/y)

// 幂和根
pow(x, y)       // x^y（开销大！尽量用其他替代）
sqrt(x)         // 平方根
inversesqrt(x)  // 1/sqrt(x)（比先sqrt再除法快）
exp(x)          // e^x
exp2(x)         // 2^x
log(x)          // ln(x)
log2(x)         // log2(x)

// 混合和步进
mix(x, y, a)    // 线性插值：x*(1-a) + y*a
step(edge, x)   // x < edge ? 0.0 : 1.0
smoothstep(e0, e1, x) // 平滑过渡 [e0,e1] → [0,1]
clamp(x, min, max)    // 钳制到[min, max]
saturate(x)     // 钳制到[0,1]（HLSL有，GLSL用clamp(x,0.0,1.0)）

// 向量函数
length(v)       // 向量长度
distance(a, b)  // 两点距离
normalize(v)    // 归一化
dot(a, b)       // 点积
cross(a, b)     // 叉积（仅vec3）
reflect(I, N)   // 反射向量：I - 2*dot(N,I)*N
refract(I, N, eta) // 折射向量

// 三角函数
sin, cos, tan, asin, acos, atan

// 偏导数（在片元着色器中计算屏幕空间梯度）
dFdx(p)         // x方向偏导数（用于法线贴图LOD、平面着色）
dFdy(p)         // y方向偏导数
```

### 5.2 纹理函数

```glsl
// 基本采样
texture(sampler, coord)                    // 普通采样
textureLod(sampler, coord, lod)           // 指定LOD层级
textureOffset(sampler, coord, offset)     // 带整数偏移的采样
textureProj(sampler, coord)               // 透视除法后采样
texelFetch(sampler, ivec2(x,y), lod)      // 直接获取纹素（无过滤）

// 偏导数采样（手动控制Mip层级计算）
textureGrad(sampler, coord, dPdx, dPdy)

// 采样比较（Shadow Map）
texture(sampler2DShadow, vec3(coord, refZ)) // 返回0或1
```

### 5.3 原子操作与同步（Compute Shader）

```glsl
// 原子操作
atomicAdd(mem, data)     // mem += data，返回旧值
atomicMin(mem, data)     // mem = min(mem, data)
atomicMax(mem, data)     // mem = max(mem, data)
atomicAnd/Or/Xor/Exchange/CompSwap

// 内存屏障
memoryBarrier()           // 确保之前的写入对所有后续读取可见
memoryBarrierShared()     // 共享内存屏障
groupMemoryBarrier()      // 组内内存屏障
barrier()                 // 线程同步点（组内所有线程到达此处才继续）
```

## 6. 控制流与函数

```glsl
// 条件语句
if (condition) { ... }
else if (condition2) { ... }
else { ... }

// 循环
for (int i = 0; i < 10; i++) { ... }
while (condition) { ... }
do { ... } while (condition);

// 提前退出
break;
continue;
return;
discard;  // 仅片元着色器：丢弃当前片元

// 自定义函数
// 在调用之前需要声明或定义
float CalcAttenuation(float distance, float radius)
{
    // 物理正确的平方衰减
    float atten = 1.0 - clamp(distance / radius, 0.0, 1.0);
    return atten * atten;  // 平方衰减
}

// 函数参数限定符
void ProcessLight(in vec3 lightPos,      // 输入参数（默认）
                  out vec3 lightColor,    // 输出参数
                  inout vec3 accumColor)  // 输入输出参数
{
    // ...
}
```

## 7. 预处理器

```glsl
#version 460 core

// 条件编译
#define QUALITY_HIGH
#ifdef QUALITY_HIGH
    #define SHADOW_SAMPLES 16
#elif defined(QUALITY_MEDIUM)
    #define SHADOW_SAMPLES 8
#else
    #define SHADOW_SAMPLES 4
#endif

// 宏函数（注意参数需要括号避免运算符优先级问题）
#define saturate(x) clamp(x, 0.0, 1.0)
#define LERP(a, b, t) mix(a, b, t)

// 条件排除
#ifndef MOBILE_PLATFORM
    // PC端专属代码
#endif

// 行号和文件名
#line 100  // 设置当前行号
#pragma optimize(on)   // 编译器优化提示
#pragma debug(on)      // 调试信息
```

## 8. Unity中使用GLSL（通过HLSLcc转译）

Unity底层使用HLSLcc将HLSL转译为GLSL/Vulkan SPIR-V。若要在Unity中写GLSL风格代码：

```hlsl
// Unity HLSL（语法类似GLSL）
#pragma vertex vert
#pragma fragment frag

struct Attributes {
    float4 positionOS : POSITION;
    float2 uv : TEXCOORD0;
};

struct Varyings {
    float4 positionCS : SV_POSITION;
    float2 uv : TEXCOORD0;
};

Varyings vert(Attributes v) {
    Varyings o;
    o.positionCS = TransformObjectToHClip(v.positionOS.xyz);
    o.uv = v.uv;
    return o;
}

half4 frag(Varyings i) : SV_Target {
    half4 col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.uv);
    return col;
}
```

## 9. 性能提示

- `pow()` 是最昂贵的标量函数之一，尽量用 `x*x` 替代 `pow(x,2)`，用 `dot(v,v)` 替代 `pow(length(v),2)`
- `normalize()` 内部包含 `inversesqrt` + 乘法，避免对已归一化的向量重复调用
- 在片元着色器中避免使用 `discard`，它会禁用Early-Z优化
- 精度限定符在移动端有实际影响：颜色/UV用`mediump`，世界坐标用`highp`
- 循环展开：固定次数的小循环通常会被编译器自动展开

## 10. 常见问题与调试

**问题1：Shader编译失败但无错误信息**
- 检查版本声明是否为文件第一行
- 检查是否使用了版本不支持的特性

**问题2：画面全黑**
- 检查`gl_Position`是否正确赋值
- 检查uniform变量名称是否与CPU端匹配

**问题3：精度导致的色带（Banding）**
- 将颜色计算的精度从`lowp`提升到`mediump`
- 使用抖动（Dither）技术平滑色阶

## 11. 实际使用案例

- Three.js中所有自定义Shader使用GLSL ES 3.0（WebGL2）编写
- Minecraft Shader Mod（如SEUS Renewed）使用GLSL 4.x实现光线追踪近似效果
- 移动端《原神》使用OpenGL ES 3.0 + GLSL ES 300，精简Shader保证移动端性能
- Shadertoy平台上所有Shader均使用GLSL ES编写，是学习GLSL的最佳资源之一
