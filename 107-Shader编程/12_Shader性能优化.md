# Shader性能优化

## 1. 核心理论

Shader性能优化的目标是在保持画面质量的前提下减少GPU的工作量。GPU的性能瓶颈主要来自四个方面：**ALU运算量**、**纹理带宽**、**过度绘制**和**状态切换**。理解GPU的硬件架构是有效优化的前提。

### 1.1 GPU架构基础

```
现代GPU架构概要：

SM (Streaming Multiprocessor) / CU (Compute Unit)
├── 一组执行核心（Cores / ALUs）— 通常 64-128 个
├── 寄存器文件 — 每个线程独立的寄存器
├── 共享内存 / LDS — 组内线程共享
├── 纹理单元 — 硬件纹理采样和过滤
├── 调度器 — 管理Warp/Wavefront的执行
│
Warp / Wavefront（线程束）
├── NVIDIA: 32线程/Warp
├── AMD: 64线程/Wavefront
├── 同一Warp内的线程执行相同的指令（SIMT模型）
├── 分支发散时，不同路径串行执行
│
关键性能指标：
  ALU利用率：实际执行的ALU指令 / 理论峰值
  纹理带宽：纹理采样的数据传输速率
  寄存器压力：每个线程使用的寄存器数
  占用率（Occupancy）：活跃Warp数 / 最大Warp数
```

### 1.2 着色器复杂度度量

```
常见操作的ALU开销（相对值）：

操作                  ALU周期    说明
──────────────────────────────────────────
整数加法               1         最基本操作
浮点加法/乘法          1         GPU高度优化
mad (乘加)             1         融合指令，1周期完成
倒数 (1/x)             2-4       快速近似可用
平方根 sqrt            4-6       慢
倒数平方根 rsqrt       1-2       比sqrt快很多
sin/cos                6-12      非常慢，尽量避免
pow(x,y)              10-20     最昂贵的标量函数
exp/log               6-10      慢
纹理采样 (cache hit)   ~4        带宽密集
纹理采样 (cache miss)  ~100+     严重瓶颈
```

## 2. 过度绘制（Overdraw）

过度绘制指同一像素被多次渲染，是最常见也最致命的性能问题。

### 2.1 Overdraw分析

```
过度绘制热力图颜色含义：
  蓝色 = 1次绘制（理想）
  绿色 = 2次
  黄色 = 3次
  红色 = 4次以上（严重问题）

常见原因：
  1. 半透明物体层层叠加（粒子、UI）
  2. 不透明物体未从前到远排序
  3. 不必要的全屏后处理Pass
  4. 地形、水面等大面积半透明
```

### 2.2 Overdraw优化策略

```
1. 不透明物体从前到远渲染（Front-to-Back）
   → 利用Early-Z：近处物体先写入深度，远处物体被深度测试剔除
   → 被剔除的像素不执行片元着色器

2. Z-Pre Pass（深度预渲染）
   Pass 1：仅写入深度缓冲（片元着色器为空或极简）
   Pass 2：正常渲染，深度测试通过的像素才执行完整着色器
   → 优势：所有不透明物体的隐藏像素都不执行片元着色器
   → 代价：多一次Draw Call遍历和顶点处理

3. 减少半透明物体的屏幕覆盖面积
   → 使用更小的粒子
   → 限制粒子密度
   → UI元素使用不透明背景

4. 移动端Tile-Based渲染优化
   → 减少渲染目标切换
   → 使用on-chip内存避免带宽浪费
```

```cpp
// Z-Pre Pass实现（伪代码）
void RenderWithZPrePass()
{
    // Pass 1：深度预渲染
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); // 不输出颜色
    glDepthFunc(GL_LESS);
    for (auto& obj : opaqueObjects) {
        RenderDepthOnly(obj); // 片元着色器为空，只写深度
    }

    // Pass 2：完整渲染
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthFunc(GL_EQUAL); // 只渲染深度等于预渲染的像素
    for (auto& obj : opaqueObjects) {
        RenderFull(obj);
    }
}
```

## 3. 数学运算优化

### 3.1 ALU优化技巧

```glsl
// ====== 技巧1：用mul替代pow ======
// 差：pow(x, 2.0) — 约10-20 ALU周期
// 好：x * x — 1 ALU周期

float distSq = dot(dir, dir);       // 距离平方（比length()快，不需sqrt）
float x4 = x * x;                   // x²（比pow(x,2)快10倍）
float x8 = x4 * x4;                 // x⁴

// ====== 技巧2：使用rsqrt替代1/sqrt ======
// 差：1.0 / sqrt(x) — sqrt(4-6) + 除法(2-4) = 6-10周期
// 好：inversesqrt(x) 或 rsqrt(x) — 1-2周期

vec3 dir = target - origin;
vec3 dirNorm = dir * inversesqrt(dot(dir, dir)); // 比normalize()快

// ====== 技巧3：避免sin/cos ======
// 差：sin(angle), cos(angle) — 各6-12周期
// 好：用多项式近似（精度要求不高时）

float fastSin(float x)
{
    // Bhuban-Ram的二次近似（精度±0.02）
    x = x * 0.31830988618; // x / π
    return 4.0 * x * (1.0 - abs(x));
}

// ====== 技巧4：mad融合指令 ======
// a * b + c 在GPU上是1条指令（而非2条）
// 编译器通常自动优化，但显式写出更好
float result = a * b + c; // 编译为mad

// ====== 技巧5：减少除法 ======
// 除法约2-4 ALU周期
// 差：x / y
// 好：x * (1.0 / y) 如果除数可复用
// 好：x * rcp_y 其中rcp_y = 1.0/y在循环外计算
```

### 3.2 精度优化（移动端关键）

```glsl
// GLSL精度选择
// PC端OpenGL忽略精度限定符（始终32位float）
// OpenGL ES / WebGL中精度有实际影响

precision highp float;    // 32位 — 世界坐标、深度、复杂计算
precision mediump float;  // 16位 — UV坐标、一般颜色计算
precision lowp float;     // 10位 — 颜色、简单标志

// HLSL精度（Unity）
float f = 1.0;           // 32位
half h = 1.0;            // 16位（移动端）或32位（PC端）
fixed fx = 1.0;          // 11位（移动端）或32位（PC端）

// 精度选择指南：
// 位置/深度 → float (highp)
// UV坐标    → half (mediump)
// 颜色      → half 或 fixed (mediump/lowp)
// 法线      → half (mediump)
// 光照结果  → half (mediump)
// 矩阵运算  → float (highp) — 不要降低矩阵精度！
```

```hlsl
// HLSL精度优化示例
half4 Frag(Varyings input) : SV_Target
{
    half4 texColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv);
    half3 N = normalize(input.normalWS);          // half足够
    half3 L = normalize(_MainLight.direction);     // half足够
    half NdotL = saturate(dot(N, L));              // half足够

    half3 color = texColor.rgb * NdotL;
    return half4(color, texColor.a);
}
```

## 4. 分支优化

GPU以Warp/Wavefront为单位执行，组内线程走不同分支会导致**分支发散（Branch Divergence）**，两条路径串行执行。

### 4.1 分支发散详解

```
分支发散示例：
Warp内32个线程执行同一段代码：

if (pixel.x < screenWidth / 2)
    color = texture(texA, uv);   // 路径A
else
    color = texture(texB, uv);   // 路径B

执行过程：
1. 谓词寄存器标记哪些线程走路径A（active mask）
2. 执行路径A（走路径B的线程被禁用）
3. 谓词寄存器翻转，标记走路径B的线程
4. 执行路径B（走路径A的线程被禁用）
5. 合并结果

总时间 = A的执行时间 + B的执行时间（而非max(A,B)）
```

### 4.2 分支避免技巧

```glsl
// ====== 技巧1：用mix/lerp替代if ======
// 差：有分支
if (useTexture)
    color = texture(tex, uv);
else
    color = uniformColor;

// 好：无分支（两路都计算，用标志选择）
vec4 texColor = texture(tex, uv);
color = mix(uniformColor, texColor, float(useTexture));

// ====== 技巧2：用step/moothstep替代比较 ======
// 差：if (value > 0.5) result = 1.0;
// 好：result = step(0.5, value);  // value >= 0.5 ? 1.0 : 0.0

// 平滑step
float smooth = smoothstep(edge0, edge1, value); // 平滑过渡

// ====== 技巧3：常量分支不受影响 ======
// 编译时已知的分支不产生发散
#ifdef QUALITY_HIGH
    // 编译器直接选择这段代码，无运行时开销
    #define SHADOW_SAMPLES 16
#else
    #define SHADOW_SAMPLES 4
#endif

// ====== 技巧4：使用预处理变体 ======
// Unity的shader变体系统
#pragma multi_compile _ QUALITY_LOW QUALITY_HIGH
// 编译时生成两个Shader变体，运行时切换，无分支发散
```

## 5. 纹理带宽优化

### 5.1 纹理压缩

```
纹理格式比较：
格式           bpp (bits per pixel)   压缩比    平台
─────────────────────────────────────────────────────────
RGBA32         32                      1:1      全平台（未压缩）
RGBA16         16                      2:1      全平台
BC1 (DXT1)     4                       8:1      PC
BC3 (DXT5)     8                       4:1      PC
BC7            8                       4:1      PC（高质量）
ASTC 4×4       8                       4:1      移动端（高质量）
ASTC 6×6       3.56                    9:1      移动端
ASTC 8×8       2                       16:1     移动端（低质量）
ETC2           8                       4:1      移动端（Android）
```

### 5.2 纹理打包

```glsl
// ====== 多张单通道纹理合并为一张RGBA ======
// 差：4次采样
float ao = texture(aoMap, uv).r;
float rough = texture(roughMap, uv).r;
float metal = texture(metalMap, uv).r;
float height = texture(heightMap, uv).r;

// 好：1次采样
vec4 packed = texture(packedMap, uv);
float ao = packed.r;
float rough = packed.g;
float metal = packed.b;
float height = packed.a;

// 节省：3次纹理采样 × ~4周期 = 12周期
// 更重要的是节省了3倍的纹理带宽！
```

### 5.3 降低采样次数

```glsl
// 技巧1：顶点着色器中预计算，传入片元着色器
// 差：每个像素都做相同的计算
// 好：在顶点着色器中计算，让光栅化插值

// 技巧2：使用更小的LUT（查找表）替代复杂计算
// 用1D/2D纹理存储预计算结果
vec3 envBRDF = texture(envBRDFLUT, vec2(NdotV, roughness)).rgb;

// 技巧3：跳过远处像素的纹理采样
float distance = length(v_WorldPos - u_CameraPos);
if (distance > u_LODDistance)
{
    // 远处用简化计算
    color = u_SimplifiedColor;
}
else
{
    color = texture(u_DetailedTex, uv).rgb;
}
```

## 6. Shader变体与变体剥离

### 6.1 Shader变体（Variant）

Shader变体通过预处理宏编译出多个版本的Shader，运行时根据需要切换。

```hlsl
// Unity变体声明
#pragma multi_compile _ _SHADOWS_SOFT
#pragma multi_compile _ _MAIN_LIGHT_SHADOWS
#pragma multi_compile _ _ADDITIONAL_LIGHTS
#pragma shader_feature _NORMALMAP

// 每个multi_compile组合都会生成一个独立的Shader变体
// _SHADOWS_SOFT × _MAIN_LIGHT_SHADOWS × _ADDITIONAL_LIGHTS = 2×2×2 = 8个变体
```

### 6.2 变体数量控制

```
变体爆炸问题：
  5个multi_compile开关 = 2⁵ = 32个变体
  10个multi_compile开关 = 2¹⁰ = 1024个变体
  每个变体都要编译和加载 → 内存和加载时间爆炸

解决方案：
1. 使用shader_feature代替multi_compile
   shader_feature只编译被使用的变体
   需要材质上启用对应关键字

2. 变体剥离（Variant Stripping）
   // Unity中通过IPreprocessShaders接口
   // 移除不需要的变体组合

3. 减少multi_compile数量
   // 合并相关功能到一个开关
   #pragma multi_compile _ _ADVANCED_FEATURES
   // 而非
   #pragma multi_compile _ _FEATURE_A
   #pragma multi_compile _ _FEATURE_B
   #pragma multi_compile _ _FEATURE_C
```

## 7. 寄存器压力与占用率

```
占用率（Occupancy）= 活跃Warp数 / 最大Warp数

高占用率 → 更好的延迟隐藏（当一个Warp等待内存时切换到另一个）
低占用率 → 延迟隐藏不足，GPU空闲等待

影响占用率的因素：
1. 每线程寄存器数
   寄存器越多 → 每个SM能容纳的线程越少 → 占用率越低
   优化：减少临时变量，让编译器复用寄存器

2. 共享内存使用量
   共享内存越多 → 每个SM能运行的线程组越少 → 占用率越低

3. 线程组大小
   更大的线程组不一定更好，需要平衡
```

```hlsl
// 减少寄存器使用的技巧：
// 1. 重用变量，不要同时持有太多中间结果
// 2. 使用更小的数据类型（half代替float）
// 3. 避免在循环中声明大量局部变量
// 4. 使用inline函数减少参数传递的寄存器开销
```

## 8. 调试工具与方法

```
常用GPU性能分析工具：

工具                    平台            功能
────────────────────────────────────────────────────────
RenderDoc              PC              帧捕获、Draw Call分析、资源查看
PIX                    Windows/Xbox    DirectX性能分析、时序分析
NVIDIA Nsight          PC (NVIDIA)     CUDA/图形性能分析、Shader调试
Xcode GPU Debugger     macOS/iOS       Metal性能分析
Mali Offline Compiler  移动端          Shader编译分析、寄存器使用
Adreno Profiler        Android         GPU性能分析
Unity Frame Debugger   Unity           Draw Call分析、合批检查
Unity Profiler         Unity           CPU/GPU耗时分析

常用分析方法：
1. GPU Profile → 找到最耗时的Pass
2. Shader分析 → 查看每条指令的周期数
3. 带宽分析 → 查看纹理采样和Buffer读写
4. Overdraw热力图 → 定位过度绘制区域
5. Shader变体分析 → 确认编译了哪些变体
```

## 9. 平台特定优化

### 9.1 移动端优化

```
移动端特别注意：
1. Tile-Based GPU架构（Mali、Adreno、Apple GPU）
   → 减少Render Target切换（每次切换都要flush tile memory）
   → 尽量在同一Pass完成多个RT输出（MRT）
   → 减少读回主存的操作

2. 浮点精度
   → 使用half代替float可节省50% ALU周期
   → 使用mediump代替highp

3. 纹理带宽
   → 使用ASTC/ETC2压缩
   → 降低非关键纹理分辨率

4. 功耗
   → 减少Shader复杂度
   → 降低帧率（30fps vs 60fps）
```

### 9.2 PC端优化

```
PC端特别注意：
1. Shader变体数量
   → PC硬件差异大，需要更多变体适配
   → 使用Shader变体剥离减少不必要的变体

2. 驱动开销
   → 使用SRP Batcher减少驱动瓶颈
   → 减少状态切换

3. 高端特效
   → 光线追踪有专用硬件（RT Core）
   → 使用DXR/Vulkan RT而非Shader近似
```

## 10. 性能优化检查清单

```
Shader优化检查清单：

□ ALU优化
  □ 避免pow()，用x*x替代
  □ 使用rsqrt替代1/sqrt
  □ 使用mad融合指令
  □ 避免不必要的normalize()

□ 精度优化（移动端）
  □ 颜色用half/mediump
  □ 矩阵保持float/highp
  □ UV用half/mediump

□ 分支优化
  □ 用mix/lerp替代if-else
  □ 用step/smoothstep替代比较
  □ 使用multi_compile变体替代运行时分支

□ 纹理优化
  □ 使用压缩纹理格式
  □ 打包单通道纹理为RGBA
  □ 启用Mipmap
  □ 减少纹理采样次数

□ 过度绘制
  □ 不透明从前到远渲染
  □ 使用Z-Pre Pass
  □ 减少半透明面积

□ 变体管理
  □ 使用shader_feature代替multi_compile
  □ 剥离未使用的变体
  □ 控制变体总数 < 200
```

## 11. 实际使用案例

- 《王者荣耀》移动端大量使用纹理打包、half精度计算和简化光照模型保证60fps
- VR游戏特别注重过度绘制控制，因为VR需要同时渲染两只眼睛（2倍像素量）
- 《原神》针对不同机型使用Shader变体：高端用完整效果（PBR+SSR+Bloom），低端用简化版（Lambert+无后处理）
- Unity的URP通过SRP Batcher和Shader变体系统实现了高效的跨平台Shader管理
- 《赛博朋克2077》的RTX模式与光栅化模式使用完全不同的Shader变体集
- GPU Profiling工具（RenderDoc、PIX、Xcode GPU Debugger）是定位Shader瓶颈的必备工具
- 移动端《和平精英》使用ASTC纹理压缩 + half精度 + 简化PCF阴影，将Shader开销控制在2ms以内
