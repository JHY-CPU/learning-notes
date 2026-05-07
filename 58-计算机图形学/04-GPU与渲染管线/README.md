# 04 - GPU 与渲染管线

> 现代 GPU 是图形渲染的核心引擎，理解渲染管线和 GPU 架构是掌握实时图形学的关键。

---

## 1. 图形渲染管线概述

### 1.1 管线四个阶段

图形渲染管线将 3D 场景描述转化为 2D 屏幕图像：

```
应用程序阶段 → 几何阶段 → 光栅化阶段 → 像素处理阶段
 (Application)  (Geometry)  (Rasterization) (Pixel Processing)
```

| 阶段 | 主要任务 | 运行位置 |
|------|---------|---------|
| 应用程序 | 场景管理、剔除、状态设置 | CPU |
| 几何处理 | 顶点变换、投影、曲面细分 | GPU（顶点/曲面/几何着色器） |
| 光栅化 | 图元转为片段（Fragment） | GPU 固定硬件 |
| 像素处理 | 深度测试、混合、写入帧缓冲 | GPU（片段着色器 + ROP） |

### 1.2 数据流动

```
顶点数据 → 顶点着色器 → (曲面细分) → (几何着色器) → 光栅化 → 片段着色器 → 帧缓冲
```

---

## 2. 顶点着色器（Vertex Shader）

### 2.1 职责

每个顶点执行一次，主要完成：
- **模型变换**：将顶点从模型空间变换到世界空间
- **观察变换**：变换到相机空间
- **投影变换**：变换到裁剪空间
- 输出插值数据（纹理坐标、法线、颜色等）

### 2.2 输入与输出

```glsl
// GLSL 示例
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
```

### 2.3 注意事项

- 顶点着色器不能创建或销毁顶点
- 执行次数 = 顶点数（注意共享顶点的处理）
- 输出变量会被光栅化阶段自动插值

---

## 3. 曲面细分（Tessellation）

### 3.1 三个子阶段

1. **曲面细分控制着色器（TCS）**：决定细分程度（细分级别）
2. **曲面细分器（固定硬件）**：生成细分后的顶点
3. **曲面细分评估着色器（TES）**：计算新顶点的位置

### 3.2 应用场景

- **LOD（细节层级）**：近处细分程度高，远处低
- **置换贴图**：细分后沿法线方向偏移顶点
- **减少带宽**：传输少量控制点，GPU 端细分

---

## 4. 几何着色器（Geometry Shader）

### 4.1 功能

以图元（点、线、三角形）为单位输入，可输出零个或多个图元：
- 可以创建新顶点（区别于顶点着色器）
- 可以改变图元类型（如将点扩展为四边形）

### 4.2 典型应用

- 粒子系统（点→四边形）
- 阴影体积生成
- 轮廓线渲染
- GPU 端布料模拟

### 4.3 性能警告

几何着色器的输出数量不确定，可能成为性能瓶颈。现代 GPU 中建议优先使用 Compute Shader 或曲面细分替代。

---

## 5. 片段着色器（Fragment Shader）

### 5.1 职责

每个片段（候选像素）执行一次：
- 纹理采样
- 光照计算
- 阴影判断
- 输出片段颜色

### 5.2 GLSL 示例

```glsl
in vec2 TexCoord;
in vec3 Normal;

out vec4 FragColor;

uniform sampler2D diffuseMap;
uniform vec3 lightDir;

void main() {
    vec3 color = texture(diffuseMap, TexCoord).rgb;
    float diff = max(dot(normalize(Normal), normalize(lightDir)), 0.0);
    FragColor = vec4(color * diff, 1.0);
}
```

### 5.3 片段着色器之后

- 深度测试（Z-Test）
- 模板测试（Stencil Test）
- 混合（Blending）
- 写入帧缓冲

---

## 6. GPU 架构

### 6.1 统一着色器架构

现代 GPU 使用统一的处理单元（Shader Core / Streaming Multiprocessor）：
- 同一硬件可以执行顶点/片段/计算着色器
- 根据负载动态分配着色器类型
- 最大化硬件利用率

### 6.2 SIMT 执行模型

**SIMT（Single Instruction, Multiple Thread）**：
- 多个线程（通常 32 个，称为 Warp/Wavefront）同时执行相同指令
- 每个线程有自己的寄存器和数据
- 分支发散（Branch Divergence）会降低效率

**Warp 发散**：同一 Warp 内的线程走不同分支时，未执行的线程被屏蔽，串行执行各分支。

### 6.3 内存层次

```
寄存器（最快，每线程私有）
  ↓
共享内存/L1 缓存（线程组共享）
  ↓
L2 缓存（全局共享）
  ↓
显存/全局内存（最慢，容量最大）
```

**优化原则**：最大化寄存器和共享内存的使用，最小化全局内存访问，保证内存访问合并（Coalesced Access）。

---

## 7. 图形 API 对比

### 7.1 OpenGL

- 由 Khronos 维护的跨平台 API
- 状态机模型，API 简洁易学
- 驱动层较厚，性能开销相对大
- 现代 OpenGL（3.3+）采用可编程管线

### 7.2 Vulkan

- Khronos 的下一代 API
- 显式控制：手动管理内存、同步、命令缓冲
- 多线程友好，CPU 开销更低
- 学习曲线陡峭，代码量大

### 7.3 DirectX 12

- 微软的底层图形 API
- 仅限 Windows/Xbox 平台
- 与 Vulkan 类似的显式设计
- 优秀的工具链（PIX、Visual Studio 图形调试）

### 7.4 Metal

- Apple 的图形 API
- macOS / iOS 独占
- 针对 Apple 芯片深度优化
- 工具链完善（Xcode GPU Debugger）

### 对比总结

| 特性 | OpenGL | Vulkan | DirectX 12 | Metal |
|------|--------|--------|------------|-------|
| 平台 | 跨平台 | 跨平台 | Windows/Xbox | Apple |
| 抽象层级 | 高 | 低 | 低 | 低 |
| 多线程 | 弱 | 强 | 强 | 强 |
| 学习难度 | 低 | 高 | 高 | 中 |

---

## 8. Shader 编程基础

### 8.1 GLSL（OpenGL Shading Language）

```glsl
#version 450 core

// 顶点着色器
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
```

### 8.2 HLSL（High-Level Shading Language）

```hlsl
// DirectX 使用
struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float3 worldPos : TEXCOORD0;
    float3 normal : TEXCOORD1;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.worldPos = mul(model, float4(input.position, 1.0)).xyz;
    output.normal = mul((float3x3)transpose(inverse(model)), input.normal);
    output.position = mul(viewProjection, float4(output.worldPos, 1.0));
    return output;
}
```

### 8.3 关键概念

- **Uniform**：CPU 传递给 GPU 的常量数据
- **Varying / In/Out**：着色器阶段之间传递的插值数据
- **采样器（Sampler）**：控制纹理采样的过滤和寻址方式
- **UBO / SSBO**：Uniform Buffer Object / Shader Storage Buffer Object

---

## 9. 延迟渲染（Deferred Rendering）

### 9.1 流程

分为两个 Pass：

**几何 Pass（G-Buffer 填充）**：
- 渲染场景到多张缓冲（G-Buffer）：位置、法线、反照率、粗糙度等
- 不计算光照

**光照 Pass**：
- 对每个光源，在屏幕空间逐像素计算光照
- 仅读取 G-Buffer 数据

### 9.2 G-Buffer 内容

| 附件 | 存储内容 | 格式 |
|------|---------|------|
| Albedo | 颜色 | RGB8 |
| Normal | 世界空间法线 | RGB10A2 |
| Position/Depth | 位置或深度 | RGBA32F / D24S8 |
| Roughness/Metallic | 材质属性 | RG8 |

### 9.3 优缺点

**优点**：
- 光照计算与场景复杂度解耦
- 多光源场景效率高

**缺点**：
- G-Buffer 内存带宽大
- 透明物体处理困难（需要额外的前向渲染 Pass）
- 不支持 MSAA（或需要特殊处理）

---

## 10. 前向渲染 vs 延迟渲染

| 特性 | 前向渲染 | 延迟渲染 |
|------|---------|---------|
| 流程 | 逐物体逐光照 | G-Buffer + 屏幕空间光照 |
| 多光源 | 瓶颈（物体×光源） | 高效（屏幕像素×光源） |
| 透明 | 容易 | 需额外处理 |
| MSAA | 原生支持 | 需特殊处理 |
| 内存 | 较小 | G-Buffer 较大 |
| 带宽 | 较低 | G-Buffer 读写较重 |

**Forward+（Tiled Forward Shading）**：结合两者优点，将屏幕分块，每块筛选影响的光源后再计算光照。

---

## 11. PBR（基于物理的渲染）

### 11.1 核心原则

- **能量守恒**：反射光不会超过入射光
- **微表面理论**：粗糙表面由大量微小镜面组成
- **Fresnel 效应**：掠射角反射增强

### 11.2 Cook-Torrance BRDF

```
f_r = (kd/π) + (D·F·G) / (4·(n·ωo)·(n·ωi))
```

- **D**（法线分布函数）：GGX/Trowbridge-Reitz 最常用
- **F**（Fresnel）：Schlick 近似
- **G**（几何遮蔽函数）：Smith 模型

### 11.3 PBR 材质参数

| 参数 | 说明 |
|------|------|
| Albedo/Base Color | 基础颜色（不含光照信息） |
| Metallic | 金属度（0=绝缘体，1=金属） |
| Roughness | 粗糙度（0=光滑，1=粗糙） |
| AO（环境光遮蔽） | 微小缝隙的阴影 |
| Emissive | 自发光 |

---

## 12. WebGL 基础

### 12.1 概述

WebGL 是基于 OpenGL ES 的浏览器端 3D 图形 API：
- WebGL 1.0 对应 OpenGL ES 2.0（GLSL ES 100）
- WebGL 2.0 对应 OpenGL ES 3.0（GLSL ES 300）

### 12.2 基本流程

```javascript
// 获取上下文
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl2');

// 编译着色器
const vertexShader = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vertexShader, vertexSource);
gl.compileShader(vertexShader);

// 链接程序
const program = gl.createProgram();
gl.attachShader(program, vertexShader);
gl.attachShader(program, fragmentShader);
gl.linkProgram(program);
gl.useProgram(program);

// 设置顶点数据
const buffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

// 绘制
gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
```

### 12.3 注意事项

- 坐标系 Y 轴向上（与屏幕坐标相反）
- 纹理坐标 (0,0) 在左下角（与 HTML 图像不同）
- 矩阵以列主序存储
- 浮点纹理支持有限，PBR 需注意精度

---

*本章涵盖了从渲染管线到 GPU 架构、从图形 API 到着色器编程的完整知识体系，是理解和实现现代图形渲染的基础。*
