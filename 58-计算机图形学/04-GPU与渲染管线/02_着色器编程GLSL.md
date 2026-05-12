# 着色器编程GLSL


## 着色器编程GLSL

一、GLSL概述

## 一、GLSL概述


GLSL（OpenGL Shading Language）是OpenGL的着色器编程语言，语法类似C语言，在GPU上并行执行。现代图形渲染管线的每个可编程阶段都由对应的着色器控制。


### 1.1 可编程着色器阶段


| 着色器 | 管线阶段 | 执行频率 | 主要职责 |
| --- | --- | --- | --- |
| 顶点着色器（Vertex） | 顶点处理 | 每个顶点 | 坐标变换、属性传递 |
| 细分控制着色器（TCS） | 细分 | 每个控制点 | 控制细分程度 |
| 细分评估着色器（TES） | 细分 | 每个细分顶点 | 计算细分后顶点位置 |
| 几何着色器（Geometry） | 图元处理 | 每个图元 | 增删改图元 |
| 片段着色器（Fragment） | 光栅化后 | 每个像素 | 颜色计算、光照 |
| 计算着色器（Compute） | 独立 | 每个线程 | 通用GPU计算 |

二、GLSL语言基础

## 二、GLSL语言基础


### 2.1 数据类型


| 类型 | 说明 | 示例 |
| --- | --- | --- |
| float | 32位浮点 | `float a = 1.0;` |
| vec2/vec3/vec4 | 2/3/4维浮点向量 | `vec3 color = vec3(1.0, 0.5, 0.0);` |
| ivec2/ivec3/ivec4 | 整数向量 | `ivec2 pos = ivec2(10, 20);` |
| mat2/mat3/mat4 | 2×2/3×3/4×4矩阵 | `mat4 mvp = proj * view * model;` |
| sampler2D | 2D纹理采样器 | `sampler2D tex;` |
| bool | 布尔值 | `bool flag = true;` |


### 2.2 向量分量访问

`vec4 v = vec4(1.0, 2.0, 3.0, 4.0);`


`v.x / v.r / v.s`
→ 1.0（第一个分量）


`v.y / v.g / v.t`
→ 2.0（第二个分量）


`v.z / v.b / v.p`
→ 3.0（第三个分量）


`v.w / v.a / v.q`
→ 4.0（第四个分量）


支持swizzle：
`v.rgb`
= vec3(1,2,3),
`v.xy`
= vec2(1,2)

### 2.3 类型限定符


| 限定符 | 说明 | 示例 |
| --- | --- | --- |
| in | 输入变量 | `in vec3 position;` |
| out | 输出变量 | `out vec4 fragColor;` |
| uniform | 全局一致的常量 | `uniform mat4 mvp;` |
| varying（旧版） | 顶点→片段插值 | 已被in/out替代 |
| const | 编译时常量 | `const float PI = 3.14159;` |

三、顶点着色器

## 三、顶点着色器（Vertex Shader）


顶点着色器处理每个顶点，主要负责将顶点位置从模型空间变换到裁剪空间，同时传递顶点属性到后续阶段。


### 3.1 基本顶点着色器


```
#version 330 core

// 输入属性
layout (location = 0) in vec3 aPos;       // 顶点位置
layout (location = 1) in vec3 aNormal;    // 顶点法线
layout (location = 2) in vec2 aTexCoord;  // 纹理坐标

// Uniform变量
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// 输出到片段着色器
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

void main() {
    // 计算世界空间位置
    FragPos = vec3(model * vec4(aPos, 1.0));

    // 法线变换（使用法线矩阵处理非均匀缩放）
    Normal = mat3(transpose(inverse(model))) * aNormal;

    // 传递纹理坐标
    TexCoord = aTexCoord;

    // 裁剪空间位置
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
```

四、片段着色器

## 四、片段着色器（Fragment Shader）


片段着色器处理每个像素片段，计算最终输出颜色。光照、纹理采样、后处理效果等都在这里实现。


### 4.1 Phong光照片段着色器


```
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

out vec4 FragColor;

// 材质
uniform sampler2D diffuseTex;
uniform sampler2D specularTex;
uniform float shininess;

// 光源
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main() {
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfDir = normalize(lightDir + viewDir);

    // 环境光
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // 漫反射
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * texture(diffuseTex, TexCoord).rgb;

    // 镜面反射
    float spec = pow(max(dot(norm, halfDir), 0.0), shininess);
    vec3 specular = spec * lightColor * texture(specularTex, TexCoord).rgb;

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}
```

五、ShaderToy

## 五、ShaderToy示例


ShaderToy是一个在线着色器编写和分享平台，只需编写片段着色器，通过内置uniform变量获取输入。


### 5.1 ShaderToy内置变量


| 变量 | 类型 | 说明 |
| --- | --- | --- |
| iResolution | vec3 | 视口分辨率（x, y, 1） |
| iTime | float | 程序运行时间（秒） |
| iMouse | vec4 | 鼠标位置（x, y, 点击x, 点击y） |
| iChannel0~3 | sampler2D | 输入纹理 |
| fragCoord | vec4 | 片段像素坐标（输入） |
| fragColor | vec4 | 输出颜色 |


### 5.2 经典ShaderToy示例


```
// 渐变背景 + 动态圆
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // 归一化坐标到[-1, 1]
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // 动态背景渐变
    vec3 col = vec3(0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4)));

    // 中心圆
    float d = length(uv);
    float circle = smoothstep(0.31, 0.3, d);
    col = mix(col, vec3(1.0, 0.3, 0.1), circle);

    // 围绕中心运动的小球
    for (int i = 0; i < 5; i++) {
        float angle = iTime + float(i) * 6.28318 / 5.0;
        vec2 pos = 0.5 * vec2(cos(angle), sin(angle));
        float dist = length(uv - pos);
        float ball = smoothstep(0.06, 0.05, dist);
        col = mix(col, vec3(1.0), ball);
    }

    fragColor = vec4(col, 1.0);
}
```

六、常用GLSL函数

## 六、常用GLSL内置函数


| 函数 | 说明 | 示例 |
| --- | --- | --- |
| mix(a, b, t) | 线性插值 a*(1-t) + b*t | 颜色混合 |
| smoothstep(e0, e1, x) | 平滑阶跃（Hermite插值） | 平滑边缘 |
| step(edge, x) | 阶跃函数 x<edge→0, else→1 | 阈值判断 |
| clamp(x, min, max) | 钳制到[min,max] | 限制范围 |
| dot(a, b) | 点积 | 光照计算 |
| cross(a, b) | 叉积 | 法线计算 |
| normalize(v) | 归一化 | 方向向量 |
| length(v) | 向量长度 | 距离计算 |
| texture(sampler, uv) | 纹理采样 | 读取纹理颜色 |
| fract(x) | 取小数部分 | 重复纹理 |
| sin/cos/tan | 三角函数 | 动画、波形 |
| pow(x, y) | x的y次方 | 高光计算 |

========================================
  文件总结
========================================
  主题：着色器编程GLSL
  内容概要：
    1. GLSL概述 - 管线各阶段着色器：顶点/细分/几何/片段/计算
    2. 语言基础 - 数据类型(vec/mat/sampler)、类型限定符(in/out/uniform)
    3. 顶点着色器 - 坐标变换(MVP)，传递属性
    4. 片段着色器 - Phong光照模型实现
    5. ShaderToy - iResolution/iTime等内置变量
    6. 常用函数 - mix/smoothstep/dot/texture等
  重点知识：
    - gl_Position在顶点着色器中设置裁剪空间位置
    - 法线矩阵 = mat3(transpose(inverse(model)))处理非均匀缩放
    - smoothstep(edge0, edge1, x)产生平滑过渡
    - ShaderToy只需片段着色器，mainImage(out color, in coord)
    - uniform变量在CPU端设置，所有顶点/片段共享同一值
========================================


<!-- Converted from: 02_着色器编程GLSL.html -->
