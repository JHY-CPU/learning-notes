# PBR物理渲染


## PBR物理渲染

一、PBR概述

## 一、PBR（物理渲染）概述


PBR（Physically Based Rendering）是一种基于物理光学原理的渲染方法，通过精确模拟光线与材质的交互来产生真实感图像。它的核心目标是让材质在任何光照环境下都看起来正确。


### 1.1 PBR核心原则


- **能量守恒：**
   表面反射的光总和不会超过入射光
- **基于物理的BRDF：**
   使用微表面模型描述表面反射
- **Fresnel效应：**
   掠射角反射率增加
- **线性空间渲染：**
   所有计算在线性颜色空间进行
- **参数化材质：**
   使用物理可解释的参数（金属度、粗糙度）

二、微表面理论

## 二、微表面理论（Microfacet Theory）


微表面理论假设宏观光滑的表面在微观尺度上由大量微小平面组成，每个微表面是完美镜面反射。表面的粗糙度由微表面法线的分布决定。


### 2.1 Cook-Torrance BRDF

**Cook-Torrance BRDF：**


`f_r = k_d × (c/π) + k_s × (D × F × G) / (4 × cos(θ_i) × cos(θ_o))`


漫反射项：k_d × (c/π)（Lambert漫反射）


镜面反射项：D × F × G / (4 × cosθ_i × cosθ_o)


D = 法线分布函数（Normal Distribution Function）


F = Fresnel方程


G = 几何遮蔽函数（Geometry / Shadowing-Masking）


k_d + k_s = 1（能量守恒）
三、GGX法线分布

## 三、GGX / Trowbridge-Reitz NDF


GGX是目前最广泛使用的法线分布函数，由Walter等人推广，能产生更真实的高光衰减和长尾效果。


### 3.1 GGX公式

**GGX NDF：**


`D_GGX(h) = α² / (π × ((n·h)² × (α² - 1) + 1)²)`


α = roughness²（粗糙度参数需要平方变换）


h = 半角向量 = normalize(lightDir + viewDir)


n·h = 法线与半角向量的点积


α越大（越粗糙）→ 分布越平坦（高光越大越模糊）


α越小（越光滑）→ 分布越尖锐（高光越小越集中）

### 3.2 常见NDF对比


| NDF | 公式复杂度 | 效果特点 | 应用 |
| --- | --- | --- | --- |
| Beckmann | 指数函数 | 经典分布 | 早期PBR |
| GGX/TR | 有理函数 | 长尾衰减，更真实 | 主流方案 |
| GTR | 广义形式 | 可调节尾部 | Disney BRDF |

四、Fresnel方程

## 四、Fresnel方程


Fresnel效应描述了反射率随观察角度变化的现象：正对表面时反射率低，掠射角时反射率趋近100%。


### 4.1 Schlick近似

**Fresnel-Schlick近似：**


`F(θ) = F_0 + (1 - F_0) × (1 - cos(θ))⁵`


F_0 = 基础反射率（正对表面时的反射率）


θ = 观察方向与半角向量的夹角


非金属 F_0 ≈ 0.04（4%反射率）


金属 F_0 = 漫反射颜色（60~100%反射率）

### 4.2 不同材质的F_0值


| 材质 | F_0 (RGB) | 说明 |
| --- | --- | --- |
| 水 | (0.02, 0.02, 0.02) | 约2%反射率 |
| 玻璃 | (0.04, 0.04, 0.04) | 约4%反射率 |
| 塑料 | (0.04, 0.04, 0.04) | 非金属统一~4% |
| 金 | (1.0, 0.765, 0.336) | 高反射率，有色 |
| 铜 | (0.955, 0.638, 0.538) | 高反射率，有色 |
| 铁 | (0.560, 0.570, 0.580) | 中等反射率 |
| 铝 | (0.913, 0.922, 0.924) | 高反射率，接近白色 |

五、几何遮蔽函数

## 五、几何遮蔽函数（G函数）


G函数描述微表面之间的自遮挡效应：粗糙表面上，微表面会遮挡入射光或出射光，导致有效反射面积减小。


### 5.1 Schlick-GGX

**Schlick-GGX遮蔽函数：**


`G_SchlickGGX(n, v, k) = (n·v) / ((n·v) × (1 - k) + k)`


Smith方法将遮蔽和阴影分离：


`G(n, v, l, k) = G_SchlickGGX(n, v, k) × G_SchlickGGX(n, l, k)`


k = (roughness + 1)² / 8（直接光照）


k = roughness² / 2（IBL光照）
六、金属-粗糙度工作流

## 六、金属-粗糙度工作流


金属-粗糙度（Metallic-Roughness）工作流是最流行的PBR材质参数化方案，由Disney提出，被glTF标准采用。


### 6.1 贴图通道分配


| 贴图 | 通道 | 内容 | 范围 |
| --- | --- | --- | --- |
| 基础色（Albedo） | RGB | 漫反射颜色 / 金属F_0 | sRGB |
| 金属粗糙度 | R | — | — |
| 金属粗糙度 | G | 粗糙度（Roughness） | 线性 [0, 1] |
| 金属粗糙度 | B | 金属度（Metallic） | 线性 [0, 1] |
| 法线贴图 | RGB | 切线空间法线 | 线性 |
| 环境光遮蔽 | R | AO值 | 线性 [0, 1] |

**金属/非金属区别：**


metallic = 0（非金属）：


- 漫反射 = albedo颜色


- 镜面F_0 = 0.04（固定值）


metallic = 1（金属）：


- 漫反射 = 0（金属无漫反射）


- 镜面F_0 = albedo颜色（金属的F_0是彩色的）
七、IBL

## 七、基于图像的照明（IBL）


IBL使用环境贴图（HDRI）代替点光源/面光源来照明场景，是PBR渲染中环境光照的标准方案。


### 7.1 IBL漫反射


对环境贴图进行卷积（模糊），预计算每个方向的平均辐射度，存储在辐照度贴图（Irradiance Map）中。采样时直接查表。


### 7.2 IBL镜面反射

**Split-Sum近似：**


镜面IBL被拆分为两个预计算贴图：


1.
**预滤波环境贴图（Pre-filtered Env Map）：**


按粗糙度级别模糊的环境贴图（mipmap级别）


根据roughness选择mipmap级别采样


2.
**BRDF积分贴图（BRDF LUT）：**


2D查找表，输入(cosθ, roughness)，输出(scale, bias)


最终镜面IBL = prefilteredColor * (F0 * scale + bias)

### 7.3 PBR片段着色器


```
// PBR片段着色器 - IBL光照
vec3 PBR_IBL(vec3 N, vec3 V, vec3 albedo, float metallic,
             float roughness, float ao) {
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // 漫反射IBL
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * albedo;

    // 镜面IBL (Split-Sum)
    vec3 R = reflect(-V, N);
    vec3 prefilteredColor = textureLod(prefilterMap, R,
        roughness * MAX_REFLECTION_LOD).rgb;
    vec2 brdf = texture(brdfLUT, vec2(max(dot(N, V), 0.0),
        roughness)).rg;
    vec3 specular = prefilteredColor * (F0 * brdf.x + brdf.y);

    // 合并
    vec3 kS = F_Schlick(max(dot(N, V), 0.0), F0);
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    return (kD * diffuse + specular) * ao;
}
```

========================================
  文件总结
========================================
  主题：PBR物理渲染
  内容概要：
    1. 微表面理论 - D(法线分布) × F(Fresnel) × G(几何遮蔽) / (4cosθi×cosθo)
    2. GGX NDF - α²/(π((n·h)²(α²-1)+1)²)，长尾衰减
    3. Fresnel-Schlick - F=F0+(1-F0)(1-cosθ)⁵
    4. 金属-粗糙度工作流 - metallic控制漫反射/镜面分配
    5. IBL - Split-Sum近似：预滤波环境贴图 + BRDF LUT
  重点知识：
    - Cook-Torrance BRDF: fr = kd(c/π) + ks(D×F×G)/(4cosθi×cosθo)
    - α = roughness²（粗糙度需要平方变换）
    - 非金属F0≈0.04，金属F0=albedo颜色
    - metallic=1时无漫反射，镜面=albedo
    - IBL Split-Sum：预滤波贴图按roughness选mipmap级别
========================================


<!-- Converted from: 04_PBR物理渲染.html -->
