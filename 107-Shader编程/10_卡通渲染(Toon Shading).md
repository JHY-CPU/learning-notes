# 卡通渲染（Toon Shading）

## 1. 核心理论

卡通渲染（Cel Shading / Toon Shading）将连续的光照结果离散化为有限的色阶，产生手绘动画风格的画面效果。其核心哲学是"用非真实感技术（NPR, Non-Photorealistic Rendering）模拟手绘质感"，追求的是艺术表现力而非物理正确性。

### 1.1 色阶化的数学原理

```
标准Lambert漫反射：I = max(N·L, 0)
  → 连续值 [0, 1]，光照平滑渐变

半兰伯特（Half Lambert）：I = N·L × 0.5 + 0.5
  → 值域 [0, 1]，但阴影区域更亮，避免全黑
  → Valve在《半条命2》中发明，成为卡通渲染的标准基础

色阶量化（Quantization）：
  连续值 [0, 1] → 离散值 {0, 0.33, 0.67, 1.0}（4级）
  公式：Q(v, levels) = floor(v × levels) / levels
  例如：Q(0.7, 3) = floor(2.1) / 3 = 2/3 = 0.667
```

## 2. Ramp贴图（色阶控制）

Ramp贴图是一维或二维渐变纹理，用于将连续的漫反射值映射为离散的色阶，比硬编码的量化更灵活可控。

### 2.1 原理

```
Ramp贴图（水平方向1D渐变）：

UV坐标:  0.0       0.33      0.67      1.0
         |         |         |         |
颜色:    深色  |  暗色  |  中间色 |  亮色
         ■■■■■■■■|■■■■■■■■|■■■■■■■■|■■■■■■■■

采样方式：
  halfLambert = dot(N, L) * 0.5 + 0.5    // [0, 1]
  rampColor = texture(rampTex, vec2(halfLambert, 0.5)).rgb
  finalColor = baseColor × rampColor

2D Ramp贴图可以控制不同区域的阴影色：
  U轴 = 光照强度
  V轴 = 材质ID（皮肤用暖色阴影，金属用冷色阴影）
```

### 2.2 GLSL实现

```glsl
#version 330 core

uniform sampler2D u_RampTex;
uniform sampler2D u_BaseTex;
uniform vec3 u_LightDir;

in vec3 v_Normal;
in vec2 v_TexCoord;

out vec4 fragColor;

vec3 ToonShading(vec3 baseColor)
{
    vec3 N = normalize(v_Normal);
    vec3 L = normalize(u_LightDir);

    // 半兰伯特
    float NdotL = dot(N, L) * 0.5 + 0.5;

    // Ramp采样（1D贴图，V坐标固定为0.5）
    vec3 rampColor = texture(u_RampTex, vec2(NdotL, 0.5)).rgb;

    return baseColor * rampColor;
}

void main()
{
    vec3 baseColor = texture(u_BaseTex, v_TexCoord).rgb;
    vec3 color = ToonShading(baseColor);
    fragColor = vec4(color, 1.0);
}
```

### 2.3 HLSL实现（Unity URP）

```hlsl
half4 ToonFrag(Varyings input) : SV_Target
{
    half4 baseColor = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv);
    half3 N = normalize(input.normalWS);
    Light mainLight = GetMainLight();

    // 半兰伯特
    half halfLambert = dot(N, mainLight.direction) * 0.5 + 0.5;

    // Ramp贴图采样
    half3 rampColor = SAMPLE_TEXTURE2D(_RampTex, sampler_RampTex,
                        float2(halfLambert, 0.5)).rgb;

    half3 diffuse = baseColor.rgb * rampColor * mainLight.color;

    return half4(diffuse, baseColor.a);
}
```

## 3. 描边（Outline）

描边是卡通渲染最标志性的特征。最常用的方法是在背面渲染一个稍大的轮廓。

### 3.1 法线外扩描边

```
原理：
1. 第一个Pass（正面）：正常渲染物体
2. 第二个Pass（背面）：
   - 将顶点沿法线方向外扩一定距离
   - 渲染为纯色（通常黑色）
   - 因为是背面，外扩的部分从正面可见 → 形成描边

外扩公式：
expandedPos = originalPos + normal × outlineWidth

观察空间外扩（更好）：
expandedPosVS = viewPos + viewNormal × outlineWidth
// 在观察空间外扩，描边宽度不随距离变化
```

### 3.2 GLSL实现

```glsl
// ============ 描边Pass（背面渲染）============
// 顶点着色器
#version 330 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

uniform mat4 u_MVP;
uniform mat4 u_View;
uniform mat4 u_Model;
uniform float u_OutlineWidth;
uniform vec3 u_OutlineColor;

out vec3 v_OutlineColor;

void main()
{
    // 方法1：世界空间外扩
    vec3 worldPos = (u_Model * vec4(a_Position, 1.0)).xyz;
    vec3 worldNormal = normalize(mat3(transpose(inverse(u_Model))) * a_Normal);
    vec3 expandedPos = worldPos + worldNormal * u_OutlineWidth;
    gl_Position = u_MVP * vec4(expandedPos, 1.0);

    // 方法2（推荐）：观察空间外扩 — 描边宽度不受距离影响
    // vec4 viewPos = u_View * u_Model * vec4(a_Position, 1.0);
    // vec3 viewNormal = normalize(mat3(u_View) * mat3(transpose(inverse(u_Model))) * a_Normal);
    // viewPos.xyz += viewNormal * u_OutlineWidth;
    // gl_Position = u_Projection * viewPos;

    // 方法3（最优）：屏幕空间外扩 — 描边宽度在屏幕空间恒定
    // vec4 clipPos = u_MVP * vec4(a_Position, 1.0);
    // vec3 clipNormal = normalize((u_MVP * vec4(a_Normal, 0.0)).xyz);
    // clipPos.xy += clipNormal.xy * u_OutlineWidth * clipPos.w / resolution;
    // gl_Position = clipPos;

    v_OutlineColor = u_OutlineColor;
}

// 片元着色器 — 输出纯色
out vec4 fragColor;
in vec3 v_OutlineColor;

void main()
{
    fragColor = vec4(v_OutlineColor, 1.0);
}
```

### 3.3 HLSL实现（Unity URP）

```hlsl
// 描边Pass
Pass
{
    Name "Outline"
    Cull Front    // 剔除正面，只渲染背面
    ZWrite On

    HLSLPROGRAM
    #pragma vertex OutlineVert
    #pragma fragment OutlineFrag

    float _OutlineWidth;
    half4 _OutlineColor;

    Varyings OutlineVert(Attributes input)
    {
        Varyings output;
        // 沿法线外扩
        float3 expandedPos = input.positionOS.xyz +
                             input.normalOS * _OutlineWidth * 0.01;
        output.positionCS = TransformObjectToHClip(expandedPos);
        return output;
    }

    half4 OutlineFrag(Varyings input) : SV_TARGET
    {
        return _OutlineColor;
    }
    ENDHLSL
}
```

## 4. 边缘光（Rim Light）

边缘光在模型与背景交界处添加发光效果，增强角色轮廓感。

```glsl
// Rim Light计算
uniform vec3 u_RimColor;
uniform float u_RimPower;   // 控制边缘光宽度
uniform float u_RimIntensity;

vec3 CalcRimLight(vec3 normal, vec3 viewDir)
{
    // 边缘 = 法线与视线垂直的区域
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);
    rim = pow(rim, u_RimPower);  // 指数控制衰减

    // 只在亮面显示边缘光（与半兰伯特结合）
    float halfLambert = dot(normal, u_LightDir) * 0.5 + 0.5;
    rim *= step(0.5, halfLambert);  // 只在亮区显示

    return u_RimColor * rim * u_RimIntensity;
}
// rimPower = 2-4 适合卡通风格（宽边缘光）
// rimPower = 6-8 适合写实风格（窄边缘光）
```

```hlsl
// Unity HLSL Rim Light
half CalcRimLight(half3 normalWS, half3 viewDirWS, half NdotL)
{
    half rim = 1.0 - saturate(dot(normalWS, viewDirWS));
    rim = pow(rim, _RimPower);

    // 只在亮面显示
    rim *= step(0.5, NdotL * 0.5 + 0.5);
    return rim;
}

half3 rimLight = _RimColor.rgb * rim * _RimIntensity;
```

## 5. 高光处理

卡通渲染中的高光通常是硬边的、形状明确的，与写实渲染的柔和高光不同。

```glsl
// 卡通风格硬高光
vec3 CalcToonSpecular(vec3 normal, vec3 lightDir, vec3 viewDir,
                       float specularSize, float specularSmoothness)
{
    vec3 halfDir = normalize(lightDir + viewDir);
    float NdotH = max(dot(normal, halfDir), 0.0);

    // 硬高光：step或smoothstep产生锐利边缘
    float spec = smoothstep(1.0 - specularSize - specularSmoothness,
                            1.0 - specularSize + specularSmoothness,
                            NdotH);

    // 或使用贴图控制高光形状
    // float specTex = texture(u_SpecularMap, v_UV).r;
    // spec *= specTex;

    return vec3(spec);
}
```

## 6. 色阶化（Color Quantization）

```glsl
// 硬色阶化
float Quantize(float value, int levels)
{
    return floor(value * float(levels)) / float(levels);
}

// 带过渡的色阶化（更自然）
float SmoothQuantize(float value, int levels, float smoothness)
{
    float quantized = floor(value * float(levels)) / float(levels);
    float next = min(quantized + 1.0 / float(levels), 1.0);
    float blend = smoothstep(0.0, smoothness, fract(value * float(levels)));
    return mix(quantized, next, blend);
}
```

## 7. 完整卡通着色器

### 7.1 GLSL完整版

```glsl
#version 330 core

uniform sampler2D u_BaseTex;
uniform sampler2D u_RampTex;
uniform sampler2D u_SpecularMap;

uniform vec3 u_LightDir;
uniform vec3 u_LightColor;
uniform vec3 u_CameraPos;
uniform vec3 u_RimColor;
uniform float u_RimPower;
uniform vec3 u_ShadowColor;

in vec3 v_Normal;
in vec3 v_WorldPos;
in vec2 v_TexCoord;

out vec4 fragColor;

void main()
{
    vec3 baseColor = texture(u_BaseTex, v_TexCoord).rgb;
    vec3 N = normalize(v_Normal);
    vec3 L = normalize(u_LightDir);
    vec3 V = normalize(u_CameraPos - v_WorldPos);
    vec3 H = normalize(L + V);

    // 1. 半兰伯特 + Ramp
    float halfLambert = dot(N, L) * 0.5 + 0.5;
    vec3 rampColor = texture(u_RampTex, vec2(halfLambert, 0.5)).rgb;
    vec3 diffuse = baseColor * rampColor * u_LightColor;

    // 2. 硬高光
    float NdotH = max(dot(N, H), 0.0);
    float specMask = texture(u_SpecularMap, v_TexCoord).r;
    float specular = smoothstep(0.95, 0.98, NdotH) * specMask;
    vec3 specColor = u_LightColor * specular;

    // 3. 边缘光
    float rim = pow(1.0 - max(dot(N, V), 0.0), u_RimPower);
    rim *= step(0.5, halfLambert); // 只在亮面
    vec3 rimLight = u_RimColor * rim;

    // 4. 环境色（暗部用自定义阴影色而非纯黑）
    float inShadow = 1.0 - step(0.5, halfLambert);
    vec3 ambient = u_ShadowColor * baseColor * inShadow * 0.3;

    vec3 finalColor = diffuse + specColor + rimLight + ambient;
    fragColor = vec4(finalColor, 1.0);
}
```

### 7.2 HLSL完整版（Unity URP）

```hlsl
Shader "Custom/ToonShading"
{
    Properties
    {
        _BaseMap ("Base Texture", 2D) = "white" {}
        _RampTex ("Ramp Texture", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1, 1, 1, 1)
        _ShadowColor ("Shadow Color", Color) = (0.3, 0.3, 0.5, 1)
        _OutlineWidth ("Outline Width", Range(0, 0.1)) = 0.02
        _OutlineColor ("Outline Color", Color) = (0, 0, 0, 1)
        _RimColor ("Rim Color", Color) = (1, 1, 1, 1)
        _RimPower ("Rim Power", Range(0.5, 8)) = 3
        _SpecularSize ("Specular Size", Range(0, 1)) = 0.95
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        // Pass 1：描边
        Pass
        {
            Name "Outline"
            Cull Front
            ZWrite On

            HLSLPROGRAM
            #pragma vertex OutlineVert
            #pragma fragment OutlineFrag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            CBUFFER_START(UnityPerMaterial)
                float _OutlineWidth;
                half4 _OutlineColor;
            CBUFFER_END

            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; };
            struct Varyings { float4 positionCS : SV_POSITION; };

            Varyings OutlineVert(Attributes v) {
                Varyings o;
                float3 expanded = v.positionOS.xyz + v.normalOS * _OutlineWidth * 0.01;
                o.positionCS = TransformObjectToHClip(expanded);
                return o;
            }
            half4 OutlineFrag(Varyings i) : SV_Target { return _OutlineColor; }
            ENDHLSL
        }

        // Pass 2：卡通着色
        Pass
        {
            Name "ToonForward"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex ToonVert
            #pragma fragment ToonFrag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                half4 _ShadowColor;
                half4 _RimColor;
                half _RimPower;
                half _SpecularSize;
            CBUFFER_END

            TEXTURE2D(_BaseMap);    SAMPLER(sampler_BaseMap);
            TEXTURE2D(_RampTex);    SAMPLER(sampler_RampTex);

            struct Attributes {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD0;
            };
            struct Varyings {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float3 normalWS : TEXCOORD1;
                float2 uv : TEXCOORD2;
            };

            Varyings ToonVert(Attributes v) {
                Varyings o;
                o.positionCS = TransformObjectToHClip(v.positionOS.xyz);
                o.positionWS = TransformObjectToWorld(v.positionOS.xyz);
                o.normalWS = TransformObjectToWorldNormal(v.normalOS);
                o.uv = v.uv;
                return o;
            }

            half4 ToonFrag(Varyings i) : SV_Target {
                half4 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv) * _BaseColor;
                half3 N = normalize(i.normalWS);
                half3 V = normalize(_WorldSpaceCameraPos - i.positionWS);
                Light mainLight = GetMainLight();

                // 半兰伯特 + Ramp
                half halfLambert = dot(N, mainLight.direction) * 0.5 + 0.5;
                half3 ramp = SAMPLE_TEXTURE2D(_RampTex, sampler_RampTex,
                                float2(halfLambert, 0.5)).rgb;
                half3 diffuse = baseColor.rgb * ramp * mainLight.color;

                // 硬高光
                half3 H = normalize(mainLight.direction + V);
                half spec = smoothstep(_SpecularSize - 0.02, _SpecularSize + 0.02,
                                saturate(dot(N, H)));

                // 边缘光
                half rim = pow(1.0 - saturate(dot(N, V)), _RimPower);
                rim *= step(0.5, halfLambert);
                half3 rimLight = _RimColor.rgb * rim;

                // 阴影色
                half inShadow = 1.0 - step(0.5, halfLambert);
                half3 shadow = _ShadowColor.rgb * baseColor.rgb * inShadow * 0.3;

                return half4(diffuse + spec * mainLight.color + rimLight + shadow, 1.0);
            }
            ENDHLSL
        }
    }
}
```

## 8. 高级技术

### 8.1 Cross Hatching（交叉阴影线）

```glsl
// 交叉阴影线 — 用线条密度表示阴影深度
float HatchingPattern(vec2 uv, float angle, float density)
{
    float s = sin(angle), c = cos(angle);
    vec2 rotUV = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
    return smoothstep(0.4, 0.5, abs(fract(rotUV.x * density) - 0.5));
}

vec3 CrossHatch(vec3 baseColor, float light, vec2 uv)
{
    float hatch1 = HatchingPattern(uv, 0.785, 20.0);   // 45度线
    float hatch2 = HatchingPattern(uv, -0.785, 20.0);  // -45度线

    // 根据光照强度选择线密度
    if (light < 0.3)
        return baseColor * min(hatch1, hatch2); // 交叉阴影（最暗）
    else if (light < 0.6)
        return baseColor * hatch1; // 单向阴影线
    else
        return baseColor; // 无线条（最亮）
}
```

### 8.2 等高线（Height Contours）

```glsl
// 用等高线替代色阶化
float ContourLines(float value, float lineCount, float lineWidth)
{
    float lines = fract(value * lineCount);
    return smoothstep(0.0, lineWidth, lines) * smoothstep(lineWidth * 2.0, lineWidth, lines);
}
```

## 9. 性能提示

- 卡通渲染比PBR轻量：无需菲涅尔、G项、D项等复杂BRDF计算
- 描边Pass增加了一个额外Draw Call，但顶点着色器计算简单
- Ramp贴图是性能友好的色阶控制方式（1次纹理采样替代多次if判断）
- 多材质角色需要多个Ramp贴图（按材质区域采样不同行）
- 边缘光的`pow()`可以用查找表（LUT）替代

## 10. 常见问题与调试

**问题1：描边断裂**
- 模型法线不连续处（UV接缝、硬边）会导致外扩不连续
- 解决：在建模工具中软化法线，或使用平滑法线组

**问题2：描边宽度随距离变化**
- 使用观察空间外扩代替世界空间外扩
- 或使用屏幕空间外扩（clip space操作）

**问题3：Ramp贴图接缝**
- 确保Ramp贴图的Wrap模式为Clamp而非Repeat
- 检查UV坐标是否精确在[0,1]范围内

**问题4：阴影区域出现色带（Banding）**
- 在阴影色中添加噪声抖动（Dither）
- 或使用更多的Ramp贴图色阶

**问题5：边缘光在多光源下过于明亮**
- 只对主光源应用边缘光
- 或将边缘光强度除以光源数量

## 11. 实际使用案例

- 《罪恶装备 Xrd》和《龙珠斗士Z》使用极高质量的卡通渲染，成为2D动画3D化的行业标杆
- 《原神》使用卡通渲染 + Ramp贴图 + 自定义阴影 + 描边实现二次元风格
- 《崩坏：星穹铁道》在卡通基础上叠加SSR和Bloom增强画面表现力
- 《塞尔达传说：风之律者》开创了经典的卡通渲染风格
- 《街头霸王6》使用风格化的卡通渲染结合面部表情系统
- Arc System Works分享了其卡通渲染技术的详细GDC演讲，是学习的最佳资源
