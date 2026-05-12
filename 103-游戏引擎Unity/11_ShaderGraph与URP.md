# ShaderGraph与URP

## 核心概念

ShaderGraph是Unity的可视化Shader编辑工具，无需编写HLSL代码即可创建着色器。URP（Universal Render Pipeline）是Unity的通用渲染管线，替代内置渲染管线，针对移动端和中端硬件优化。URP基于Scriptable Render Pipeline（SRP）架构，允许通过C#代码完全控制渲染流程。

## URP渲染管线架构

```
URP渲染流程:
Camera
  ↓
Setup Camera Properties (设置相机参数)
  ↓
Renderer Features (自定义Pass注入点)
  ↓
Culling (视锥体剔除)
  ↓
Shadow Caster Pass (生成阴影贴图)
  ↓
Depth Prepass (深度预渲染，可选)
  ↓
Opaque Objects (不透明物体渲染，Forward+路径)
  ↓
Skybox / Reflection Probes
  ↓
Transparent Objects (透明物体，从后往前排序)
  ↓
Post Processing (后处理: Bloom, Tonemapping, etc.)
  ↓
Final Blit to Screen

URP vs Built-in对比:
- URP使用Forward+渲染（非Deferred），每个物体受光照影响通过Light Volume计算
- URP最大像素光源数有限（默认4个额外光源），需合理使用
- URP不支持实时全局光照（需使用Lightmap或探针）
```

## URP配置详解

### 创建和配置URP项目

```csharp
// URP项目配置步骤:
// 1. 通过Package Manager安装Universal RP包 (com.unity.render-pipelines.universal)
// 2. 创建URP Asset: Assets -> Create -> Rendering -> URP Asset (with Universal Renderer)
// 3. 在Graphics Settings中设置Scriptable Render Pipeline Settings
// 4. 创建Renderer Data并添加到URP Asset
// 5. 将现有材质球的Shader替换为URP兼容的Shader

// 运行时切换渲染管线配置
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class URPSettingsManager : MonoBehaviour
{
    [SerializeField] private UniversalRenderPipelineAsset[] qualityProfiles;

    public void SetQualityProfile(int index)
    {
        if (index >= 0 && index < qualityProfiles.Length)
        {
            GraphicsSettings.defaultRenderPipeline = qualityProfiles[index];
        }
    }

    public void AdjustRuntimeSettings()
    {
        var urp = GraphicsSettings.defaultRenderPipeline as UniversalRenderPipelineAsset;
        if (urp != null)
        {
            urp.renderScale = 0.8f;        // 渲染分辨率缩放（降采样提升性能）
            urp.shadowDistance = 80f;        // 阴影渲染距离
            urp.shadowCascadeCount = 3;      // 级联阴影数量
        }
    }
}
```

### URP Renderer Feature详解

Renderer Feature是扩展URP渲染管线的核心机制：

```csharp
// 自定义Renderer Feature：全屏灰度效果
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class GrayscaleRenderFeature : ScriptableRendererFeature
{
    // 渲染Pass
    class GrayscalePass : ScriptableRenderPass
    {
        private Material material;
        private RenderTargetIdentifier source;
        private RenderTargetHandle tempTexture;

        public GrayscalePass(Material mat)
        {
            material = mat;
            tempTexture.Init("_TemporaryColorTexture");
        }

        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            // 配置此Pass需要的Render Target
            var descriptor = renderingData.cameraData.cameraTargetDescriptor;
            descriptor.depthBufferBits = 0;
            cmd.GetTemporaryRT(tempTexture.id, descriptor);
        }

        public override void Execute(ScriptableRenderContext context,
            ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("GrayscaleEffect");

            // 获取相机的颜色缓冲
            RenderTargetIdentifier cameraColorTarget = renderingData.cameraData.renderer
                .cameraColorTarget;

            // 全屏Blit应用效果
            Blit(cmd, cameraColorTarget, tempTexture.Identifier(), material);
            Blit(cmd, tempTexture.Identifier(), cameraColorTarget);

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }

        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            cmd.ReleaseTemporaryRT(tempTexture.id);
        }
    }

    [System.Serializable]
    public class Settings
    {
        public Material grayscaleMaterial;
        public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing;
    }

    public Settings settings = new Settings();
    private GrayscalePass pass;

    public override void Create()
    {
        pass = new GrayscalePass(settings.grayscaleMaterial);
        pass.renderPassEvent = settings.renderPassEvent;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer,
        ref RenderingData renderingData)
    {
        if (settings.grayscaleMaterial != null)
            renderer.EnqueuePass(pass);
    }
}

// 另一个常见的Feature: Outlines/描边效果
public class OutlineRenderFeature : ScriptableRendererFeature
{
    class OutlinePass : ScriptableRenderPass
    {
        private Material outlineMaterial;
        private int outlineLayerMask;

        public OutlinePass(Material mat, LayerMask mask)
        {
            outlineMaterial = mat;
            outlineLayerMask = mask;
        }

        public override void Execute(ScriptableRenderContext context,
            ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("OutlinePass");

            // 只渲染指定Layer的物体到单独的RT
            var drawingSettings = CreateDrawingSettings(
                new ShaderTagId("UniversalForward"),
                ref renderingData,
                SortingCriteria.CommonOpaque
            );

            var filteringSettings = new FilteringSettings(RenderQueueRange.opaque,
                outlineLayerMask);

            context.DrawRenderers(renderingData.cullResults,
                ref drawingSettings, ref filteringSettings);

            // 使用后处理材质描边
            Blit(cmd, source, tempRT, outlineMaterial);
            Blit(cmd, tempRT, source);

            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
    }

    public override void Create() { }
    public override void AddRenderPasses(ScriptableRenderer renderer,
        ref RenderingData renderingData) { }
}
```

## ShaderGraph节点系统

### 节点分类与常用节点

| 节点类型 | 功能 | 典型用途 |
|----------|------|---------|
| Property | 外部可调参数 | Color, Float, Texture2D, Vector |
| Math | 数学运算 | Add, Multiply, Lerp, Power, Saturate |
| Input | 输入数据 | UV, Position, Normal, Time, View Direction |
| Artistic | 美术效果 | Blend, Gradient, Color Adjust, Hue Shift |
| Procedural | 程序化图案 | Checker, Noise, Voronoi, Brick |
| Channel | 通道操作 | Split, Append, Swizzle, Flip |
| UV | UV操作 | Tiling, Offset, Polar Coordinates, Rotate |
| Utility | 工具节点 | Preview, Keyword, Custom Function |

### 常用ShaderGraph效果

**溶解效果(Dissolve)**:
```
节点图:
Property(_NoiseTex) → Texture2D Sample
Property(_DissolveAmount) → Remap(0,1 → 0,1)
Noise → Add(DissolveAmount)
         ↓
    Step(0.5, result) → Alpha Clip
         ↓
    Lerp(ColorA, ColorB, step_result) → Base Color
```

**水面效果(Water)**:
```
节点图:
UV → Panner(Time * Speed_A) → NormalMap_A
UV → Panner(Time * Speed_B, 0.7倍) → NormalMap_B
NormalMap_A + NormalMap_B → Blend(Overlay) → Normal Strength → Normal Output

Time → Sine → Remap(0,1 → -0.1,0.1) → Vertex Offset Y

Fresnel Effect → Lerp(DeepColor, ShallowColor, fresnel) → Base Color
Scene Depth → Depth Fade → Alpha
```

**边缘发光(Rim Light)**:
```
节点图:
Normal(World Space) → Dot Product(View Direction) → One Minus → Power(Exponent)
    → Saturate → Multiply(GlowColor) → Add(Base Color) → Base Color Output
```

**地形混合(Terrain Blend)**:
```
节点图:
Blend Map(R通道) → Split
R通道 → Lerp(Grass, Dirt) → Result1
G通道 → Lerp(Result1, Rock) → Result2
B通道 → Lerp(Result2, Snow) → Base Color
```

### Custom Function Node (自定义函数节点)

```hlsl
// 在ShaderGraph中使用Custom Function Node嵌入HLSL代码

// 方法1: 内联代码
// Name: FresnelEffect
// Type: String
// 输入: float3 WorldNormal, float3 ViewDir, float Power
// 输出: float Out
Out = pow(1.0 - saturate(dot(normalize(WorldNormal), normalize(ViewDir))), Power);

// 方法2: 引用外部HLSL文件
// 在ShaderGraph的Custom Function节点中选择File模式
// 指定.hlsl文件路径

// MyCustomFunctions.hlsl内容:
void TriplanarMapping_float(
    float3 WorldPosition,
    float3 WorldNormal,
    float Tile,
    float Blend,
    UnityTexture2D TopTex,
    UnityTexture2D SideTex,
    UnitySamplerState Sampler,
    out float4 Out)
{
    float3 weights = abs(WorldNormal);
    weights = pow(weights, Blend);
    weights /= (weights.x + weights.y + weights.z);

    float2 uvX = WorldPosition.yz * Tile;
    float2 uvY = WorldPosition.xz * Tile;
    float2 uvZ = WorldPosition.xy * Tile;

    float4 topColor = SAMPLE_TEXTURE2D(TopTex, Sampler, uvY);
    float4 sideColorX = SAMPLE_TEXTURE2D(SideTex, Sampler, uvX);
    float4 sideColorZ = SAMPLE_TEXTURE2D(SideTex, Sampler, uvZ);

    Out = topColor * weights.y + sideColorX * weights.x + sideColorZ * weights.z;
}
```

```csharp
// 运行时修改ShaderGraph参数
public class ShaderGraphController : MonoBehaviour
{
    private MaterialPropertyBlock mpb;
    private Renderer rend;
    private static readonly int DissolveAmount = Shader.PropertyToID("_DissolveAmount");
    private static readonly int BaseColor = Shader.PropertyToID("_BaseColor");
    private static readonly int EmissionColor = Shader.PropertyToID("_EmissionColor");

    void Start()
    {
        rend = GetComponent<Renderer>();
        mpb = new MaterialPropertyBlock();
    }

    void Update()
    {
        // 使用MaterialPropertyBlock不创建材质实例
        // 同一材质的不同对象可以有不同的参数值
        mpb.SetFloat(DissolveAmount, Mathf.PingPong(Time.time, 1f));
        mpb.SetColor(BaseColor, Color.Lerp(Color.white, Color.red, dissolveProgress));

        // 设置自发光
        mpb.SetColor(EmissionColor, Color.yellow * Mathf.Sin(Time.time) * 0.5f + 0.5f);

        rend.SetPropertyBlock(mpb);
    }

    // MaterialPropertyBlock vs 材质实例对比:
    // PropertyBlock: 不创建新材质实例，节省内存，适合少量参数变化
    // 材质实例: renderer.material = new Material(shader)，每个对象独立材质
    // 注意: 修改renderer.material会创建实例（名为instance），原始材质不变
}
```

## URP Shader编写要点

URP使用不同于内置管线的Shader语法和约定：

```hlsl
// URP Unlit Shader基础模板
Shader "Custom/URPUnlit"
{
    Properties
    {
        _BaseMap ("Base Texture", 2D) = "white" {}
        _BaseColor ("Base Color", Color) = (1,1,1,1)
        _Cutoff ("Alpha Cutoff", Range(0,1)) = 0.5
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
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float fogFactor : TEXCOORD1;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                half4 _BaseColor;
                half _Cutoff;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output;
                UNITY_SETUP_INSTANCE_ID(input);
                output.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                output.fogFactor = ComputeFogFactor(output.positionCS.z);
                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                half4 texColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv);
                half3 color = texColor.rgb * _BaseColor.rgb;
                color = MixFog(color, input.fogFactor);
                return half4(color, texColor.a);
            }
            ENDHLSL
        }

        // Shadow Caster Pass（投射阴影）
        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On
            ZTest LEqual
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex ShadowVert
            #pragma fragment ShadowFrag
            #include "Packages/com.unity.render-pipelines.universal/Shaders/ShadowCasterPass.hlsl"
            ENDHLSL
        }

        // Depth Only Pass（深度预渲染）
        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode" = "DepthOnly" }

            ZWrite On
            ColorMask 0

            HLSLPROGRAM
            #pragma vertex DepthVert
            #pragma fragment DepthFrag
            #include "Packages/com.unity.render-pipelines.universal/Shaders/DepthOnlyPass.hlsl"
            ENDHLSL
        }
    }

    // 回退到内置管线Shader
    Fallback "Unlit/Texture"
}
```

### URP内置属性映射

| 内置管线属性 | URP属性 | 说明 |
|-------------|---------|------|
| _MainTex | _BaseMap | 基础纹理 |
| _Color | _BaseColor | 基础颜色 |
| _BumpMap | _BumpMap | 法线贴图（相同） |
| _Metallic | _Metallic | 金属度 |
| _Glossiness | _Smoothness | 光滑度 |
| _EmissionMap | _EmissionMap | 自发光贴图 |
| _Cutoff | _Cutoff | Alpha裁剪阈值 |

## 后处理Volume系统

URP通过Volume系统实现后处理效果：

```csharp
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// 自定义后处理效果
[System.Serializable, VolumeComponentMenu("Custom/VHS Effect")]
public class VHSPostProcess : VolumeComponent, IPostProcessComponent
{
    [Tooltip("VHS效果强度")]
    public ClampedFloatParameter intensity = new ClampedFloatParameter(0f, 0f, 1f);

    [Tooltip("扫描线频率")]
    public ClampedFloatParameter scanlineFrequency = new ClampedFloatParameter(100f, 10f, 500f);

    [Tooltip("色差偏移")]
    public ClampedFloatParameter chromaticAberration = new ClampedFloatParameter(0f, 0f, 0.1f);

    [Tooltip("噪点强度")]
    public ClampedFloatParameter noiseIntensity = new ClampedFloatParameter(0.1f, 0f, 1f);

    public bool IsActive() => intensity.value > 0f;
    public bool IsTileCompatible() => false; // 需要全屏采样
}

// 对应的Renderer Feature和Pass
public class VHSRenderPass : ScriptableRenderPass
{
    private VHSPostProcess vhsEffect;
    private Material vhsMaterial;

    public override void Execute(ScriptableRenderContext context,
        ref RenderingData renderingData)
    {
        var stack = VolumeManager.instance.stack;
        vhsEffect = stack.GetComponent<VHSPostProcess>();

        if (vhsEffect == null || !vhsEffect.IsActive()) return;

        CommandBuffer cmd = CommandBufferPool.Get("VHS Effect");

        vhsMaterial.SetFloat("_Intensity", vhsEffect.intensity.value);
        vhsMaterial.SetFloat("_ScanlineFreq", vhsEffect.scanlineFrequency.value);
        vhsMaterial.SetFloat("_ChromaticAberration", vhsEffect.chromaticAberration.value);
        vhsMaterial.SetFloat("_NoiseIntensity", vhsEffect.noiseIntensity.value);

        var source = renderingData.cameraData.renderer.cameraColorTarget;
        Blit(cmd, source, source, vhsMaterial);

        context.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
    }
}
```

## Shader变体管理

Shader变体爆炸是URP项目的常见性能问题：

```csharp
// Shader变体预编译
public class ShaderVariantPreloader : MonoBehaviour
{
    [SerializeField] private ShaderVariantCollection[] variantCollections;

    void Start()
    {
        // 预热Shader变体（避免运行时编译卡顿）
        foreach (var collection in variantCollections)
        {
            collection.WarmUp();
        }
    }
}

// 关键字(Keyword)管理
public class KeywordManager : MonoBehaviour
{
    // Shader中定义的多编译关键字
    // #pragma multi_compile _ _RAIN_ON
    // #pragma shader_feature _SNOW_ON

    public void EnableRainEffect(bool enable)
    {
        // 设置全局Shader关键字
        if (enable)
            Shader.EnableKeyword("_RAIN_ON");
        else
            Shader.DisableKeyword("_RAIN_ON");
    }

    // 注意: 每个关键字组合生成一个变体
    // 3个multi_compile关键字 = 2^3 = 8个变体
    // 变体数量过多会导致:
    // 1. 内存占用增大
    // 2. Shader编译时间增长
    // 3. 运行时WarmUp卡顿
}
```

## 常见陷阱与最佳实践

1. **内置Shader不兼容URP**: 迁移项目需要将所有Shader替换为URP版本，可用Edit -> Rendering -> Materials -> Convert
2. **ShaderGraph变体爆炸**: 过多的条件分支和关键字会导致Shader变体数量激增，使用shader_feature替代multi_compile
3. **移动端优化**: 减少纹理采样次数，避免复杂数学运算，使用half精度浮点（URP默认使用half）
4. **MaterialPropertyBlock vs 材质实例**: 修改少量对象用PropertyBlock节省内存，大量不同修改用材质实例
5. **Shader预编译**: 使用Shader Variant Collection预编译常用变体，避免运行时首次使用卡顿
6. **渲染路径选择**: URP默认Forward+，如需更多光源考虑Deferred（URP 2022.2+支持）
7. **SRP Batcher兼容**: 确保Shader使用CBUFFER块包装材质属性，以启用SRP Batcher优化

## 性能分析

| 操作 | 开销 | 说明 |
|------|------|------|
| Shader编译（首次） | 高 | 使用Variant预编译避免 |
| MaterialPropertyBlock | 极低 | 不创建材质实例 |
| Renderer Feature | 低中 | 每个Feature增加一个Pass |
| 全屏后处理 | 中 | 取决于分辨率和效果复杂度 |
| Shader变体数量 | 高（间接） | 变体多导致内存和加载开销大 |
| Texture Sample | 低（单次） | 移动端限制4-8次采样 |

## 与其他系统的关联

- **后处理**: URP通过Volume系统实现后处理效果（Bloom、Tone Mapping、Vignette等）
- **光照**: URP使用Forward+渲染，光照数量和质量与Built-in不同
- **粒子系统**: 粒子Shader需使用URP兼容的Particles Shader或自定义ShaderGraph
- **2D渲染**: URP提供2D Renderer，支持2D光照和阴影
- **Shader Stripping**: 打包时会自动剥离未使用的Shader变体，减小包体
