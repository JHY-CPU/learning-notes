# Draw Call合批技术

## 核心概念

Draw Call是CPU向GPU发送渲染指令的次数。每次Draw Call都有CPU端的开销（准备渲染状态、绑定材质、提交顶点数据），通常每个Draw Call约消耗0.01-0.1ms的CPU时间。在复杂场景中可能有数千个渲染对象，如果每个都单独提交Draw Call，CPU将成为瓶颈。合批技术的核心思想是将多个渲染对象合并为少量Draw Call提交，从而降低CPU负担。

### 各种合批技术对比

| 技术 | 适用对象 | 合批条件 | CPU开销 | 内存开销 | 限制 |
|------|---------|---------|--------|---------|------|
| Static Batching | 静态物体 | 共享材质 | 零(预计算) | 高(合并网格) | 不能移动 |
| Dynamic Batching | 动态物体 | 顶点<300, 共享材质 | 中等(每帧合并) | 低 | 仅简单物体 |
| GPU Instancing | 相同网格实例 | 共享材质+Shader支持 | 低 | 低 | 需Shader支持 |
| SRP Batcher | URP/HDRP物体 | 相同Shader变体 | 极低 | 低 | 需规范Shader |

### Static Batching（静态合批）深度解析

Static Batching在构建时/加载时将共享材质的静态物体网格合并为一个大网格：

**工作原理**：
```
构建前: 物体A(材质X) + 物体B(材质X) + 物体C(材质Y) = 3个Draw Call
构建后: 合并网格AB(材质X) + 物体C(材质Y) = 2个Draw Call
```

**条件**：
- 物体必须标记为Static（Inspector中勾选Batching Static）
- 共享相同的材质（相同Material实例）
- 使用的Shader必须相同

**代价**：合并后的网格数据在内存中保留，可能导致内存占用增加2-10倍。

### Dynamic Batching（动态合批）深度解析

Dynamic Batching在运行时自动将符合条件的小物体合并：

**条件（非常严格）**：
- 顶点数 <= 300（WebGL上为200）
- 使用相同材质和材质属性
- 相同的Shader变体
- 不使用镜像变换（scale不能有负值）
- 使用不同材质属性的子Mesh不能合批
- 多Pass的Shader不能合批

**适用场景**：小道具、粒子、2D精灵。

### GPU Instancing深度解析

GPU Instancing允许单次Draw Call绘制同一网格的多个实例，每个实例可以有不同的位置、旋转、缩放甚至颜色：

**工作原理**：
```
传统方式: N个相同物体 = N个Draw Call
Instancing: N个相同物体 = 1个Draw Call + 实例数据数组
```

**条件**：
- 材质开启Enable GPU Instancing
- Shader支持Instancing（使用UNITY_INSTANCING_BUFFER）
- 实例间只有少量属性差异（位置、颜色等）

### SRP Batcher深度解析

SRP Batcher是URP/HDRP特有的合批优化。它不合并网格，而是优化材质数据的上传流程：

**工作原理**：
```
传统: 每个Draw Call都要重新上传材质属性到GPU
SRP Batcher: 批量上传材质属性CBUFFER，Draw Call只切换绑定的CBUFFER
```

**条件**：
- 使用URP或HDRP渲染管线
- Shader必须使用CBUFFER包裹所有属性
- 相同Shader变体的物体可合批

## 具体实现方法

### 开启Static Batching

```csharp
// Player Settings中确保开启
// Edit > Project Settings > Player > Other Settings
// Static Batching: 勾选（默认开启）
// Dynamic Batching: 根据需要勾选（移动端建议关闭，开销大收益小）

// 代码中标记物体为Static
void MarkStatic()
{
    // 标记所有子物体为静态
    foreach (var renderer in GetComponentsInChildren<Renderer>())
    {
        renderer.gameObject.isStatic = true;
    }

    // 注意：isStatic是一个位掩码，标记为true意味着所有静态选项都开启
    // 如果只想标记Batching Static，需要使用StaticEditorFlags
#if UNITY_EDITOR
    GameObjectUtility.SetStaticEditorFlags(
        gameObject, StaticEditorFlags.Batching);
#endif
}
```

### GPU Instancing完整配置

```csharp
// === C#端设置Instancing属性 ===
/// <summary>
/// 为大量相同物体设置不同属性
/// 使用MaterialPropertyBlock避免创建多个Material实例
/// </summary>
public class InstancedSpawner : MonoBehaviour
{
    [SerializeField] private GameObject prefab;
    [SerializeField] private int count = 1000;
    [SerializeField] private float spreadRadius = 50f;

    void Start()
    {
        MaterialPropertyBlock props = new MaterialPropertyBlock();
        Renderer[] renderers = new Renderer[count];

        for (int i = 0; i < count; i++)
        {
            Vector3 pos = Random.insideUnitSphere * spreadRadius;
            GameObject go = Instantiate(prefab, pos, Random.rotation);
            renderers[i] = go.GetComponent<Renderer>();

            // 使用MaterialPropertyBlock设置不同颜色
            // 不会创建新Material实例，保持合批
            props.SetColor("_BaseColor", Random.ColorHSV());
            renderers[i].SetPropertyBlock(props);
        }

        // 关键：所有实例共享同一材质（不要在代码中new Material）
        // MaterialPropertyBlock只修改着色器参数，不创建新材质
    }
}
```

```hlsl
// === Shader端支持Instancing ===
Shader "Custom/InstancedColor"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1,1,1,1)
        _Smoothness ("Smoothness", Range(0,1)) = 0.5
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" }

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing // 关键：启用Instancing变体

            #include "UnityCG.cginc"

            // 实例化属性缓冲区
            UNITY_INSTANCING_BUFFER_START(Props)
                UNITY_DEFINE_INSTANCED_PROP(fixed4, _BaseColor)
                UNITY_DEFINE_INSTANCED_PROP(float, _Smoothness)
            UNITY_INSTANCING_BUFFER_END(Props)

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                UNITY_VERTEX_INPUT_INSTANCE_ID // 关键：实例ID
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            v2f vert(appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v); // 关键：设置实例ID
                UNITY_TRANSFER_INSTANCE_ID(v, o);

                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_INSTANCE_ID(i);
                fixed4 color = UNITY_ACCESS_INSTANCED_PROP(Props, _BaseColor);
                float smoothness = UNITY_ACCESS_INSTANCED_PROP(Props, _Smoothness);

                // 简单光照
                float ndl = max(0, dot(i.worldNormal, _WorldSpaceLightPos0.xyz));
                return color * (0.2 + 0.8 * ndl) * smoothness;
            }
            ENDCG
        }
    }
}
```

### SRP Batcher兼容Shader

```hlsl
// URP中Shader必须使用CBUFFER包裹所有属性才能兼容SRP Batcher
Shader "Custom/SRPBatcherCompatible"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1,1,1,1)
        _BaseMap ("Base Map", 2D) = "white" {}
        _Smoothness ("Smoothness", Range(0,1)) = 0.5
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            // 关键：必须使用CBUFFER包裹所有属性
            CBUFFER_START(UnityPerMaterial)
                float4 _BaseColor;
                float4 _BaseMap_ST;
                float _Smoothness;
            CBUFFER_END

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings vert(Attributes input)
            {
                Varyings o;
                o.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                o.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                return o;
            }

            half4 frag(Varyings input) : SV_Target
            {
                half4 tex = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, input.uv);
                return tex * _BaseColor;
            }
            ENDHLSL
        }
    }
}
```

### 合批分析脚本（完整版）

```csharp
/// <summary>
/// 合批分析工具
/// 帮助识别合批打断原因
/// </summary>
public class BatchAnalyzer : MonoBehaviour
{
    [SerializeField] private KeyCode analyzeKey = KeyCode.F1;
    [SerializeField] private bool logEveryFrame = false;

    private int lastBatches;
    private int lastSetPass;

    void Update()
    {
        if (Input.GetKeyDown(analyzeKey) || logEveryFrame)
        {
            Analyze();
        }
    }

    void Analyze()
    {
#if UNITY_EDITOR
        int batches = UnityEditor.UnityStats.batches;
        int setPass = UnityEditor.UnityStats.setPassCalls;
        int tris = UnityEditor.UnityStats.triangles;
        int verts = UnityEditor.UnityStats.vertices;

        Debug.Log("=== Batch Analysis ===");
        Debug.Log($"Batches: {batches} (delta: {batches - lastBatches})");
        Debug.Log($"SetPass Calls: {setPass} (delta: {setPass - lastSetPass})");
        Debug.Log($"Triangles: {tris}");
        Debug.Log($"Vertices: {verts}");

        // 分析Canvas合批
        Canvas[] canvases = FindObjectsOfType<Canvas>();
        foreach (var canvas in canvases)
        {
            if (!canvas.gameObject.activeInHierarchy) continue;
            int graphics = canvas.GetComponentsInChildren<Graphic>(true).Length;
            Debug.Log($"  Canvas[{canvas.sortingOrder}] {canvas.name}: {graphics} graphics");
        }

        lastBatches = batches;
        lastSetPass = setPass;
#endif
    }
}
```

## 性能基准数据

| 场景 | 无合批 | Static Batching | GPU Instancing | SRP Batcher |
|------|--------|----------------|----------------|-------------|
| 1000个相同石头 | 1000 DC | 1 DC | 1 DC | ~10 DC |
| 10000棵草 | 10000 DC | 不适用(太多) | 1 DC | ~50 DC |
| 100个不同材质UI | 100 DC | 不适用 | 不适用 | 不适用 |
| 内存增加 | 0 | +200MB | 0 | 0 |
| CPU合批开销 | 0 | 0(预计算) | 0.1ms | 0.05ms |

## 最佳实践

- 场景中的静态物体标记为Batching Static（墙壁、地形、建筑）
- 使用图集将多个小纹理合并，使UI元素共享材质以实现合批
- 大量相同物体（草、树、弹壳、粒子）使用GPU Instancing
- URP项目确保所有自定义Shader兼容SRP Batcher（使用CBUFFER）
- 减少材质变体数量，相同Shader不同参数使用MaterialPropertyBlock而非创建新Material
- 使用Frame Debugger逐Draw Call分析合批打断原因
- 移动端关闭Dynamic Batching（开销大于收益）
- UI使用图集确保同Canvas的元素共享材质

## 常见陷阱与修复

**陷阱1：静态合批增加内存**
- 症状：1000个石头的合并网格占用200MB内存
- 修复：只对确实需要合批的物体标记Static，大量重复物体改用GPU Instancing

**陷阱2：动态合批对顶点数有严格限制**
- 症状：标记了Dynamic但没有生效
- 修复：检查顶点数是否超过300，检查是否有镜像scale（负值）

**陷阱3：GPU Instancing需要Shader支持**
- 症状：勾选了Enable Instancing但没有减少Draw Call
- 修复：检查Shader是否有`#pragma multi_compile_instancing`和`UNITY_INSTANCING_BUFFER`

**陷阱4：SRP Batcher要求Shader结构规范**
- 症状：自定义Shader不兼容SRP Batcher
- 修复：所有属性必须在CBUFFER中，使用HLSL而非CG

**陷阱5：不同图集的UI元素交替出现打断合批**
- 症状：UI Draw Call数量远超预期
- 修复：按图集组织UI元素，确保同图集元素连续渲染
