# LOD层次细节

## 核心概念

LOD（Level of Detail）根据物体与相机的距离切换不同精度的模型，远处使用低面数模型以减少渲染开销。这是3D游戏中最常用的优化技术之一，几乎所有3A游戏都使用LOD系统。

### LOD级别策略详解

| LOD级别 | 使用距离 | 面数比例 | 典型用途 |
|---------|---------|---------|---------|
| LOD0 | 0-10m | 100% | 近距离最高精度 |
| LOD1 | 10-30m | 50% | 中距离简化 |
| LOD2 | 30-60m | 25% | 远距离低精度 |
| LOD3/Culled | 60m+ | 0%或Billboard | 完全不渲染或用广告牌 |

距离阈值应根据游戏类型调整：
- FPS游戏：需要更远的LOD切换距离（玩家会仔细观察远处）
- RTS/RPG：可以更激进地切换LOD（视角较高，细节不明显）

### CrossFade过渡

LOD切换时的硬切会产生视觉跳变（popping）。CrossFade通过混合两个LOD级别的透明度实现平滑过渡：

```csharp
// LOD Group设置CrossFade
LODGroup group = GetComponent<LODGroup>();
group.fadeMode = LODFadeMode.CrossFade; // 或 SpeedTree（用于植被）
group.animateCrossFading = true;
```

CrossFade需要Shader支持，使用LOD Blend参数控制混合权重：
```hlsl
// 在Shader中支持LOD混合
float lodBlend = unity_LODFade.x; // 从LOD Group传入
// 使用dither混合而非alpha混合（避免排序问题）
float dither = DitherPattern(screenPos);
clip(lodBlend - dither);
```

### HLOD（Hierarchical LOD）

HLOD适用于大世界场景，将远处多个小物体合并为一个简化模型：
- UE中通过HLOD Volume自动生成
- 将一组Actor的网格烘焙为单个简化网格
- 大幅减少远处场景的Draw Call（100个物体 → 1个HLOD网格）

### Impostor技术

将3D物体在多个角度预渲染为2D图像（类似Billboard但更精确）：
- 存储8-32个角度的法线贴图和颜色贴图
- 运行时根据视角选择最近的角度
- 极低LOD时使用Impostor代替3D模型

## 具体实现方法

### Unity LOD Group完整配置

```csharp
/// <summary>
/// 代码方式配置LOD Group
/// 适用于程序化生成的物体
/// </summary>
public class LODSetup : MonoBehaviour
{
    [System.Serializable]
    public class LODConfig
    {
        public float transitionHeight; // 屏幕高度比例（0-1）
        public Renderer[] renderers;
    }

    [SerializeField] private LODConfig[] lodConfigs;
    [SerializeField] private bool useCrossFade = true;

    void Start()
    {
        SetupLODGroup();
    }

    void SetupLODGroup()
    {
        LODGroup group = gameObject.GetComponent<LODGroup>();
        if (group == null)
            group = gameObject.AddComponent<LODGroup>();

        LOD[] lods = new LOD[lodConfigs.Length];
        for (int i = 0; i < lodConfigs.Length; i++)
        {
            lods[i] = new LOD(lodConfigs[i].transitionHeight,
                lodConfigs[i].renderers);
        }

        group.SetLODs(lods);
        group.fadeMode = useCrossFade
            ? LODFadeMode.CrossFade
            : LODFadeMode.None;
        group.animateCrossFading = useCrossFade;
        group.RecalculateBounds();
    }
}

/// <summary>
/// LOD模型顶点数参考规范
/// </summary>
public class LODModelSpec
{
    // LOD0: 100% (原始模型，如角色15000面)
    // LOD1: 50% (7500面)
    // LOD2: 25% (3750面)
    // LOD3: 10% (1500面)
    // LOD4: Billboard (1-2个三角形)

    // 自动生成工具：
    // - Simplygon（业界标准，高质量减面）
    // - InstaLOD（集成在UE中）
    // - Unity Mesh Simplifier（开源）
    // - Blender Decimate修改器（免费）
}
```

### 自定义LOD管理系统

```csharp
/// <summary>
/// 自定义LOD管理器
/// 比Unity LOD Group更灵活，支持动态调整切换距离
/// 和批量管理（适合大量相同物体如植被）
/// </summary>
public class CustomLODManager : MonoBehaviour
{
    public static CustomLODManager Instance { get; private set; }

    [System.Serializable]
    public class LODLevel
    {
        public GameObject model;
        public float maxDistance;
    }

    [Header("全局LOD配置")]
    [SerializeField] private LODLevel[] globalLODLevels;

    [Header("性能自适应")]
    [SerializeField] private bool dynamicLOD = true;
    [SerializeField] private float targetFrameTime = 16.67f;

    private Camera targetCamera;
    private List<LODObject> registeredObjects = new List<LODObject>();

    void Awake()
    {
        Instance = this;
        targetCamera = Camera.main;
    }

    void LateUpdate()
    {
        if (targetCamera == null) return;

        // 自适应LOD：根据帧率调整切换距离
        if (dynamicLOD)
        {
            float frameTime = Time.unscaledDeltaTime * 1000f;
            float adjustFactor = frameTime > targetFrameTime ? 0.9f : 1.01f;
            for (int i = 0; i < globalLODLevels.Length; i++)
                globalLODLevels[i].maxDistance *= adjustFactor;
        }

        // 更新所有注册物体的LOD级别
        Vector3 camPos = targetCamera.transform.position;
        foreach (var obj in registeredObjects)
        {
            if (obj == null) continue;
            float dist = Vector3.Distance(camPos, obj.transform.position);
            UpdateLOD(obj, dist);
        }
    }

    void UpdateLOD(LODObject obj, float distance)
    {
        int targetLevel = -1;
        for (int i = 0; i < globalLODLevels.Length; i++)
        {
            if (distance <= globalLODLevels[i].maxDistance)
            {
                targetLevel = i;
                break;
            }
        }

        if (targetLevel == obj.CurrentLevel) return;

        // 切换LOD
        if (obj.CurrentLevel >= 0 && obj.CurrentLevel < globalLODLevels.Length)
            globalLODLevels[obj.CurrentLevel].model?.SetActive(false);

        if (targetLevel >= 0 && targetLevel < globalLODLevels.Length)
        {
            globalLODLevels[targetLevel].model?.SetActive(true);
            obj.CurrentLevel = targetLevel;
        }
        else
        {
            obj.CurrentLevel = -1; // 完全剔除
        }
    }

    public void Register(LODObject obj) => registeredObjects.Add(obj);
    public void Unregister(LODObject obj) => registeredObjects.Remove(obj);
}

public class LODObject : MonoBehaviour
{
    public int CurrentLevel { get; set; } = 0;

    void OnEnable() => CustomLODManager.Instance?.Register(this);
    void OnDisable() => CustomLODManager.Instance?.Unregister(this);
}
```

### Billboard LOD实现

```csharp
/// <summary>
/// Billboard LOD - 最远距离的替代方案
/// 始终面向相机的2D精灵，常用于远处的树
/// </summary>
public class BillboardLOD : MonoBehaviour
{
    [SerializeField] private SpriteRenderer spriteRenderer;
    [SerializeField] private bool lockYAxis = true; // 锁定Y轴旋转（树只水平旋转）

    private Camera mainCamera;

    void Start()
    {
        mainCamera = Camera.main;
    }

    void LateUpdate()
    {
        if (mainCamera == null) return;

        Vector3 direction = mainCamera.transform.position - transform.position;

        if (lockYAxis)
            direction.y = 0;

        if (direction != Vector3.zero)
            transform.rotation = Quaternion.LookRotation(direction);
    }
}
```

### LOD预渲染Impostor方案

```csharp
/// <summary>
/// Impostor生成器（编辑器工具）
/// 在多个角度预渲染物体为2D纹理
/// </summary>
#if UNITY_EDITOR
public class ImpostorBaker : EditorWindow
{
    [MenuItem("Tools/Impostor Baker")]
    static void Open() => GetWindow<ImpostorBaker>();

    private GameObject targetObject;
    private int angleCount = 8; // 8个角度（水平）
    private int textureSize = 512;

    void OnGUI()
    {
        targetObject = (GameObject)EditorGUILayout.ObjectField(
            "Target", targetObject, typeof(GameObject), true);
        angleCount = EditorGUILayout.IntSlider("Angles", angleCount, 4, 32);
        textureSize = EditorGUILayout.IntPopup("Texture Size", textureSize,
            new[] { "256", "512", "1024" },
            new[] { 256, 512, 1024 });

        if (GUILayout.Button("Bake Impostor"))
            BakeImpostor();
    }

    void BakeImpostor()
    {
        // 创建临时相机
        // 在每个角度渲染物体到RenderTexture
        // 合并为Impostor图集
        // 自动生成Billboard材质和Shader
    }
}
#endif
```

## 性能基准数据

| 场景 | 无LOD | 3级LOD | 3级LOD+Billboard | 性能提升 |
|------|-------|--------|-----------------|---------|
| 1000棵树(近) | 500 DC, 8ms | 500 DC, 8ms | 500 DC, 8ms | 无 |
| 1000棵树(中) | 500 DC, 8ms | 200 DC, 3ms | 200 DC, 3ms | 2.5x |
| 1000棵树(远) | 500 DC, 8ms | 50 DC, 0.5ms | 10 DC, 0.1ms | 16x-80x |
| 10000棵草 | 10000 DC | 2000 DC | 200 DC | 5x-50x |
| 内存增加 | 0 | +50%模型 | +60%模型+纹理 | - |

## 最佳实践

- LOD切换距离根据游戏类型调整：FPS需要更远的切换距离（2-3倍）
- 使用CrossFade或Dither过渡避免LOD切换时的视觉跳变
- 确保各级LOD的包围盒一致，避免LOD切换时阴影/碰撞异常
- 开放世界中对植被使用HLOD或GPU Instancing+LOD
- 远处使用Billboard代替3D模型（如远处的树、灌木）
- LOD模型制作时保持大致轮廓一致，避免切换时形变明显
- 使用LOD Group的SpeedTree fade mode处理植被过渡

## 常见陷阱与修复

**陷阱1：LOD级别之间面数差异太小**
- 症状：3级LOD分别减少5%、10%的面数，优化效果不明显
- 修复：LOD1至少减少50%面数，LOD2至少减少75%

**陷阱2：忘记设置LOD Group的Bounds**
- 症状：LOD切换判断基于错误的包围盒，导致切换时机不对
- 修复：调用`group.RecalculateBounds()`确保Bounds正确

**陷阱3：阴影投射在LOD切换时出现闪烁**
- 症状：LOD0和LOD1的阴影形状不同，切换时阴影闪变
- 修复：为阴影设置独立的LOD（Shadow LOD），始终使用LOD0投射阴影

**陷阱4：HLOD烘焙时丢失了碰撞体和触发器**
- 症状：远处物体失去碰撞
- 修复：碰撞体不参与HLOD合并，保留原始碰撞体

**陷阱5：LOD切换距离硬编码**
- 症状：低端设备帧率低但LOD切换距离不变，远处细节仍在渲染
- 修复：使用CustomLODManager的dynamicLOD功能，根据帧率自动调整
