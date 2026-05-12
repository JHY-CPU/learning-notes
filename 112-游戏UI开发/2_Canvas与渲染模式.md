# Canvas与渲染模式

## 核心概念

Canvas是Unity UGUI的渲染根节点，其渲染模式直接决定了UI的坐标系、显示方式和性能特性。理解Canvas的内部工作机制是优化UI性能的前提。

### Canvas内部工作机制

Unity的UI渲染流程分为三个阶段：
1. **Layout（布局计算）**：计算所有RectTransform的最终位置和大小，触发条件包括位置、大小、锚点变化
2. **Rebuild（网格重建）**：将UI元素的视觉属性（颜色、纹理、文字内容）转化为顶点数据和三角形索引
3. **Batch（合批渲染）**：将相同材质/图集的UI元素合并为少数Draw Call提交给GPU

关键认知：Rebuild的范围是整个Canvas，而非单个元素。一个文本内容变化可能导致整个Canvas的网格重建。

### 三种渲染模式详解

**Screen Space - Overlay**
- UI直接覆盖在屏幕最上层，渲染在所有3D内容之后，不依赖任何Camera
- Canvas自动跟随屏幕分辨率变化（RectTransform的尺寸自动同步Screen.width/height）
- 像素完美对齐，不存在透视变形问题
- 适合：主菜单、HUD、大部分2D UI
- 缺点：无法被3D物体遮挡，无法实现UI嵌入3D场景，不支持后处理效果应用到UI上

**Screen Space - Camera**
- UI渲染在指定Camera的前方，距离由Plane Distance参数控制
- 可通过Camera的Culling Mask控制UI与其他物体的渲染关系
- UI会被后处理效果影响（Bloom、Color Grading等）
- Plane Distance决定UI与Camera的距离：
  - 值越小（如0.1），UI越靠近近裁面，透视变形越小，但可能被近裁面截断
  - 值越大（如10），UI在3D空间中占用更多深度，可能被物体穿插
- 适合：需要透视效果的UI、需要被3D物体遮挡的UI、需要后处理影响UI的项目

**World Space**
- Canvas作为普通GameObject存在于世界空间，完全参与3D渲染管线
- 需手动设置RectTransform尺寸（Width/Height），不受屏幕分辨率自动控制
- 渲染顺序由3D空间中的位置决定，而非Sort Order
- 需要GraphicRaycaster配合才能接收输入事件
- 适合：3D场景中的血条、场景内交互提示、VR UI、游戏内电脑屏幕

### 渲染模式对比表

| 特性 | Overlay | Camera | World Space |
|------|---------|--------|-------------|
| 依赖Camera | 否 | 是 | 是（可选） |
| 坐标系 | 屏幕像素 | Camera视平面 | 世界坐标 |
| 分辨率自适应 | 自动 | 半自动 | 手动 |
| 后处理影响 | 否 | 是 | 是 |
| 3D遮挡 | 否 | 是 | 是 |
| 性能开销 | 最低 | 中等 | 最高 |
| 像素完美 | 天然支持 | 需配置 | 需配置 |

### 排序与层级控制

Canvas的`Sort Order`属性控制同一渲染模式下的显示优先级，值越大越靠前。不同Camera之间由Camera Depth决定渲染顺序（值小的先渲染）。

```csharp
// 运行时动态设置Canvas排序
Canvas canvas = GetComponent<Canvas>();
canvas.sortingOrder = 10;
canvas.overrideSorting = true; // 必须开启才能生效

// 注意：开启overrideSorting会导致该Canvas独立合批，
// 与父Canvas打断合批关系，增加Draw Call
```

## 具体实现方法

### 多Canvas分层策略（完整实现）

```csharp
// 推荐的Canvas分层方案
public enum UILayer
{
    Background = 0,    // 背景层（全屏背景图、渐变遮罩）
    Normal = 100,      // 普通界面（背包、角色面板、商城）
    Popup = 200,       // 弹窗层（确认对话框、物品详情）
    Guide = 300,       // 新手引导层（遮罩、高亮框、指引箭头）
    Top = 400,         // 顶部通知层（系统公告、成就弹出）
    System = 500       // 系统提示层（网络断开、加载中）
}

public class UILayerManager : MonoBehaviour
{
    private Dictionary<UILayer, Canvas> layerCanvases
        = new Dictionary<UILayer, Canvas>();

    public Canvas GetOrCreateLayer(UILayer layer)
    {
        if (layerCanvases.TryGetValue(layer, out Canvas canvas))
            return canvas;

        GameObject go = new GameObject($"Canvas_{layer}");
        go.transform.SetParent(transform);

        canvas = go.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = (int)layer;
        canvas.pixelPerfect = false; // 关闭像素完美以提升性能

        // CanvasScaler配置
        CanvasScaler scaler = go.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight = 0.5f;

        go.AddComponent<GraphicRaycaster>();

        layerCanvases[layer] = canvas;
        return canvas;
    }
}
```

每个层级使用独立Canvas，只有该层级内元素变化时才触发Rebuild。这是UGUI性能优化的核心策略之一。

### Canvas Group组件详解

CanvasGroup用于整组UI的透明度、交互状态控制，常用于弹窗淡入淡出和禁用整块UI的交互：

```csharp
public class PopupWindow : UIBase
{
    private CanvasGroup group;

    public override void OnInit()
    {
        group = GetComponent<CanvasGroup>();
    }

    public override void OnShow(params object[] args)
    {
        gameObject.SetActive(true);
        group.alpha = 0f;
        group.interactable = false;
        group.blocksRaycasts = false;

        // 淡入动画
        DOTween.To(() => group.alpha, x => group.alpha = x, 1f, 0.3f)
            .OnComplete(() =>
            {
                group.interactable = true;
                group.blocksRaycasts = true;
            });
    }

    public override void OnHide()
    {
        group.interactable = false;
        group.blocksRaycasts = false;

        DOTween.To(() => group.alpha, x => group.alpha = x, 0f, 0.2f)
            .OnComplete(() => gameObject.SetActive(false));
    }
}
```

**CanvasGroup的关键属性**：
- `alpha`：控制自身及所有子元素的透明度（乘法叠加），比逐个设置Image.alpha更高效
- `interactable`：控制按钮、输入框等交互组件是否可交互
- `blocksRaycasts`：控制是否阻挡射线检测（点击穿透）。设为false时，点击会穿透到下方UI
- `ignoreParentGroups`：是否忽略父级CanvasGroup的影响

### 动态创建Canvas（完整版）

```csharp
public class CanvasFactory
{
    /// <summary>
    /// 创建一个独立层级的Canvas，用于隔离频繁更新的UI元素
    /// </summary>
    public static Canvas CreateLayerCanvas(string name, int sortOrder,
        RenderMode mode = RenderMode.ScreenSpaceOverlay)
    {
        GameObject go = new GameObject(name);

        Canvas canvas = go.AddComponent<Canvas>();
        canvas.renderMode = mode;
        canvas.sortingOrder = sortOrder;
        canvas.overrideSorting = true;

        // 添加CanvasScaler保持与其他UI一致的缩放
        CanvasScaler scaler = go.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight = 0.5f;

        // 必须添加GraphicRaycaster才能接收输入
        go.AddComponent<GraphicRaycaster>();

        return canvas;
    }

    /// <summary>
    /// 创建World Space Canvas（用于3D场景中的UI）
    /// </summary>
    public static Canvas CreateWorldCanvas(string name, Vector2 size,
        Transform parent = null)
    {
        GameObject go = new GameObject(name);
        if (parent != null) go.transform.SetParent(parent);

        Canvas canvas = go.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;

        RectTransform rt = go.GetComponent<RectTransform>();
        rt.sizeDelta = size;
        rt.localScale = Vector3.one * 0.01f; // World Space UI通常需要缩小

        go.AddComponent<CanvasScaler>();
        go.AddComponent<GraphicRaycaster>();

        return canvas;
    }
}
```

### Canvas性能分析工具

```csharp
/// <summary>
/// 运行时Canvas分析器，帮助识别合批打断和性能问题
/// </summary>
public class CanvasAnalyzer : MonoBehaviour
{
    void Update()
    {
        if (!Input.GetKeyDown(KeyCode.F3)) return;

        Canvas[] canvases = FindObjectsOfType<Canvas>();
        Debug.Log("=== Canvas Analysis ===");
        foreach (var c in canvases)
        {
            int graphicCount = c.GetComponentsInChildren<Graphic>().Length;
            int canvasCount = c.GetComponentsInChildren<Canvas>().Length;
            bool active = c.gameObject.activeInHierarchy;

            Debug.Log($"[{c.sortingOrder}] {c.name}: " +
                $"{graphicCount} graphics, {canvasCount} sub-canvases, " +
                $"active={active}, overrideSorting={c.overrideSorting}");
        }

#if UNITY_EDITOR
        Debug.Log($"Total Batches: {UnityEditor.UnityStats.batches}");
        Debug.Log($"SetPass Calls: {UnityEditor.UnityStats.setPassCalls}");
#endif
    }
}
```

## Canvas性能基准数据

| 场景 | 元素数量 | BuildBatch耗时 | Draw Calls | 说明 |
|------|---------|---------------|------------|------|
| 单Canvas 100个Image | 100 | 0.5ms | 1-3 | 同图集合批好 |
| 单Canvas 1000个Image | 1000 | 5-8ms | 5-15 | 开始出现卡顿 |
| 5个Canvas各100个Image | 500 | 0.8ms | 5-10 | 分Canvas更优 |
| 单Canvas含100个Text | 100 | 3-5ms | 50+ | 文字打断合批严重 |
| World Space 100个血条 | 100 | 2-4ms | 10-20 | 需要合批优化 |

## 最佳实践

- **按功能模块拆分Canvas**：HUD一个Canvas、聊天一个Canvas、背包一个Canvas，更新频率差异大的元素务必分开
- **静态UI和动态UI分开放在不同Canvas中**：背景图、装饰纹理由于从不变化，放独立Canvas不会触发Rebuild
- **Screen Space Camera模式下保持Plane Distance较小**：推荐0.1-1之间，减少精度问题和3D穿插
- **World Space UI务必添加GraphicRaycaster**：否则无法响应任何输入事件
- **关闭不需要的pixelPerfect**：pixelPerfect会增加额外的计算开销
- **尽量减少Canvas数量**：每个独立Canvas增加一个Draw Call，不要过度拆分

## 常见陷阱与修复

**陷阱1：在一个Canvas下放置所有UI**
- 症状：单个文本变化触发全UI的Rebuild，Profiler中Canvas.BuildBatch耗时>10ms
- 修复：按更新频率拆分Canvas，高频更新元素（倒计时、伤害数字）独立Canvas

**陷阱2：Overlay模式下误以为UI能被3D物体遮挡**
- 症状：3D角色模型无法遮挡UI，UI始终在最上层
- 修复：改用Screen Space - Camera模式，利用深度测试实现遮挡

**陷阱3：Camera模式忘记设置EventCamera**
- 症状：UI按钮点击无响应，EventSystem报错
- 修复：在GraphicRaycaster的Event Camera字段中设置对应的Camera

**陷阱4：过度使用overrideSorting**
- 症状：Draw Call数量暴增
- 修复：只在确实需要独立排序层级时才开启overrideSorting，子对象继承父Canvas排序即可

**陷阱5：未理解CanvasGroup.alpha与Image.alpha的区别**
- CanvasGroup.alpha对所有子元素生效，且只在GPU端做一次颜色乘法
- Image.alpha需要每个元素独立计算，性能稍差但可分别控制
