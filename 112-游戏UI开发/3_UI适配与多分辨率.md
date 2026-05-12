# UI适配与多分辨率

## 核心概念

移动设备分辨率碎片化是UI适配的最大挑战。当前市场上存在数百种分辨率组合，从iPhone SE的667p到iPad Pro的2732p，从16:9到21:9甚至折叠屏的多变比例。UI系统必须优雅地处理所有这些情况。

### 锚点系统（Anchors）数学原理

RectTransform的锚点系统是Unity UI适配的核心机制。锚点定义了UI元素相对于父容器的定位方式：

**锚点计算公式**：
```
最终位置.x = 父容器.width * anchorMin.x + anchoredPosition.x
最终位置.x（右） = 父容器.width * anchorMax.x + anchoredPosition.x + sizeDelta.x
```

理解这个公式是正确使用锚点的关键：
- `anchorMin/anchorMax`：归一化坐标(0-1)，定义锚点区域
- `anchoredPosition`：相对于锚点的偏移
- `sizeDelta`：在锚点拉伸模式下的额外尺寸

**四种常见锚点配置**：

1. **固定位置（四角锚点收缩到一点）**：UI大小不变，位置固定相对于锚点偏移
   - 适合：固定位置的按钮、标题
   - 不同分辨率下位置不变，但可能超出屏幕

2. **拉伸填充（四角锚点铺满）**：UI完全跟随父容器尺寸
   - 适合：全屏背景图、遮罩层
   - `anchorMin=(0,0), anchorMax=(1,1), offsetMin=(0,0), offsetMax=(0,0)`

3. **水平/垂直拉伸**：一个方向固定，另一个方向拉伸
   - 适合：顶部工具栏（水平拉伸、高度固定）、侧边栏

4. **中心锚点**：锚点在父容器中心
   - 适合：居中显示的标题、弹窗
   - 不同分辨率下保持居中

### Canvas Scaler组件深入

CanvasScaler是控制UI整体缩放策略的关键组件：

**Scale With Screen Size模式详解**：

```csharp
CanvasScaler scaler = canvas.GetComponent<CanvasScaler>();
scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
sccaler.referenceResolution = new Vector2(1920, 1080);
sccaler.matchWidthOrHeight = 0.5f; // 0=适配宽度, 1=适配高度, 0.5=折中
```

`matchWidthOrHeight`的计算逻辑：
```
实际缩放 = Lerp(
    屏幕宽度 / 参考宽度,
    屏幕高度 / 参考高度,
    matchWidthOrHeight
)
```

- 当 `match = 0`：缩放比 = 屏幕宽/参考宽。16:9屏幕在1920x1080参考下缩放=1.0，但在21:9屏幕上缩放可能=1.2，导致纵向内容被裁切
- 当 `match = 1`：缩放比 = 屏幕高/参考高。保证纵向内容完整，但21:9屏幕两侧可能有空白
- 当 `match = 0.5`：取两者几何平均值

**Constant Physical Size模式**：
- 按物理尺寸（英寸/厘米）缩放，确保不同DPI设备上UI物理大小一致
- 需要设备正确报告DPI（`Screen.dpi`），部分设备报告不准确

### 安全区域（Safe Area）完整处理

刘海屏、挖孔屏、圆角屏需要避开系统UI遮挡区域：

```csharp
using UnityEngine;

/// <summary>
/// 安全区域适配器，处理刘海屏/挖孔屏/圆角屏
/// 在每帧检测安全区域变化（横竖屏切换、系统UI变化时可能改变）
/// </summary>
[RequireComponent(typeof(RectTransform))]
public class SafeAreaAdapter : MonoBehaviour
{
    private RectTransform panel;
    private Rect lastSafeArea = Rect.zero;
    private Vector2Int lastScreenSize = Vector2Int.zero;
    private ScreenOrientation lastOrientation = ScreenOrientation.AutoRotation;

    void Start()
    {
        panel = GetComponent<RectTransform>();
        ApplySafeArea();
    }

    void Update()
    {
        if (lastSafeArea != Screen.safeArea
            || lastScreenSize.x != Screen.width
            || lastScreenSize.y != Screen.height
            || lastOrientation != Screen.orientation)
        {
            ApplySafeArea();
        }
    }

    void ApplySafeArea()
    {
        Rect safeArea = Screen.safeArea;

        if (safeArea == Rect.zero) return; // 无效数据

        // 归一化到0-1范围
        Vector2 anchorMin = safeArea.position;
        Vector2 anchorMax = safeArea.position + safeArea.size;
        anchorMin.x /= Screen.width;
        anchorMin.y /= Screen.height;
        anchorMax.x /= Screen.width;
        anchorMax.y /= Screen.height;

        // 防止浮点精度问题导致的微小抖动
        anchorMin.x = Mathf.Round(anchorMin.x * 1000f) / 1000f;
        anchorMin.y = Mathf.Round(anchorMin.y * 1000f) / 1000f;
        anchorMax.x = Mathf.Round(anchorMax.x * 1000f) / 1000f;
        anchorMax.y = Mathf.Round(anchorMax.y * 1000f) / 1000f;

        panel.anchorMin = anchorMin;
        panel.anchorMax = anchorMax;

        lastSafeArea = safeArea;
        lastScreenSize = new Vector2Int(Screen.width, Screen.height);
        lastOrientation = Screen.orientation;

        Debug.Log($"Safe Area applied: {safeArea} on {Screen.width}x{Screen.height}");
    }
}
```

## 具体实现方法

### 横竖屏适配策略（完整版）

```csharp
public class OrientationAdapter : MonoBehaviour
{
    [System.Serializable]
    public class LayoutConfig
    {
        public RectTransform panel;
        public Vector2 landscapePosition;
        public Vector2 portraitPosition;
        public Vector2 landscapeSize;
        public Vector2 portraitSize;
    }

    [SerializeField] private LayoutConfig[] configs;
    [SerializeField] private HorizontalLayoutGroup horizontalGroup;
    [SerializeField] private VerticalLayoutGroup verticalGroup;

    private ScreenOrientation currentOrientation;

    void Start()
    {
        currentOrientation = Screen.orientation;
        ApplyLayout();
    }

    void Update()
    {
        if (Screen.orientation != currentOrientation)
        {
            currentOrientation = Screen.orientation;
            ApplyLayout();
        }
    }

    void ApplyLayout()
    {
        bool isLandscape = Screen.width > Screen.height;

        foreach (var config in configs)
        {
            if (config.panel == null) continue;

            config.panel.anchoredPosition = isLandscape
                ? config.landscapePosition
                : config.portraitPosition;
            config.panel.sizeDelta = isLandscape
                ? config.landscapeSize
                : config.portraitSize;
        }

        // 切换布局组方向
        if (horizontalGroup != null)
            horizontalGroup.childAlignment = isLandscape
                ? TextAnchor.MiddleCenter
                : TextAnchor.UpperCenter;

        if (verticalGroup != null)
            verticalGroup.childAlignment = isLandscape
                ? TextAnchor.MiddleLeft
                : TextAnchor.UpperCenter;
    }
}
```

### matchWidthOrHeight选择指南

| 游戏类型 | 推荐match值 | 理由 |
|---------|------------|------|
| 横屏动作/RPG | 1.0 | 保证纵向内容完整，横向可能出现额外视野但不丢失信息 |
| 竖屏卡牌/社交 | 0.0 | 保证横向内容完整，纵向可以滚动 |
| 横竖屏切换 | 0.5 | 两种比例都有妥协但都能接受 |
| 棋牌/桌面 | 0.5 | 四方布局需要等比缩放 |
| 超宽屏优化 | 1.0 + 代码扩展 | 额外视野区域显示更多信息 |

### 响应式布局系统

```csharp
/// <summary>
/// 响应式布局管理器：根据屏幕宽高比自动切换布局方案
/// </summary>
public class ResponsiveLayoutManager : MonoBehaviour
{
    public enum AspectCategory { Narrow, Standard, Wide, UltraWide }

    [System.Serializable]
    public class LayoutPreset
    {
        public AspectCategory category;
        public GameObject layoutRoot;
    }

    [SerializeField] private LayoutPreset[] presets;

    private AspectCategory currentCategory;

    void Start()
    {
        UpdateLayout();
    }

    void Update()
    {
        AspectCategory newCategory = GetAspectCategory();
        if (newCategory != currentCategory)
        {
            currentCategory = newCategory;
            UpdateLayout();
        }
    }

    AspectCategory GetAspectCategory()
    {
        float aspect = (float)Screen.width / Screen.height;
        if (aspect < 1.5f) return AspectCategory.Narrow;    // 4:3, 竖屏
        if (aspect < 1.9f) return AspectCategory.Standard;  // 16:9
        if (aspect < 2.2f) return AspectCategory.Wide;      // 18:9, 19.5:9
        return AspectCategory.UltraWide;                     // 21:9+
    }

    void UpdateLayout()
    {
        foreach (var preset in presets)
        {
            if (preset.layoutRoot != null)
                preset.layoutRoot.SetActive(preset.category == currentCategory);
        }
    }
}
```

### DIP（设备无关像素）处理

```csharp
/// <summary>
/// 处理不同DPI设备的UI尺寸一致性
/// </summary>
public class DPIAwareScaler : MonoBehaviour
{
    [SerializeField] private float targetDPI = 264f; // iPad标准DPI
    [SerializeField] private RectTransform[] scaleTargets;

    void Start()
    {
        float currentDPI = Screen.dpi;
        if (currentDPI <= 0) currentDPI = 150f; // 回退值

        float scaleFactor = targetDPI / currentDPI;
        scaleFactor = Mathf.Clamp(scaleFactor, 0.7f, 1.5f);

        foreach (var target in scaleTargets)
        {
            target.localScale = Vector3.one * scaleFactor;
        }
    }
}
```

## 最佳实践

- UI设计稿使用1920x1080或项目主目标分辨率作为基准
- 核心交互区域（按钮、输入框）放在屏幕中央80%范围内，确保安全区域覆盖
- 使用锚点拉伸代替代码中手动计算位置，让Unity自动处理适配
- 利用Layout Group自动处理子元素排列，减少硬编码坐标
- 针对Pad和Phone分别提供适配方案，关键UI元素两套布局
- 安全区域适配器挂载在UI根节点，所有需要适配的UI作为其子节点
- 测试时覆盖极端比例：4:3、16:9、19.5:9、21:9、竖屏

## 常见陷阱与修复

**陷阱1：忽略安全区域导致刘海遮挡关键按钮**
- 症状：iPhone 14 Pro上返回按钮被挖孔屏遮挡
- 修复：使用SafeAreaAdapter包裹所有UI内容

**陷阱2：使用代码`transform.position`硬编码UI位置**
- 症状：不同分辨率下UI位置偏移、按钮超出屏幕
- 修复：全部使用RectTransform的anchoredPosition和锚点系统

**陷阱3：matchWidthOrHeight设置不当导致内容裁切或黑边**
- 症状：21:9手机上上下出现黑边，或4:3平板上左右内容被裁切
- 修复：根据游戏类型选择match值，极端比例用ResponsiveLayoutManager切换布局

**陷阱4：未考虑屏幕DPI差异，导致高分屏上UI元素过小**
- 症状：iPad Pro上按钮手指难以点击
- 修复：CanvasScaler使用Constant Physical Size，或用DPIAwareScaler辅助缩放

**陷阱5：安全区域每帧检测但未做变化判断**
- 症状：不必要的性能开销
- 修复：仅在Screen.safeArea或Screen.orientation变化时才重新计算
