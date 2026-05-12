# UI性能优化

## 核心概念

UI是游戏中性能开销的主要来源之一，尤其在移动平台上。一个中等复杂度的游戏UI可能包含数百个Graphic组件，如果管理不当，每帧的Rebuild和Draw Call开销可达10-20ms，直接导致帧率掉到30fps以下。理解UGUI的Rebuild机制和合批规则是优化的关键。

### UGUI Rebuild机制深度解析

当UI元素的视觉属性（位置、颜色、文字内容、图片sprite等）发生变化时，Unity会将该元素所在的Canvas标记为"Dirty"。在下一帧渲染前，Unity执行Canvas.BuildBatch进行网格重建：

```
属性变化 → Canvas标记Dirty → 下一帧Canvas.BuildBatch
    ↓
遍历Canvas下所有Graphic组件
    ↓
检查每个Graphic是否Dirty
    ↓
重新计算顶点、UV、颜色数据
    ↓
生成新的Mesh网格
    ↓
提交给GPU渲染
```

关键点：Rebuild范围是整个Canvas，而非单个元素。即使只有一个文本组件内容变化，整个Canvas下的所有Graphic都会被重新检查。这就是为什么需要将不同更新频率的UI元素放到不同Canvas中。

### 合批打断因素详解

UGUI的合批条件非常苛刻。合批要求连续渲染的UI元素满足以下所有条件：
1. **相同材质**：不同Shader或不同材质属性都会打断
2. **相同纹理/图集**：使用不同图集的Image无法合批
3. **相同渲染层级**：中间插入不同层级的Canvas会打断
4. **渲染顺序**：深度排序后，不同图集的元素交替出现会打断

最隐蔽的打断源：文字和图片交替排列。因为TMP文字使用SDF字体图集，而Image使用UI图集，两者交替出现会导致频繁的材质切换。

### Draw Call vs Batches vs SetPass Calls

| 指标 | 含义 | 优化目标 |
|------|------|---------|
| Draw Call | CPU向GPU发送的渲染指令数 | <50（UI部分） |
| Batches | Unity实际合批后的批次数 | 接近Draw Call |
| SetPass Calls | 材质切换次数（最影响性能） | <20（UI部分） |
| Triangles | 总三角形数 | <50000（UI部分） |

## 具体实现方法

### Draw Call优化策略（完整方案）

**策略1：按图集组织UI元素**

```csharp
/// <summary>
/// UI图集管理器
/// 确保同一界面的元素使用相同图集，最大化合批
/// </summary>
public class UIAtlasManager
{
    // 图集规划建议：
    // Atlas_Common:  公共图标（关闭按钮、金币、钻石等）
    // Atlas_Bag:     背包界面专用图标
    // Atlas_Shop:    商城界面专用图标
    // Atlas_Skill:   技能图标
    // Atlas_Status:  状态图标（Buff/Debuff）

    // 关键原则：同一Canvas下尽量只使用1-2个图集
    // 不同界面使用不同Canvas → 不同图集不会互相打断
}
```

**策略2：文字和图片分离Canvas**

```csharp
/// <summary>
/// 将文字和图片分离到不同Canvas
/// 避免文字图集和图片图集交替导致合批打断
/// </summary>
public class TextCanvasIsolator : MonoBehaviour
{
    void Start()
    {
        // 为文字创建独立Canvas
        TextMeshProUGUI[] texts = GetComponentsInChildren<TextMeshProUGUI>(true);
        foreach (var text in texts)
        {
            // 如果文字在频繁更新的元素下，为其创建独立Canvas
            if (text.transform.GetComponentInParent<DynamicElement>() != null)
            {
                Canvas subCanvas = text.gameObject.AddComponent<Canvas>();
                subCanvas.overrideSorting = false; // 继承父Canvas排序
                text.gameObject.AddComponent<GraphicRaycaster>();
            }
        }
    }
}
```

### Raycast Target精简（完整版）

```csharp
/// <summary>
/// 批量关闭不需要射线检测的UI元素
/// 可以减少50-80%的射线检测开销
/// </summary>
public static class RaycastOptimizer
{
    /// <summary>
    /// 在UI面板初始化时调用
    /// </summary>
    public static int OptimizeRaycastTargets(Transform root)
    {
        var graphics = root.GetComponentsInChildren<Graphic>(true);
        int disabledCount = 0;

        foreach (var graphic in graphics)
        {
            if (!ShouldReceiveRaycast(graphic))
            {
                graphic.raycastTarget = false;
                disabledCount++;
            }
        }

        return disabledCount;
    }

    static bool ShouldReceiveRaycast(Graphic graphic)
    {
        // Button组件 → 必须保留
        if (graphic.GetComponent<Button>() != null) return true;
        // Toggle组件 → 必须保留
        if (graphic.GetComponent<Toggle>() != null) return true;
        // InputField组件 → 必须保留
        if (graphic.GetComponent<TMP_InputField>() != null) return true;
        // Slider组件 → 必须保留
        if (graphic.GetComponent<Slider>() != null) return true;
        // Scrollbar组件 → 必须保留
        if (graphic.GetComponent<Scrollbar>() != null) return true;
        // Dropdown组件 → 必须保留
        if (graphic.GetComponent<TMP_Dropdown>() != null) return true;
        // 实现了事件接口 → 保留
        if (graphic is IPointerClickHandler) return true;
        if (graphic is IDropHandler) return true;
        if (graphic is IDragHandler) return true;

        // 纯装饰性元素 → 关闭
        return false;
    }
}
```

### Canvas分离策略（完整版）

```csharp
/// <summary>
/// 按更新频率智能分离Canvas
/// 核心原则：相同更新频率的元素放同一Canvas
/// </summary>
public class CanvasSeparationStrategy : MonoBehaviour
{
    // 更新频率分类：
    // 0Hz (静态):   背景图、装饰纹理、固定文字 → Canvas_Static
    // 1-5Hz (低频): 等级显示、金币数量、任务进度 → Canvas_LowFreq
    // 10-30Hz (中频): 血条、技能冷却、小地图 → Canvas_MidFreq
    // 60Hz (高频):  伤害数字、实时倒计时、帧数显示 → Canvas_HighFreq

    /// <summary>
    /// 为频繁更新的子元素自动创建独立Canvas
    /// </summary>
    public static Canvas IsolateDynamicElement(GameObject go, int sortOrder = 0)
    {
        Canvas existingCanvas = go.GetComponent<Canvas>();
        if (existingCanvas != null) return existingCanvas;

        Canvas subCanvas = go.AddComponent<Canvas>();
        subCanvas.overrideSorting = false; // 继承父层级排序
        go.AddComponent<GraphicRaycaster>();
        return subCanvas;
    }
}
```

### 对象池化UI（完整泛型实现）

```csharp
/// <summary>
/// 泛型UI对象池
/// 避免频繁Instantiate/Destroy UI元素导致的GC和卡顿
/// </summary>
public class UIObjectPool<T> where T : Component
{
    private Queue<T> inactivePool = new Queue<T>();
    private HashSet<T> activeObjects = new HashSet<T>();
    private T prefab;
    private Transform parent;
    private int maxSize;

    public int ActiveCount => activeObjects.Count;
    public int InactiveCount => inactivePool.Count;

    public UIObjectPool(T prefab, Transform parent, int prewarm = 10, int maxSize = 100)
    {
        this.prefab = prefab;
        this.parent = parent;
        this.maxSize = maxSize;

        // 预热：提前创建对象
        for (int i = 0; i < prewarm; i++)
        {
            T obj = CreateNew();
            obj.gameObject.SetActive(false);
            inactivePool.Enqueue(obj);
        }
    }

    public T Get()
    {
        T obj;
        if (inactivePool.Count > 0)
        {
            obj = inactivePool.Dequeue();
        }
        else
        {
            obj = CreateNew();
        }

        obj.gameObject.SetActive(true);
        activeObjects.Add(obj);
        return obj;
    }

    public void Return(T obj)
    {
        if (obj == null) return;

        obj.gameObject.SetActive(false);
        activeObjects.Remove(obj);

        if (inactivePool.Count < maxSize)
        {
            inactivePool.Enqueue(obj);
        }
        else
        {
            // 超出容量上限，直接销毁
            GameObject.Destroy(obj.gameObject);
        }
    }

    public void ReturnAll()
    {
        var toReturn = activeObjects.ToArray();
        foreach (var obj in toReturn)
            Return(obj);
    }

    private T CreateNew()
    {
        T obj = GameObject.Instantiate(prefab, parent);
        return obj;
    }

    public void Dispose()
    {
        foreach (var obj in inactivePool)
            if (obj != null) GameObject.Destroy(obj.gameObject);
        foreach (var obj in activeObjects)
            if (obj != null) GameObject.Destroy(obj.gameObject);
        inactivePool.Clear();
        activeObjects.Clear();
    }
}
```

### 文本更新优化

```csharp
/// <summary>
/// 高性能文本更新器
/// 仅在值变化时更新文字，避免每帧触发Canvas Rebuild
/// </summary>
public class OptimizedTextDisplay : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI targetText;

    private int lastIntValue = int.MinValue;
    private float lastFloatValue = float.MinValue;
    private string lastStringValue = null;

    /// <summary>
    /// 更新整数值（仅在变化时）
    /// </summary>
    public void SetValue(int value, string format = "{0}")
    {
        if (value == lastIntValue) return;
        lastIntValue = value;
        targetText.text = string.Format(format, value);
    }

    /// <summary>
    /// 更新浮点值（带精度判断）
    /// </summary>
    public void SetValue(float value, float precision = 0.01f, string format = "{0:F1}")
    {
        if (Mathf.Abs(value - lastFloatValue) < precision) return;
        lastFloatValue = value;
        targetText.text = string.Format(format, value);
    }

    /// <summary>
    /// 更新字符串值
    /// </summary>
    public void SetValue(string value)
    {
        if (value == lastStringValue) return;
        lastStringValue = value;
        targetText.text = value;
    }

    /// <summary>
    /// 更新血条等高频变化UI
    /// </summary>
    public void SetHPBar(Slider slider, int currentHP, int maxHP)
    {
        float ratio = (float)currentHP / maxHP;
        if (Mathf.Abs(slider.value - ratio) < 0.001f) return;
        slider.value = ratio;
    }
}
```

### Mask组件替代方案

```csharp
/// <summary>
/// RectMask2D vs Mask 对比
///
/// Mask组件：
/// - 通过模板缓冲实现，每个Mask增加2个Draw Call（开启+关闭模板测试）
/// - 支持任意形状遮罩
/// - 性能开销较大
///
/// RectMask2D：
/// - 通过裁剪矩形实现，不增加Draw Call
/// - 仅支持矩形裁剪
/// - 性能开销极低（推荐使用）
///
/// 结论：只要矩形裁剪能满足需求，就用RectMask2D代替Mask
/// </summary>
public class MaskComparison
{
    // 滚动列表中的Mask → 使用RectMask2D
    // 圆形头像遮罩 → 必须使用Mask（RectMask2D无法实现圆形）
    // 复杂形状裁剪 → 必须使用Mask
}
```

## 性能基准数据

| 优化场景 | 优化前 | 优化后 | 提升幅度 |
|---------|--------|--------|---------|
| 100个Image合批 | 100 Draw Calls | 3 Draw Calls | 97% |
| 关闭100个raycastTarget | 0.5ms/帧 | 0.1ms/帧 | 80% |
| 拆分Canvas(1→5) | Rebuild 8ms | Rebuild 1.5ms | 81% |
| 对象池vs Instantiate | 2ms/次创建 | 0.01ms/次获取 | 99.5% |
| 文本变化检测 | 0.3ms/帧 | 0.02ms/帧 | 93% |
| RectMask2D vs Mask | +2 DrawCalls/Mask | +0 DrawCalls | 100% |

## 最佳实践

- 使用Frame Debugger定期检查UI Draw Call数量和合批情况
- 将UI按更新频率拆分到不同Canvas中（静态/低频/中频/高频）
- 背景图、装饰性元素全部关闭Raycast Target
- 频繁创建销毁的UI元素使用对象池（伤害数字、聊天消息、邮件列表项）
- 使用图集管理小图标，减少纹理切换，同一Canvas尽量只用1-2个图集
- 低端设备上减少UI层级深度和Mask使用
- RectMask2D替代Mask用于滚动列表
- 文本更新仅在值变化时进行
- 使用Profiler的UI模块专门分析Canvas.BuildBatch耗时

## 常见陷阱与修复

**陷阱1：在Update中每帧设置Text.text**
- 症状：每帧触发Canvas Rebuild，即使文字内容没变
- 修复：先判断值是否变化，只在变化时更新。或使用OptimizedTextDisplay组件

**陷阱2：使用多个Canvas但未正确管理Sort Order**
- 症状：UI渲染顺序错乱，弹窗被遮挡
- 修复：定义UILayer枚举统一管理Sort Order

**陷阱3：Mask组件滥用**
- 症状：每个Mask至少增加2个Draw Call，10个Mask就是20个额外Draw Call
- 修复：能用RectMask2D的地方绝不使用Mask

**陷阱4：未使用Profiler的UI模块专门分析UI开销**
- 症状：总感觉UI卡但不知道卡在哪
- 修复：打开Profiler → UI模块，查看Canvas.BuildBatch和Layout重建耗时

**陷阱5：对静态UI也添加了不必要的Canvas组件**
- 症状：Canvas数量过多，每个Canvas增加Draw Call和管理开销
- 修复：只有需要独立排序或独立Rebuild的元素才添加Canvas
