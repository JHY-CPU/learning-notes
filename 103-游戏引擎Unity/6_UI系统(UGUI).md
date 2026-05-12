# UI系统(UGUI)

## 核心概念

UGUI是Unity的内置UI系统，基于Canvas渲染。所有UI元素都是Canvas的子对象，通过RectTransform进行布局。UGUI使用事件系统处理用户交互。

## Canvas渲染模式

Canvas有三种渲染模式，决定了UI与世界空间的关系：

| 渲染模式 | 特点 | 适用场景 |
|----------|------|----------|
| Screen Space - Overlay | 直接覆盖在屏幕上，不依赖Camera | 主菜单、HUD |
| Screen Space - Camera | 绑定Camera，UI在相机前方 | 大多数UI场景 |
| World Space | UI作为3D物体存在于世界中 | 血条、交互面板、VR UI |

```csharp
// 动态切换Canvas模式
Canvas canvas = GetComponent<Canvas>();
canvas.renderMode = RenderMode.ScreenSpaceCamera;
canvas.worldCamera = Camera.main;
canvas.planeDistance = 10f;
```

## RectTransform

RectTransform是Transform的子类，专用于UI元素的定位和布局：

```csharp
public class RectTransformExample : MonoBehaviour
{
    private RectTransform rt;

    void Start()
    {
        rt = GetComponent<RectTransform>();

        // 锚点和轴心
        rt.anchorMin = new Vector2(0, 0);    // 锚点最小值（左下角）
        rt.anchorMax = new Vector2(1, 1);    // 锚点最大值（右上角）- 拉伸填充
        rt.pivot = new Vector2(0.5f, 0.5f);  // 轴心点（旋转/缩放的中心）

        // 位置和尺寸
        rt.anchoredPosition = new Vector2(100, -50); // 相对锚点的偏移
        rt.sizeDelta = new Vector2(200, 50);          // 相对锚点间距的尺寸差

        // 世界坐标
        rt.position = new Vector3(Screen.width / 2, Screen.height / 2, 0);
    }
}
```

### 常用锚点预设
- **Stretch**: 锚点四角拉伸，UI自适应父容器大小
- **Top/Bottom/Left/Right**: 固定在某一边
- **Center**: 居中（默认）

## Layout Group

自动排列子UI元素的布局组件：

```csharp
// VerticalLayoutGroup - 垂直排列
VerticalLayoutGroup vlg = container.AddComponent<VerticalLayoutGroup>();
vlg.spacing = 10f;              // 元素间距
vlg.childAlignment = TextAnchor.UpperCenter;
vlg.childControlWidth = true;   // 控制子元素宽度
vlg.childControlHeight = false;
vlg.padding = new RectOffset(10, 10, 10, 10); // 内边距

// GridLayoutGroup - 网格排列（背包、技能栏）
GridLayoutGroup glg = container.AddComponent<GridLayoutGroup>();
glg.cellSize = new Vector2(80, 80);
glg.spacing = new Vector2(5, 5);
glg.constraint = GridLayoutGroup.Constraint.FixedColumnCount;
glg.constraintCount = 5;  // 每行5个

// ContentSizeFitter - 根据内容自动调整大小
ContentSizeFitter fitter = textObj.AddComponent<ContentSizeFitter>();
fitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
```

## EventSystem

EventSystem是UGUI的事件处理核心，负责将输入事件分发到UI组件：

```csharp
public class UIEventExample : MonoBehaviour, IPointerClickHandler, IPointerEnterHandler,
                              IDragHandler, IDropHandler, ISelectHandler
{
    // 点击事件
    public void OnPointerClick(PointerEventData eventData)
    {
        if (eventData.button == PointerEventData.InputButton.Left)
            Debug.Log("左键点击");
    }

    // 鼠标悬停
    public void OnPointerEnter(PointerEventData eventData)
    {
        GetComponent<Image>().color = Color.yellow;
    }

    // 拖拽
    public void OnDrag(PointerEventData eventData)
    {
        transform.position = eventData.position;
    }

    // 放置
    public void OnDrop(PointerEventData eventData)
    {
        Debug.Log("放置了物品");
    }

    // UI被选中（导航系统）
    public void OnSelect(BaseEventData eventData)
    {
        Debug.Log("UI被选中");
    }
}
```

### 常用UI组件事件绑定

```csharp
// Button点击
Button btn = GetComponent<Button>();
btn.onClick.AddListener(() => Debug.Log("按钮被点击"));

// Slider值变化
Slider slider = GetComponent<Slider>();
slider.onValueChanged.AddListener(value => Debug.Log($"音量: {value}"));

// InputField输入
InputField input = GetComponent<InputField>();
input.onValueChanged.AddListener(text => Debug.Log($"输入: {text}"));
input.onEndEdit.AddListener(text => Debug.Log($"输入结束: {text}"));

// Toggle开关
Toggle toggle = GetComponent<Toggle>();
toggle.onValueChanged.AddListener(isOn => Debug.Log($"开关: {isOn}"));

// Dropdown下拉选择
Dropdown dropdown = GetComponent<Dropdown>();
dropdown.onValueChanged.AddListener(index => Debug.Log($"选中: {index}"));
```

## 常见陷阱与最佳实践

1. **Canvas Rebuild开销大**: 动态UI变化会触发Canvas重建，尽量减少频繁更新
2. **Raycast Target**: 不需要交互的Image应关闭Raycast Target，减少射线检测开销
3. **UI排序按Hierarchy顺序**: 下方对象覆盖上方对象，与Z位置无关
4. **锚点适配**: 不同屏幕比例需合理设置锚点，不要硬编码像素位置
5. **EventSystem冲突**: 场景中只能有一个EventSystem，多个Canvas共享

## 与其他系统的关联

- **输入系统**: EventSystem依赖Input Manager或New Input System
- **渲染管线**: URP/HDRP下UI渲染路径可能不同
- **动画系统**: UI可使用Animator驱动UI动画（缩放、淡入淡出）
