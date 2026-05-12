# 游戏UI架构概览

## 核心概念

游戏UI系统是玩家与游戏世界交互的桥梁。一个设计良好的UI架构需要在功能复杂度、渲染性能和开发效率之间取得平衡。UI代码往往占据游戏总代码量的30%-50%，因此合理的架构设计是可维护UI系统的基石。

### UI分层模型

游戏UI通常按照功能和更新频率分为多个层次：

- **HUD层（Heads-Up Display）**：始终可见的游戏信息，如血条、小地图、技能冷却、弹药数量。特点是高频更新（每帧或每秒多次），需独立Canvas以避免触发其他UI的Rebuild。典型更新频率：血条每秒1-5次，小地图每秒10-30次。
- **Menu层**：暂停菜单、设置界面、角色属性面板等全屏覆盖型界面。通常独占输入焦点，打开时暂停游戏。更新频率低，内容变化少。
- **Popup层**：弹窗提示、确认对话框、物品详情Tooltip。需要栈式管理以支持多层弹窗叠加，后弹出的覆盖前面的。典型场景：获得物品时弹出提示，同时可能有系统公告弹窗。
- **Overlay层**：新手引导遮罩、截图遮罩、加载界面。通常需要在最上层渲染，覆盖一切UI。
- **Screen Space 3D层**：场景中的血条、名字标签、伤害数字。跟随3D物体位置但面向相机。

### UI管理器设计（单例模式详解）

UIManager是整个UI系统的核心调度器，负责界面的生命周期管理和层级控制。以下是完整的生产级实现：

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.ResourceManagement.AsyncOperations;

public enum UILayer
{
    Background = 0,
    Normal = 100,
    Popup = 200,
    Guide = 300,
    Top = 400,
    System = 500
}

public class UIManager : MonoBehaviour
{
    public static UIManager Instance { get; private set; }

    // 各层级的Canvas根节点
    private Dictionary<UILayer, Transform> layerRoots = new Dictionary<UILayer, Transform>();

    // 已打开的界面缓存（界面名 -> 界面实例）
    private Dictionary<string, UIBase> openedUIs = new Dictionary<string, UIBase>();

    // 已加载的界面资源句柄（用于释放）
    private Dictionary<string, AsyncOperationHandle<GameObject>> loadedHandles
        = new Dictionary<string, AsyncOperationHandle<GameObject>>();

    // 弹窗栈（用于回退关闭）
    private Stack<UIBase> popupStack = new Stack<UIBase>();

    // 界面注册表（界面名 -> Addressable地址）
    private Dictionary<string, string> uiAddressMap = new Dictionary<string, string>();

    void Awake()
    {
        if (Instance != null) { Destroy(gameObject); return; }
        Instance = this;
        DontDestroyOnLoad(gameObject);
        InitLayers();
        RegisterAllUI();
    }

    void InitLayers()
    {
        foreach (UILayer layer in Enum.GetValues(typeof(UILayer)))
        {
            GameObject layerGo = new GameObject($"Layer_{layer}");
            layerGo.transform.SetParent(transform);

            Canvas canvas = layerGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = (int)layer;

            layerGo.AddComponent<CanvasScaler>().uiScaleMode
                = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            layerGo.AddComponent<GraphicRaycaster>();

            layerRoots[layer] = layerGo.transform;
        }
    }

    void RegisterAllUI()
    {
        // 在实际项目中通常从配置表或ScriptableObject加载
        uiAddressMap["MainMenu"] = "UI/Prefabs/MainMenu";
        uiAddressMap["Inventory"] = "UI/Prefabs/Inventory";
        uiAddressMap["Settings"] = "UI/Prefabs/Settings";
        uiAddressMap["ItemTooltip"] = "UI/Prefabs/ItemTooltip";
        uiAddressMap["ConfirmDialog"] = "UI/Prefabs/ConfirmDialog";
    }

    // 打开界面（异步）
    public async void OpenUI<T>(string uiName, params object[] args) where T : UIBase
    {
        // 已打开则直接显示
        if (openedUIs.TryGetValue(uiName, out UIBase existing))
        {
            existing.OnShow(args);
            return;
        }

        // 异步加载预制体
        if (!uiAddressMap.TryGetValue(uiName, out string address))
        {
            Debug.LogError($"UI not registered: {uiName}");
            return;
        }

        AsyncOperationHandle<GameObject> handle
            = Addressables.InstantiateAsync(address);
        await handle.Task;

        if (handle.Status != AsyncOperationStatus.Succeeded)
        {
            Debug.LogError($"Failed to load UI: {uiName}");
            return;
        }

        loadedHandles[uiName] = handle;
        T ui = handle.Result.GetComponent<T>();
        UILayer targetLayer = ui.Layer;

        // 放入对应层级Canvas下
        ui.transform.SetParent(layerRoots[targetLayer], false);
        ui.UIName = uiName;

        openedUIs[uiName] = ui;
        ui.OnInit();
        ui.OnShow(args);

        // 弹窗入栈
        if (targetLayer == UILayer.Popup)
            popupStack.Push(ui);
    }

    // 关闭界面
    public void CloseUI(string uiName)
    {
        if (!openedUIs.TryGetValue(uiName, out UIBase ui)) return;

        ui.OnHide();
        ui.OnClose();

        // 弹窗出栈
        if (ui.Layer == UILayer.Popup && popupStack.Count > 0
            && popupStack.Peek() == ui)
            popupStack.Pop();

        // 释放实例和资源
        Destroy(ui.gameObject);
        if (loadedHandles.TryGetValue(uiName, out var handle))
        {
            Addressables.Release(handle);
            loadedHandles.Remove(uiName);
        }
        openedUIs.Remove(uiName);
    }

    // 关闭顶层弹窗（返回键逻辑）
    public void CloseTopPopup()
    {
        if (popupStack.Count > 0)
            CloseUI(popupStack.Peek().UIName);
    }

    // 获取已打开的界面
    public T GetUI<T>(string uiName) where T : UIBase
    {
        if (openedUIs.TryGetValue(uiName, out UIBase ui))
            return ui as T;
        return null;
    }
}
```

### MVC / MVP / MVVM 在游戏UI中的应用

这三种架构模式在游戏开发中各有适用场景，理解它们的区别有助于选择最合适的方案。

**MVC模式（Model-View-Controller）**：
- **Model**：纯数据层，不依赖Unity。例如玩家属性、背包数据、任务状态。Model的变化通过事件通知Controller。
- **View**：UI表现层，MonoBehaviour挂载的UI脚本。只负责显示，不包含业务逻辑。监听用户输入并转发给Controller。
- **Controller**：业务逻辑层，接收View的用户操作事件，修改Model数据，再通知View更新显示。

MVC的问题在于Controller容易膨胀成"上帝类"，且View和Controller之间耦合较紧。

**MVP模式（Model-View-Presenter）—— 游戏UI最常用**：
- **Presenter**：替代Controller，持有View引用并直接操作UI组件。View不直接引用Model，所有数据操作通过Presenter中转。
- View通过接口暴露UI操作方法，Presenter面向接口编程。
- 更适合Unity的组件化架构，每个UI面板对应一个Presenter。

```csharp
// Model层
[System.Serializable]
public class PlayerStatsModel
{
    public int hp;
    public int maxHP;
    public int mp;
    public int maxMP;
    public int level;
    public float exp;

    public event Action OnDataChanged;

    public void TakeDamage(int damage)
    {
        hp = Mathf.Max(0, hp - damage);
        OnDataChanged?.Invoke();
    }

    public void Heal(int amount)
    {
        hp = Mathf.Min(maxHP, hp + amount);
        OnDataChanged?.Invoke();
    }
}

// View接口
public interface IPlayerHUDView
{
    void SetHP(int current, int max);
    void SetMP(int current, int max);
    void SetLevel(int level);
    void SetExp(float ratio);
    void PlayDamageEffect();
    void PlayHealEffect();
}

// View实现
public class PlayerHUDView : MonoBehaviour, IPlayerHUDView
{
    [SerializeField] private Slider hpBar;
    [SerializeField] private Slider mpBar;
    [SerializeField] private TextMeshProUGUI levelText;
    [SerializeField] private Image expFill;
    [SerializeField] private ParticleSystem damageEffect;
    [SerializeField] private ParticleSystem healEffect;

    // View不持有Model引用，通过Presenter驱动
    public void SetHP(int current, int max)
    {
        hpBar.value = (float)current / max;
    }

    public void SetMP(int current, int max)
    {
        mpBar.value = (float)current / max;
    }

    public void SetLevel(int level)
    {
        levelText.text = $"Lv.{level}";
    }

    public void SetExp(float ratio)
    {
        expFill.fillAmount = ratio;
    }

    public void PlayDamageEffect()
    {
        damageEffect.Play();
    }

    public void PlayHealEffect()
    {
        healEffect.Play();
    }
}

// Presenter层
public class PlayerHUDPresenter
{
    private IPlayerHUDView view;
    private PlayerStatsModel model;

    public PlayerHUDPresenter(IPlayerHUDView view, PlayerStatsModel model)
    {
        this.view = view;
        this.model = model;
        model.OnDataChanged += OnModelChanged;
        RefreshAll();
    }

    void OnModelChanged()
    {
        RefreshAll();
    }

    void RefreshAll()
    {
        view.SetHP(model.hp, model.maxHP);
        view.SetMP(model.mp, model.maxMP);
        view.SetLevel(model.level);
        view.SetExp(model.exp / GetExpRequired(model.level));
    }

    int GetExpRequired(int level) => 100 * level * level;

    public void OnDestroy()
    {
        model.OnDataChanged -= OnModelChanged;
    }
}
```

**MVVM模式（Model-View-ViewModel）**：
- 通过数据绑定实现View和ViewModel的自动同步，减少手动刷新代码。
- Unity中可使用ReactiveProperty或自定义BindableProperty实现。
- 适合需要频繁UI刷新的项目（如属性面板、商城界面）。

```csharp
// 可绑定属性（简易版）
[System.Serializable]
public class BindableProperty<T>
{
    private T _value;
    public event Action<T> OnValueChanged;

    public T Value
    {
        get => _value;
        set
        {
            if (!EqualityComparer<T>.Default.Equals(_value, value))
            {
                _value = value;
                OnValueChanged?.Invoke(_value);
            }
        }
    }

    public BindableProperty(T initialValue = default)
    {
        _value = initialValue;
    }
}

// ViewModel
public class PlayerHUDViewModel
{
    public BindableProperty<int> HP = new BindableProperty<int>();
    public BindableProperty<int> MaxHP = new BindableProperty<int>();
    public BindableProperty<float> HPRatio = new BindableProperty<float>();

    // Model更新时自动计算派生属性
    public void UpdateFromModel(PlayerStatsModel model)
    {
        HP.Value = model.hp;
        MaxHP.Value = model.maxHP;
        HPRatio.Value = (float)model.hp / model.maxHP;
    }
}

// View绑定
public class PlayerHUD_MVVM : MonoBehaviour
{
    [SerializeField] private Slider hpBar;
    private PlayerHUDViewModel vm;

    public void Bind(PlayerHUDViewModel viewModel)
    {
        vm = viewModel;
        // 自动绑定：值变化时自动更新UI
        vm.HPRatio.OnValueChanged += (ratio) => hpBar.value = ratio;
    }
}
```

## 具体实现方法

### 界面基类设计（完整版）

```csharp
public abstract class UIBase : MonoBehaviour
{
    public string UIName { get; set; }
    public UILayer Layer { get; protected set; } = UILayer.Normal;

    // 缓存常用组件
    private CanvasGroup canvasGroup;
    public CanvasGroup Group
    {
        get
        {
            if (canvasGroup == null)
                canvasGroup = GetComponent<CanvasGroup>()
                    ?? gameObject.AddComponent<CanvasGroup>();
            return canvasGroup;
        }
    }

    private RectTransform rectTransform;
    public RectTransform Rect
    {
        get
        {
            if (rectTransform == null)
                rectTransform = GetComponent<RectTransform>();
            return rectTransform;
        }
    }

    // 生命周期钩子
    public virtual void OnInit() { }
    public virtual void OnShow(params object[] args) { Group.alpha = 1f; Group.interactable = true; }
    public virtual void OnHide() { Group.alpha = 0f; Group.interactable = false; }
    public virtual void OnClose() { }

    // 便捷方法：自动绑定按钮事件（避免手动拖拽）
    protected void BindButton(string path, UnityEngine.Events.UnityAction action)
    {
        Transform t = transform.Find(path);
        if (t == null) { Debug.LogWarning($"Button not found: {path}"); return; }
        Button btn = t.GetComponent<Button>();
        if (btn != null) btn.onClick.AddListener(action);
    }

    // 便捷方法：安全关闭自己
    protected void CloseSelf()
    {
        UIManager.Instance.CloseUI(UIName);
    }
}
```

### 界面打开/关闭完整流程

```
打开流程（OpenUI）:
1. 检查是否已打开（openedUIs字典查找） -> 已打开则调OnShow并返回
2. 检查是否已预加载（内存缓存中查找预制体）
3. 若不存在则从Addressables异步加载（带加载中UI）
4. 实例化到对应层级Canvas下
5. 调用OnInit()进行初始化（绑定按钮、获取组件引用）
6. 调用OnShow(args)播放打开动画并设置数据
7. 记录到openedUIs字典
8. 弹窗类型入栈

关闭流程（CloseUI）:
1. 调用OnHide()播放关闭动画
2. 动画完成后调用OnClose()清理资源
3. 从openedUIs移除
4. 弹窗出栈
5. Destroy GameObject + Release Addressable句柄
```

## 性能基准数据

| 操作 | 耗时参考 | 说明 |
|------|---------|------|
| Addressables首次加载预制体 | 5-30ms | 取决于资源大小和存储介质 |
| Addressables缓存命中加载 | 1-5ms | 内存中已有缓存 |
| Instantiate实例化 | 0.5-2ms | 取决于预制体复杂度 |
| Canvas.BuildBatch (Rebuild) | 1-15ms | 取决于Canvas元素数量 |
| 单个UI面板OnShow | 0.1-1ms | 不含资源加载 |

## 最佳实践

- 每个独立Canvas的UI只在需要时激活，减少合批打断
- 避免在UI根节点滥用Canvas组件，尽量嵌套在已有Canvas下
- 频繁变化的UI元素（如倒计时、实时伤害数字）使用独立Canvas隔离
- 使用接口（IPlayerHUDView）而非具体类型解耦UI与业务逻辑
- 界面基类提供统一的生命周期管理，所有UI面板继承自UIBase
- 弹窗使用栈管理，支持手机返回键逐层关闭
- Addressables加载失败时有降级方案（显示错误提示而非崩溃）

## 常见陷阱与修复

**陷阱1：所有界面放在同一个Canvas下**
- 症状：任何元素变化都会触发全量Rebuild，Profiler中Canvas.BuildBatch耗时高
- 修复：按更新频率拆分Canvas，HUD和背包分属不同Canvas

**陷阱2：UI脚本中深层引用业务单例**
- 症状：`GameManager.Instance.player.inventory.GetItem("sword").stats["atk"]` 这样的调用导致强耦合
- 修复：通过Presenter/ViewModel中转，UI只依赖接口

**陷阱3：界面关闭时未清理事件监听**
- 症状：关闭界面后仍收到回调，访问已销毁对象导致NullReferenceException
- 修复：在OnClose()中取消所有事件订阅，使用OnDestroy作为兜底

**陷阱4：同步加载大量UI资源**
- 症状：打开背包界面卡顿100-300ms
- 修复：使用Addressables异步加载，显示加载动画
