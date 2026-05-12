# UI动画与过渡

## 核心概念

UI动画是提升用户体验和操作反馈感的关键手段。优秀的UI动画应当让玩家感知到界面的状态变化（出现、消失、响应），同时不能影响操作流畅性。动画时长通常在0.1s-0.5s之间，过长会让玩家感到迟滞。

### UI动画设计原则

- **即时反馈**：按钮按下立即有缩放或颜色变化（0.05-0.1s）
- **状态过渡**：界面打开/关闭有平滑过渡（0.2-0.4s）
- **注意力引导**：重要提示用呼吸/脉冲动画吸引注意力
- **层次感**：多个元素依次出现，建立视觉层次
- **一致性**：同类动画使用相同的时长和缓动曲线，建立玩家预期

### 常见UI动画类型及性能开销

| 动画类型 | 实现方式 | CPU开销 | GPU开销 | 适用场景 |
|---------|---------|--------|--------|---------|
| Scale弹出 | Transform.localScale | 极低 | 低 | 弹窗、按钮 |
| 透明度渐变 | CanvasGroup.alpha | 低 | 低 | 淡入淡出、遮罩 |
| 位移滑入 | RectTransform.anchoredPosition | 极低 | 极低 | 侧边栏、通知 |
| 序列动画 | 多个DOTween序列 | 中等 | 低 | 列表项依次出现 |
| 粒子特效 | ParticleSystem | 高 | 高 | 获得稀有物品 |
| 骨骼动画 | Animator | 中等 | 低 | 角色表情、Logo |

### DOTween在UI中的深度应用

DOTween是Unity中最高效的补间动画库，对UGUI有专门优化。相比Animation组件，DOTween的内存分配和CPU开销低一个数量级。

```csharp
using DG.Tweening;
using UnityEngine;

/// <summary>
/// 完整的UI弹窗动画控制器
/// 支持打开动画、关闭动画、中断恢复
/// </summary>
public class PopupAnimator : MonoBehaviour
{
    [Header("打开动画")]
    [SerializeField] private float showDuration = 0.3f;
    [SerializeField] private Ease showEase = Ease.OutBack;
    [SerializeField] private Vector3 showFromScale = Vector3.zero;

    [Header("关闭动画")]
    [SerializeField] private float hideDuration = 0.2f;
    [SerializeField] private Ease hideEase = Ease.InBack;

    [Header("滑入动画（可选）")]
    [SerializeField] private bool useSlide;
    [SerializeField] private Vector2 slideFrom = new Vector2(0, -200);

    private RectTransform rectTransform;
    private CanvasGroup canvasGroup;
    private Vector2 originalPosition;
    private Sequence currentSequence;

    void Awake()
    {
        rectTransform = GetComponent<RectTransform>();
        canvasGroup = GetComponent<CanvasGroup>();
        if (canvasGroup == null)
            canvasGroup = gameObject.AddComponent<CanvasGroup>();
        originalPosition = rectTransform.anchoredPosition;
    }

    public void PlayShow(System.Action onComplete = null)
    {
        KillCurrentSequence();
        gameObject.SetActive(true);

        currentSequence = DOTween.Sequence();

        // 缩放动画
        rectTransform.localScale = showFromScale;
        currentSequence.Append(
            rectTransform.DOScale(Vector3.one, showDuration)
                .SetEase(showEase)
                .SetUpdate(true) // 忽略TimeScale，暂停菜单也能播放
        );

        // 透明度动画（同步进行）
        canvasGroup.alpha = 0f;
        currentSequence.Join(
            canvasGroup.DOFade(1f, showDuration * 0.8f)
        );

        // 滑入动画（可选）
        if (useSlide)
        {
            rectTransform.anchoredPosition = originalPosition + slideFrom;
            currentSequence.Join(
                rectTransform.DOAnchorPos(originalPosition, showDuration)
                    .SetEase(showEase)
            );
        }

        if (onComplete != null)
            currentSequence.OnComplete(() => onComplete());

        currentSequence.Play();
    }

    public void PlayHide(System.Action onComplete = null)
    {
        KillCurrentSequence();

        currentSequence = DOTween.Sequence();

        currentSequence.Append(
            rectTransform.DOScale(showFromScale, hideDuration)
                .SetEase(hideEase)
                .SetUpdate(true)
        );

        currentSequence.Join(
            canvasGroup.DOFade(0f, hideDuration)
        );

        if (useSlide)
        {
            currentSequence.Join(
                rectTransform.DOAnchorPos(
                    originalPosition + slideFrom, hideDuration)
            );
        }

        currentSequence.OnComplete(() =>
        {
            gameObject.SetActive(false);
            rectTransform.anchoredPosition = originalPosition;
            onComplete?.Invoke();
        });

        currentSequence.Play();
    }

    void KillCurrentSequence()
    {
        if (currentSequence != null && currentSequence.IsActive())
        {
            currentSequence.Kill();
            currentSequence = null;
        }
    }

    void OnDestroy()
    {
        KillCurrentSequence();
    }
}
```

## 具体实现方法

### 序列动画（Sequence）深度用法

序列动画用于多个UI元素的编排播放，常用于列表项依次出现、引导流程等：

```csharp
/// <summary>
/// 列表项依次弹入动画
/// 支持延迟、间隔、并行动画
/// </summary>
public class ListRevealAnimator : MonoBehaviour
{
    [SerializeField] private float itemDuration = 0.2f;
    [SerializeField] private float itemInterval = 0.05f;
    [SerializeField] private Ease itemEase = Ease.OutBack;
    [SerializeField] private float fromScale = 0.5f;

    private Sequence revealSequence;

    public void PlayReveal(List<RectTransform> items, System.Action onComplete = null)
    {
        KillSequence();

        revealSequence = DOTween.Sequence();

        for (int i = 0; i < items.Count; i++)
        {
            RectTransform item = items[i];
            item.localScale = Vector3.one * fromScale;
            item.GetComponent<CanvasGroup>().alpha = 0f;

            // 缩放动画
            revealSequence.Append(
                item.DOScale(1f, itemDuration).SetEase(itemEase)
            );

            // 透明度动画并行执行
            revealSequence.Join(
                item.GetComponent<CanvasGroup>().DOFade(1f, itemDuration)
            );

            // 每项之间的间隔
            if (i < items.Count - 1)
                revealSequence.AppendInterval(itemInterval);
        }

        if (onComplete != null)
            revealSequence.OnComplete(() => onComplete());

        revealSequence.Play();
    }

    public void KillSequence()
    {
        if (revealSequence != null && revealSequence.IsActive())
        {
            revealSequence.Kill();
            revealSequence = null;
        }
    }
}
```

### 通用动画组件（Inspector可配置）

```csharp
/// <summary>
/// 通用UI动画组件，在Inspector中配置动画类型
/// 无需编写代码即可为任何UI添加动画效果
/// </summary>
public class UIAnimation : MonoBehaviour
{
    public enum AnimType { Scale, Fade, SlideLeft, SlideRight, SlideUp, SlideDown }

    [Header("显示动画")]
    [SerializeField] private AnimType showAnim = AnimType.Scale;
    [SerializeField] private float showDuration = 0.3f;
    [SerializeField] private Ease showEase = Ease.OutBack;

    [Header("隐藏动画")]
    [SerializeField] private AnimType hideAnim = AnimType.Scale;
    [SerializeField] private float hideDuration = 0.2f;
    [SerializeField] private Ease hideEase = Ease.InBack;

    [Header("自动播放")]
    [SerializeField] private bool playOnEnable = true;

    private CanvasGroup canvasGroup;
    private RectTransform rectTransform;
    private Vector2 originalPos;
    private Sequence activeSequence;

    void Awake()
    {
        canvasGroup = GetComponent<CanvasGroup>();
        if (canvasGroup == null)
            canvasGroup = gameObject.AddComponent<CanvasGroup>();
        rectTransform = GetComponent<RectTransform>();
        originalPos = rectTransform.anchoredPosition;
    }

    void OnEnable()
    {
        if (playOnEnable) PlayShow();
    }

    void OnDisable()
    {
        KillActiveSequence();
    }

    public void PlayShow()
    {
        KillActiveSequence();
        activeSequence = DOTween.Sequence();
        float slideOffset = 300f;

        switch (showAnim)
        {
            case AnimType.Scale:
                rectTransform.localScale = Vector3.zero;
                activeSequence.Append(
                    rectTransform.DOScale(1f, showDuration).SetEase(showEase));
                break;
            case AnimType.Fade:
                canvasGroup.alpha = 0f;
                activeSequence.Append(
                    canvasGroup.DOFade(1f, showDuration));
                break;
            case AnimType.SlideLeft:
                rectTransform.anchoredPosition = originalPos + Vector2.left * slideOffset;
                activeSequence.Append(
                    rectTransform.DOAnchorPos(originalPos, showDuration).SetEase(showEase));
                break;
            case AnimType.SlideRight:
                rectTransform.anchoredPosition = originalPos + Vector2.right * slideOffset;
                activeSequence.Append(
                    rectTransform.DOAnchorPos(originalPos, showDuration).SetEase(showEase));
                break;
            case AnimType.SlideUp:
                rectTransform.anchoredPosition = originalPos + Vector2.up * slideOffset;
                activeSequence.Append(
                    rectTransform.DOAnchorPos(originalPos, showDuration).SetEase(showEase));
                break;
            case AnimType.SlideDown:
                rectTransform.anchoredPosition = originalPos + Vector2.down * slideOffset;
                activeSequence.Append(
                    rectTransform.DOAnchorPos(originalPos, showDuration).SetEase(showEase));
                break;
        }

        activeSequence.SetUpdate(true);
        activeSequence.Play();
    }

    public void PlayHide(System.Action onComplete = null)
    {
        KillActiveSequence();
        activeSequence = DOTween.Sequence();

        switch (hideAnim)
        {
            case AnimType.Scale:
                activeSequence.Append(
                    rectTransform.DOScale(0f, hideDuration).SetEase(hideEase));
                break;
            case AnimType.Fade:
                activeSequence.Append(
                    canvasGroup.DOFade(0f, hideDuration));
                break;
            default:
                activeSequence.Append(
                    canvasGroup.DOFade(0f, hideDuration));
                break;
        }

        activeSequence.SetUpdate(true);
        activeSequence.OnComplete(() => onComplete?.Invoke());
        activeSequence.Play();
    }

    void KillActiveSequence()
    {
        if (activeSequence != null && activeSequence.IsActive())
        {
            activeSequence.Kill();
            activeSequence = null;
        }
    }
}
```

### 自定义Animation Curve（不依赖DOTween时）

```csharp
/// <summary>
/// 使用Unity内置AnimationCurve实现弹性效果
/// 适用于不想引入DOTween依赖的项目
/// </summary>
public class CurveBasedAnimator : MonoBehaviour
{
    [SerializeField] private AnimationCurve scaleCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
    [SerializeField] private AnimationCurve alphaCurve = AnimationCurve.Linear(0, 0, 1, 1);
    [SerializeField] private float duration = 0.3f;

    private RectTransform rectTransform;
    private CanvasGroup canvasGroup;
    private float elapsed;
    private bool isPlaying;

    void Awake()
    {
        rectTransform = GetComponent<RectTransform>();
        canvasGroup = GetComponent<CanvasGroup>();
    }

    public void Play()
    {
        elapsed = 0f;
        isPlaying = true;
        gameObject.SetActive(true);
    }

    void Update()
    {
        if (!isPlaying) return;

        elapsed += Time.unscaledDeltaTime;
        float t = Mathf.Clamp01(elapsed / duration);

        rectTransform.localScale = Vector3.one * scaleCurve.Evaluate(t);
        canvasGroup.alpha = alphaCurve.Evaluate(t);

        if (t >= 1f) isPlaying = false;
    }
}
```

### 呼吸/脉冲动画（提示用）

```csharp
/// <summary>
/// 循环呼吸/脉冲动画，用于提示玩家注意某个UI元素
/// </summary>
public class PulseAnimator : MonoBehaviour
{
    [SerializeField] private float minScale = 0.9f;
    [SerializeField] private float maxScale = 1.1f;
    [SerializeField] private float pulseSpeed = 2f;
    [SerializeField] private bool pulseAlpha = true;
    [SerializeField] private float minAlpha = 0.6f;
    [SerializeField] private float maxAlpha = 1f;

    private RectTransform rectTransform;
    private CanvasGroup canvasGroup;
    private Tweener scaleTweener;
    private Tweener alphaTweener;

    void Start()
    {
        rectTransform = GetComponent<RectTransform>();
        canvasGroup = GetComponent<CanvasGroup>();
    }

    public void StartPulse()
    {
        // 缩放脉冲
        scaleTweener = rectTransform
            .DOScale(maxScale, 1f / pulseSpeed)
            .SetEase(Ease.InOutSine)
            .SetLoops(-1, LoopType.Yoyo)
            .SetUpdate(true);

        // 透明度脉冲
        if (pulseAlpha && canvasGroup != null)
        {
            alphaTweener = canvasGroup
                .DOFade(maxAlpha, 1f / pulseSpeed)
                .SetEase(Ease.InOutSine)
                .SetLoops(-1, LoopType.Yoyo)
                .SetUpdate(true);
        }
    }

    public void StopPulse()
    {
        scaleTweener?.Kill();
        alphaTweener?.Kill();
        rectTransform.localScale = Vector3.one;
        if (canvasGroup != null) canvasGroup.alpha = 1f;
    }

    void OnDestroy()
    {
        StopPulse();
    }
}
```

## 动画性能基准

| 动画方案 | 每帧CPU开销 | 内存分配 | 100个UI同时动画 |
|---------|-----------|---------|---------------|
| DOTween | 0.1-0.3ms | 0 GC | 流畅60fps |
| Animator | 0.5-1ms | 0 GC | 勉强30fps |
| Animation组件 | 0.3-0.8ms | 少量GC | 勉强60fps |
| Coroutine逐帧 | 0.2-0.5ms | 每帧GC | 卡顿 |

## 最佳实践

- 所有UI动画使用`SetUpdate(true)`使其不受TimeScale影响，暂停菜单动画仍能播放
- 弹窗关闭动画播放完毕后再`SetActive(false)`，而非立即隐藏，避免动画中断
- 复杂动画使用Sequence统一管理，便于取消、回调和调试
- 为低端设备提供动画降级方案：跳过粒子效果、减少动画帧数、缩短时长
- 使用对象池管理频繁创建销毁的UI动画元素
- DOTween全局初始化一次，推荐在游戏启动时调用`DOTween.Init()`
- 循环动画（脉冲、呼吸）在OnDisable时必须Kill，否则内存泄漏

## 常见陷阱与修复

**陷阱1：动画播放期间直接禁用GameObject**
- 症状：DOTween抛出MissingReferenceException
- 修复：先Kill动画再禁用，或在OnDisable中调用`DOTween.Kill(transform)`

**陷阱2：多个DOTween动画同时操作同一属性**
- 症状：两个动画抢夺scale属性导致抖动或不可预测行为
- 修复：在新动画开始前Kill旧动画（`transform.DOKill()`），或使用`SetId`管理

**陷阱3：未在OnDestroy中调用`DOTween.Kill()`**
- 症状：场景切换后DOTween继续回调已销毁对象，导致异常
- 修复：在OnDestroy中调用`DOTween.KillAll()`或指定ID的Kill

**陷阱4：动画曲线设置不当导致视觉抖动**
- 症状：高刷新率设备（120Hz/144Hz）上动画出现肉眼可见的抖动
- 修复：使用平滑缓动曲线（Ease.InOutSine），避免线性插值；考虑动画帧率锁定
