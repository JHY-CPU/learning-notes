# UMG UI系统

## 核心概念

UMG (Unreal Motion Graphics)是Unreal的UI框架，基于Slate构建。UI通过Widget Blueprint设计，支持数据绑定和动画。UMG运行在游戏线程上，由PlayerController的HUD管理。

## Widget Blueprint

Widget Blueprint是UMG的设计单元，包含可视化布局和交互逻辑：

```cpp
// C++基类（可选，蓝图Widget可继承此类）
UCLASS()
class YOURGAME_API UMyWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    // 绑定蓝图中的控件
    // 使用 meta = (BindWidget) 绑定同名控件
    UPROPERTY(meta = (BindWidget))
    TObjectPtr<UTextBlock> ScoreText;

    UPROPERTY(meta = (BindWidget))
    TObjectPtr<UButton> PauseButton;

    UPROPERTY(meta = (BindWidgetOptional)) // 可选绑定，蓝图中可以没有
    TObjectPtr<UImage> BackgroundImage;

    virtual void NativeConstruct() override
    {
        Super::NativeConstruct();

        // 绑定按钮事件
        if (PauseButton)
        {
            PauseButton->OnClicked.AddDynamic(this, &UMyWidget::OnPauseClicked);
        }
    }

    UFUNCTION()
    void OnPauseClicked()
    {
        UE_LOG(LogTemp, Log, TEXT("Pause button clicked"));
    }

    // 更新UI
    UFUNCTION(BlueprintCallable)
    void UpdateScore(int32 NewScore)
    {
        if (ScoreText)
        {
            ScoreText->SetText(FText::AsNumber(NewScore));
        }
    }
};

// 创建和显示Widget
void AMyPlayerController::ShowHUD()
{
    if (HUDWidgetClass)
    {
        HUDWidget = CreateWidget<UMyWidget>(this, HUDWidgetClass);
        HUDWidget->AddToViewport();

        // 设置ZOrder控制层级
        HUDWidget->SetZOrder(10);
    }
}

// 移除Widget
void AMyPlayerController::HideHUD()
{
    if (HUDWidget)
    {
        HUDWidget->RemoveFromParent();
    }
}
```

## UMG核心组件

| 组件 | 功能 | 典型用途 |
|------|------|----------|
| UTextBlock | 文本显示 | 分数、对话、标签 |
| UButton | 可点击按钮 | 菜单按钮 |
| UImage | 图片显示 | 图标、背景 |
| UEditableTextBox | 可编辑文本框 | 聊天输入、昵称 |
| UProgressBar | 进度条 | 血条、加载进度 |
| USlider | 滑块 | 音量控制 |
| UComboBoxString | 下拉选择 | 语言选择 |
| UListView | 背包/列表 | 物品列表 |
| UCanvasPanel | 绝对定位画布 | 主布局容器 |
| UVerticalBox | 垂直排列 | 菜单列表 |
| UHorizontalBox | 水平排列 | 工具栏 |
| UUniformGridPanel | 均匀网格 | 背包格子 |
| UOverlay | 叠加层 | 标签叠加在图标上 |
| UScrollBox | 可滚动区域 | 长列表、聊天 |

## UI动画

```cpp
// Widget中使用Timeline/Animation
// 在Widget Blueprint的Animation面板创建

UCLASS()
class UMyWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    virtual void NativeConstruct() override
    {
        Super::NativeConstruct();

        // 播放Widget动画
        if (FadeInAnimation)
        {
            PlayAnimation(FadeInAnimation);
        }
    }

    // 绑定蓝图中定义的动画
    UPROPERTY(Transient, meta = (BindWidgetAnim))
    TObjectPtr<UWidgetAnimation> FadeInAnimation;

    UPROPERTY(Transient, meta = (BindWidgetAnim))
    TObjectPtr<UWidgetAnimation> SlideInAnimation;

    void PlaySlideIn()
    {
        PlayAnimation(SlideInAnimation, 0.f, 1, EUMGSequencePlayMode::Forward, 1.f);
    }

    // 带完成回调的播放
    void PlayFadeInWithCallback()
    {
        PlayAnimation(FadeInAnimation);
        // 或使用委托
        FWidgetAnimationDynamicEvent EndEvent;
        EndEvent.BindDynamic(this, &UMyWidget::OnFadeInComplete);
        BindToAnimationFinished(FadeInAnimation, EndEvent);
    }

    UFUNCTION()
    void OnFadeInComplete()
    {
        UE_LOG(LogTemp, Log, TEXT("Fade in animation complete"));
    }
};
```

## 控件样式与主题

```cpp
// 通过Slate样式自定义控件外观
// 1. 在Content Browser创建Widget Style资产
// 2. 配置Normal/Hovered/Pressed/Disabled状态的外观
// 3. 在控件的Style属性中引用

// 运行时创建动态材质
void UMyWidget::CreateDynamicBackground()
{
    UMaterialInstanceDynamic* DynMat = UMaterialInstanceDynamic::Create(BackgroundMaterial, this);
    DynMat->SetVectorParameterValue("BaseColor", FLinearColor::Blue);
    BackgroundImage->SetBrushFromMaterial(DynMat);
}

// 数据绑定
void UMyWidget::BindToPlayerState(AMyPlayerState* PS)
{
    if (PS)
    {
        // 使用属性观察（UE5.5+）
        // 或手动轮询更新
    }
}
```

## 输入模式

```cpp
void AMyPlayerController::SetupInputMode()
{
    // 游戏模式（无鼠标光标）
    FInputModeGameOnly GameMode;
    SetInputMode(GameMode);

    // UI模式（显示鼠标光标，只响应UI）
    FInputModeUIOnly UIMode;
    UIMode.SetWidgetToFocus(HUDWidget->TakeWidget());
    SetInputMode(UIMode);

    // 混合模式（游戏+UI）
    FInputModeGameAndUI GameAndUI;
    GameAndUI.SetHideCursorDuringCapture(false);
    SetInputMode(GameAndUI);
}
```

## 常见陷阱与最佳实践

1. **BindWidget命名必须完全匹配**: C++中声明的变量名必须与蓝图中控件名一致
2. **Widget销毁时机**: 切换关卡前主动移除Widget，避免引用残留
3. **性能考虑**: 频繁更新的UI使用缓存值+脏标记，不要每帧SetText
4. **层级管理**: 使用ZOrder或Canvas Panel的Z排序管理UI层级
5. **输入消费**: UI会消费点击事件，注意设置bIsFocusable和Input Mode

## 与其他系统的关联

- **Slate**: UMG底层基于Slate UI框架
- **输入系统**: UMG通过PlayerController获取输入
- **动画系统**: Widget动画是独立的，不依赖AnimGraph
