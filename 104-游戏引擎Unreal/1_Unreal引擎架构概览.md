# Unreal引擎架构概览

## 核心概念

Unreal Engine采用模块化C++架构，所有功能以模块(Module)组织。引擎从启动到运行游戏有一套完整的引导流程，理解架构对深入开发至关重要。Unreal与Unity的最大区别在于：Unreal以C++为核心，蓝图作为可视化脚本层；而Unity以C#为核心，编辑器扩展也用C#。

## 模块系统

Unreal引擎和游戏本身都由模块构成。每个模块是一个独立的编译单元，有明确的依赖关系：

```
UnrealEngine/
├── Engine/
│   ├── Source/
│   │   ├── Runtime/          # 核心运行时模块
│   │   │   ├── Core/         # 核心基础（FString, TArray, TMap等）
│   │   │   ├── CoreUObject/  # UObject系统、反射、序列化
│   │   │   ├── Engine/       # 引擎核心（World, Actor, Component）
│   │   │   ├── InputCore/    # 输入系统
│   │   │   ├── RenderCore/   # 渲染核心
│   │   │   ├── RHI/          # 渲染硬件接口
│   │   │   └── ...
│   │   ├── Editor/           # 编辑器模块（打包时排除）
│   │   │   ├── UnrealEd/     # 编辑器主模块
│   │   │   ├── BlueprintEditor/
│   │   │   └── ...
│   │   └── Developer/        # 开发工具
│   └── Plugins/
├── Projects/
│   └── YourGame/
│       └── Source/
│           ├── YourGame/     # 运行时模块
│           ├── YourGameEditor/ # 编辑器模块
│           └── YourGame.Target.cs # 构建目标配置
```

### 模块依赖关系

模块之间的依赖是单向的，不能有循环依赖：

```
Core (最底层，无依赖)
  ↓
CoreUObject (依赖Core)
  ↓
Engine (依赖Core, CoreUObject)
  ↓
YourGame (依赖Engine, CoreUObject, 等)
  ↓
YourGameEditor (依赖YourGame, UnrealEd, 等)
```

### 模块构建文件 (Build.cs)

```csharp
// Source/YourGame/YourGame.Build.cs
using UnrealBuildTool;

public class YourGame : ModuleRules
{
    public YourGame(ReadOnlyTargetRules Target) : base(Target)
    {
        // PCH（预编译头文件）使用模式
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        // 公开依赖：其他模块引用YourGame时自动包含这些头文件
        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "UMG",           // UMG UI系统
            "AIModule",      // AI系统
            "NavigationSystem", // 导航系统
            "GameplayTags",  // Gameplay Tag系统
            "GameplayTasks"  // Gameplay Task系统
        });

        // 私有依赖：仅YourGame模块内部使用
        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "Slate",          // UI框架
            "SlateCore",      // Slate核心
            "RenderCore",     // 渲染核心
            "RHI"             // 渲染硬件接口
        });

        // 编辑器专用依赖（打包时自动排除）
        if (Target.bBuildEditor)
        {
            PrivateDependencyModuleNames.Add("UnrealEd");
        }

        // 编译选项
        bUseUnity = true;        // Unity Build（合并编译单元加速编译）
        MinFilesUsingPrecompiledHeaderOverride = 1;
        bEnforceIWYU = true;     // 强制 Include-What-You-Use
    }
}

// 构建目标配置 Source/YourGame/YourGame.Target.cs
using UnrealBuildTool;
[SupportedPlatforms(UnrealPlatformClass.All)]
public class YourGameTarget : TargetRules
{
    public YourGameTarget(TargetInfo Target) : base(Target)
    {
        Type = TargetType.Game;
        DefaultBuildSettings = BuildSettingsVersion.V4;
        IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_4;
    }
}
```

## 引擎启动流程详解

```
完整的Unreal Engine启动流程:

1. WinMain / 入口函数
   ↓
2. FEngineLoop::PreInit() - 引擎预初始化
   ├── GEngineLoop.PreInit(CommandLine)
   ├── 加载核心模块 (Core, CoreUObject, Engine)
   ├── 初始化内存分配器
   ├── 初始化日志系统
   ├── 初始化线程池
   ├── 初始化任务图(TaskGraph)
   └── 解析命令行参数和配置文件
   ↓
3. 引擎核心初始化
   ├── FModuleManager 加载所有模块
   ├── 初始化渲染系统 (RHI - Render Hardware Interface)
   │   ├── 选择图形API (DX11/DX12/Vulkan/Metal)
   │   ├── 创建渲染设备
   │   └── 初始化渲染线程
   ├── 初始化物理系统 (PhysX / Chaos)
   ├── 初始化音频系统
   └── 初始化网络系统
   ↓
4. 游戏初始化
   ├── 创建 GameEngine (UGameEngine或派生类)
   ├── 创建 GameInstance
   ├── 加载初始地图 (DefaultMap或命令行指定)
   ├── 创建 World / WorldContext
   ├── 创建 GameMode (服务端)
   ├── 创建 GameSession
   └── Spawn所有关卡中的Actor
   ↓
5. FEngineLoop::Tick() - 主循环 (每帧)
   ├── FTicker::Tick - Ticker回调
   ├── GFrameCounter++ - 帧计数递增
   ├── 处理输入 (Input Processing)
   ├── 游戏逻辑更新
   │   ├── GameInstance::Tick
   │   ├── World::Tick
   │   │   ├── LevelStreaming更新
   │   │   ├── Tick所有Actor (按TickGroup分组)
   │   │   ├── Component注册/注销
   │   │   └── Destroy待销毁的Actor
   │   └── HUD / PlayerController Tick
   ├── 物理模拟 (Chaos Physics)
   │   ├── Broad Phase
   │   ├── Narrow Phase
   │   ├── 约束求解
   │   └── 积分更新
   ├── 渲染
   │   ├── 游戏线程: Scene更新, 构建渲染命令
   │   ├── 渲染线程: 执行渲染命令
   │   └── RHI线程: 提交GPU命令
   ├── GC (垃圾回收 - UObject)
   │   ├── 标记阶段: 从根集遍历所有可达对象
   │   └── 清除阶段: 销毁不可达对象
   └── 帧率控制 / 等待VSync
   ↓
6. 引擎退出
   ├── 请求退出
   ├── 保存关卡
   ├── 销毁World
   ├── 卸载所有模块
   ├── 关闭渲染系统
   └── 关闭核心系统
```

## Tick组（Tick Groups）

Unreal的Actor Tick按预定义的组顺序执行，理解Tick顺序对避免依赖问题至关重要：

```cpp
// Tick组定义 (ETickingGroup)
enum ETickingGroup
{
    TG_PrePhysics,      // 物理模拟前（设置角色位置、动画更新）
    TG_StartPhysics,    // 物理模拟开始
    TG_DuringPhysics,   // 物理模拟中
    TG_EndPhysics,      // 物理模拟结束
    TG_PostPhysics,     // 物理模拟后（相机跟随、碰撞响应）
    TG_PostUpdateWork,  // 最终更新（动画LateUpdate等）
    TG_LastDemotable,   // 最后可降级的Tick

    TG_NewlySpawned,    // 新Spawn的Actor（下一帧才分配Tick组）
    TG_MAX
};

// 设置Actor的Tick组
void AMyActor::BeginPlay()
{
    Super::BeginPlay();

    // 将Tick设置为物理模拟后执行
    PrimaryActorTick.TickGroup = TG_PostPhysics;

    // 或者在Component中设置
    // MyComponent->PrimaryComponentTick.TickGroup = TG_PostPhysics;
}

// Tick执行顺序示例
void AMyActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    // 根据Tick组执行不同逻辑
    if (PrimaryActorTick.TickGroup == TG_PrePhysics)
    {
        // 在物理前更新位置（如动画驱动的位移）
    }
    else if (PrimaryActorTick.TickGroup == TG_PostPhysics)
    {
        // 在物理后处理结果（如相机跟随）
        // 此时刚体模拟已完成，可以读取最终位置
    }
}
```

## 游戏线程与渲染线程

Unreal使用多线程架构，游戏线程和渲染线程并行工作：

```
多线程架构:

游戏线程 (Game Thread)          渲染线程 (Render Thread)       RHI线程
├── 逻辑更新                    ├── 接收渲染命令               ├── 提交GPU命令
├── 物理模拟                    ├── 场景剔除                   ├── 状态切换
├── Actor Tick                  ├── 排序Draw Call             └── 同步GPU
├── 生成渲染命令 ───────────→   ├── 生成Mesh Draw Commands
└── 继续下一帧                  └── 提交给RHI线程 ──────────→

帧 N:  [Game Thread Work]────────[Render Thread Work N]────────[RHI Work N]
帧 N+1: [Game Thread Work]────────[Render Thread Work N+1]──────[RHI Work N+1]

注意: 游戏线程和渲染线程在同一帧的不同时刻工作
      游戏线程可能比渲染线程提前1-2帧
      使用ENQUEUE_RENDER_COMMAND同步两者
```

```cpp
// 在游戏线程中向渲染线程发送命令
void AMyActor::UpdateRenderData()
{
    // 方法1: Lambda表达式
    ENQUEUE_RENDER_COMMAND(MyRenderCommand)(
        [this](FRHICommandListImmediate& RHICmdList)
        {
            // 在渲染线程执行
            // 注意: 这里不能直接访问游戏线程的对象
            // 需要通过线程安全的方式传递数据
        }
    );

    // 方法2: 使用FRenderCommandFence等待渲染线程完成
    FRenderCommandFence Fence;
    Fence.BeginFence();
    Fence.Wait(); // 阻塞游戏线程直到渲染线程处理完之前的命令
}

// 游戏线程安全的数据传递
class FMyRenderData : public FRenderResource
{
public:
    TArray<FVector> Vertices;
    TArray<uint32> Indices;

    // 渲染资源初始化（在渲染线程调用）
    virtual void InitRHI() override
    {
        // 创建顶点缓冲区
        FRHIResourceCreateInfo CreateInfo(TEXT("MyBuffer"));
        VertexBufferRHI = RHICreateVertexBuffer(
            Vertices.Num() * sizeof(FVector),
            BUF_Static, CreateInfo);
    }

    // 渲染资源释放（在渲染线程调用）
    virtual void ReleaseRHI() override
    {
        VertexBufferRHI.SafeRelease();
    }
};
```

## 编辑器界面详解

| 面板 | 功能 | Unity对应 |
|------|------|-----------|
| Content Browser | 资源管理器，浏览和组织所有资源 | Project窗口 |
| World Outliner | 场景中Actor列表，层级视图 | Hierarchy窗口 |
| Details Panel | 选中对象的属性面板 | Inspector窗口 |
| Viewport | 3D场景编辑视图 | Scene窗口 |
| Level Blueprint | 关卡脚本（UE4概念，UE5中弱化） | 场景脚本 |
| World Settings | 关卡级别的GameMode覆盖等设置 | 场景设置 |
| Output Log | 输出日志窗口 | Console窗口 |
| World Browser | 大世界管理（World Partition） | 无直接对应 |

```cpp
// 自定义编辑器模块入口点
#include "Modules/ModuleManager.h"

class FYourGameEditorModule : public IModuleInterface
{
public:
    virtual void StartupModule() override
    {
        // 模块加载时执行
        // 注册编辑器扩展
        UE_LOG(LogTemp, Log, TEXT("YourGame Editor Module Started"));

        // 注册菜单扩展
        UToolMenus::RegisterStartupCallback(
            FSimpleMulticastDelegate::FDelegate::CreateRaw(
                this, &FYourGameEditorModule::RegisterMenus));
    }

    virtual void ShutdownModule() override
    {
        // 模块卸载时执行
        UToolMenus::UnRegisterStartupCallback(this);
    }

private:
    void RegisterMenus()
    {
        // 添加自定义菜单项
        UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Window");
        FToolMenuSection& Section = Menu->AddSection("YourGameTools",
            FText::FromString("Your Game Tools"));
    }
};

// 注册模块
IMPLEMENT_MODULE(FYourGameEditorModule, YourGameEditor);
```

## 核心头文件体系

```cpp
// ===== 最基础的头文件 =====
#include "CoreMinimal.h"         // 核心最小集合（TArray, FString, FVector等）
#include "GameFramework/Actor.h" // Actor基类
#include "UObject/ConstructorHelpers.h" // 构造函数中资源查找

// ===== 反射系统宏 =====
UCLASS(Blueprintable, BlueprintType, ClassGroup=(Custom), meta=(BlueprintSpawnableComponent))
class YOURGAME_API AMyActor : public AActor
{
    GENERATED_BODY()  // 必须包含，UHT（Unreal Header Tool）生成代码的位置

public:
    AMyActor();

    // 属性声明
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Combat")
    float Health = 100.f;

    // 函数声明
    UFUNCTION(BlueprintCallable, Category="Combat")
    void TakeDamage(float Amount);

    // 组件
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
    class UStaticMeshComponent* MeshComp;

    // 蓝图可重写事件
    UFUNCTION(BlueprintImplementableEvent, Category="Events")
    void OnHealthChanged(float OldHealth, float NewHealth);

    // C++默认实现，蓝图可覆盖
    UFUNCTION(BlueprintNativeEvent, Category="Events")
    void Die();
    virtual void Die_Implementation();

protected:
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    // 序列化（保存/加载）
    virtual void Serialize(FArchive& Ar) override;
};

// ===== USTRUCT示例 =====
USTRUCT(BlueprintType)
struct FCharacterStats
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float AttackPower = 10.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Defense = 5.f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    float Speed = 300.f;
};

// ===== UENUM示例 =====
UENUM(BlueprintType)
enum class ECharacterState : uint8
{
    Idle    UMETA(DisplayName = "空闲"),
    Moving  UMETA(DisplayName = "移动"),
    Attacking UMETA(DisplayName = "攻击"),
    Dead    UMETA(DisplayName = "死亡")
};

// ===== UINTERFACE示例 =====
UINTERFACE(MinimalAPI)
class UDamageable : public UInterface
{
    GENERATED_BODY()
};

class YOURGAME_API IDamageable
{
    GENERATED_BODY()

public:
    UFUNCTION(BlueprintNativeEvent, BlueprintCallable, Category="Combat")
    void ApplyDamage(float DamageAmount, AActor* DamageCauser);
};
```

## 核心容器与类型

```cpp
// ===== UE核心容器 =====

// TArray: 动态数组（类似std::vector但有额外优化）
TArray<int32> IntArray;
IntArray.Add(42);
IntArray.Emplace(100); // 原地构造，避免拷贝
IntArray.RemoveAt(0);
int32 Found = IntArray.Find(42);

// TMap: 哈希表（类似std::unordered_map）
TMap<FString, int32> ScoreMap;
ScoreMap.Add("Player1", 100);
ScoreMap.FindOrAdd("Player2", 0);

// TSet: 哈希集合
TSet<AActor*> OverlappingActors;

// TSharedPtr / TSharedRef: 智能引用计数指针
TSharedPtr<FMyData> Data = MakeShared<FMyData>();

// TWeakPtr: 弱引用
TWeakPtr<FMyData> WeakData = Data;

// TUniquePtr: 独占所有权
TUniquePtr<FMyResource> Resource = MakeUnique<FMyResource>();

// ===== UE核心类型 =====

FString Name = TEXT("Player");       // 字符串（TCHAR*兼容）
FName TagName = FName("MyTag");      // 不可变字符串池（高效查找）
FText DisplayName = FText::FromString("显示名称"); // 本地化文本

FVector Position(100.f, 200.f, 300.f);   // 3D向量
FRotator Rotation(0.f, 90.f, 0.f);       // 旋转（欧拉角）
FQuat QuatRotation = Rotation.Quaternion(); // 四元数
FTransform Transform(Rotation, Position);  // 变换（位置+旋转+缩放）

// UE中FVector是double精度(UE5)，float精度(UE4)
// UE5使用LWC(Large World Coordinates)支持大世界
```

## 常见陷阱与最佳实践

1. **头文件依赖**: 使用前向声明(`class AMyActor;`)替代include，减少编译时间。头文件改动会触发所有依赖文件重新编译
2. **模块循环依赖**: 模块间不能有循环依赖，需合理规划层级。如果A依赖B，B就不能依赖A
3. **PCH使用**: 正确配置PCH(Precompiled Header)可大幅加速编译。`PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs`
4. **Editor vs Runtime**: Editor模块不能包含在打包版本中，使用`if (WITH_EDITOR)`或Editor-only模块
5. **命令行工具**: 熟悉UnrealBuildTool和RunUAT提高构建效率
6. **热重载限制**: 添加新的UPROPERTY/UFUNCTION需要重新编译，不能热重载
7. **反射系统开销**: 过多的UCLASS/UFUNCTION会增加编译时间和运行时内存
8. **跨平台编译**: 注意`TEXT()`宏、`int32`/`uint32`类型的使用，避免平台差异

## 性能分析

| 操作 | 开销 | 说明 |
|------|------|------|
| Actor Tick | 低中 | 每帧调用，禁用不需要Tick的Actor |
| Component注册 | 中 | 动态添加Component有开销 |
| UFUNCTION调用 | 低 | 编译后为普通C++函数调用 |
| Blueprint函数调用 | 中高 | 解释执行，比C++慢5-10倍 |
| SpawnActor | 中 | 对象池可优化频繁Spawn/Destroy |
| 构造函数资源加载 | 低中 | Editor下同步加载，考虑异步方案 |

## 与其他系统的关联

- **UObject系统**: 所有引擎对象的基础，提供反射、序列化、GC
- **反射系统**: 通过UHT（Unreal Header Tool）在编译前生成反射代码
- **蓝图系统**: 蓝图节点的实现依赖模块导出的UFUNCTION
- **Chaos物理**: UE5的新物理引擎，替代PhysX
- **Nanite/Lumen**: UE5的核心渲染技术
