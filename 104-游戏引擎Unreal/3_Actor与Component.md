# Actor与Component

## 核心概念

Actor是Unreal中所有可放置到世界中的对象的基类。Actor本身不含位置信息，通过挂载SceneComponent获得空间属性。Component提供Actor的功能模块，类似Unity的组件化设计，但有更严格的类型系统。

## Actor生命周期

Actor从创建到销毁经历多个阶段，理解生命周期对正确初始化和清理至关重要：

```cpp
// 完整生命周期顺序
// 1. 构造函数 (Constructor)
// 2. PostInitProperties()
// 3. BeginPlay()
// 4. Tick()  每帧
// 5. EndPlay()
// 6. Destroy()
// 7. BeginDestroy()
// 8. FinishDestroy()
```

```cpp
UCLASS()
class YOURGAME_API AMyActor : public AActor
{
    GENERATED_BODY()

public:
    // 1. 构造函数 - 设置默认值和组件
    AMyActor()
    {
        // 创建子对象组件（必须在构造函数中）
        RootScene = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
        RootComponent = RootScene;

        MeshComp = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
        MeshComp->SetupAttachment(RootComponent);

        // 默认开启Tick
        PrimaryActorTick.bCanEverTick = true;
        PrimaryActorTick.TickInterval = 0.0f; // 每帧Tick（设置>0可降低频率）
    }

    // 2. PostInitProperties - 属性初始化后
    virtual void PostInitProperties() override
    {
        Super::PostInitProperties();
        // 此时UPROPERTY属性已从CDO加载
    }

    // 3. BeginPlay - 游戏开始
    virtual void BeginPlay() override
    {
        Super::BeginPlay();
        // 游戏逻辑初始化，此时所有Actor都已Spawn
        UE_LOG(LogTemp, Log, TEXT("%s BeginPlay"), *GetName());
    }

    // 4. Tick - 每帧调用
    virtual void Tick(float DeltaTime) override
    {
        Super::Tick(DeltaTime);
        // 游戏逻辑更新
    }

    // 5. EndPlay - 游戏结束或Actor销毁
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override
    {
        Super::EndPlay(EndPlayReason);
        // 清理资源、取消委托绑定
        switch (EndPlayReason)
        {
        case EEndPlayReason::Destroyed:
            break;
        case EEndPlayReason::LevelTransition:
            break;
        case EEndPlayReason::EndPlayInEditor:
            break;
        }
    }

private:
    UPROPERTY(VisibleAnywhere)
    TObjectPtr<USceneComponent> RootScene;

    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UStaticMeshComponent> MeshComp;
};
```

## SceneComponent层级

SceneComponent是具有空间属性（位置、旋转、缩放）的组件，构成Actor的空间层级：

```cpp
// 组件附着关系
UCLASS()
class AWeapon : public AActor
{
    GENERATED_BODY()

    AWeapon()
    {
        // 根组件
        WeaponMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("WeaponMesh"));
        RootComponent = WeaponMesh;

        // 枪口位置组件
        MuzzleLocation = CreateDefaultSubobject<USceneComponent>(TEXT("Muzzle"));
        MuzzleLocation->SetupAttachment(WeaponMesh);
        MuzzleLocation->SetRelativeLocation(FVector(100.f, 0, 0));

        // 粒子效果附着在枪口
        MuzzleFlash = CreateDefaultSubobject<UParticleSystemComponent>(TEXT("MuzzleFlash"));
        MuzzleFlash->SetupAttachment(MuzzleLocation);
    }

    UPROPERTY(VisibleAnywhere)
    TObjectPtr<USkeletalMeshComponent> WeaponMesh;

    UPROPERTY(VisibleAnywhere)
    TObjectPtr<USceneComponent> MuzzleLocation;

    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UParticleSystemComponent> MuzzleFlash;
};

// 运行时附着
void AMyCharacter::EquipWeapon(AWeapon* NewWeapon)
{
    // 将武器附着到角色骨骼
    NewWeapon->AttachToComponent(
        GetMesh(),
        FAttachmentTransformRules::SnapToTargetNotIncludingScale,
        FName("RightHandSocket")
    );

    // 附着规则：
    // SnapToTargetNotIncludingScale - 吸附到目标，保持自身缩放
    // SnapToTargetIncludingScale - 吸附到目标，包含目标缩放
    // KeepWorldTransform - 保持世界坐标不变
    // KeepRelativeTransform - 保持相对坐标不变
}
```

## Tick机制

Tick是每帧更新逻辑的核心，但有性能开销需谨慎使用：

```cpp
// Tick配置选项
PrimaryActorTick.bCanEverTick = true;  // 能否Tick
PrimaryActorTick.bStartWithTickEnabled = true; // 初始是否启用
PrimaryActorTick.TickInterval = 0.1f;  // Tick间隔（秒），>0降低频率
PrimaryActorTick.TickGroup = TG_PrePhysics; // Tick组

// Tick组顺序
// TG_PrePhysics    - 物理模拟前
// TG_DuringPhysics - 物理模拟中
// TG_PostPhysics   - 物理模拟后
// TG_PostUpdateWork - 最后执行

// 运行时启用/禁用Tick
SetActorTickEnabled(false);
SetActorTickInterval(0.5f); // 降低Tick频率
```

## SpawnActor动态创建

```cpp
// 基本Spawn
AMyActor* NewActor = GetWorld()->SpawnActor<AMyActor>(
    SpawnLocation,
    SpawnRotation
);

// 带参数的Spawn
FActorSpawnParameters SpawnParams;
SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
SpawnParams.Owner = this;
SpawnParams.Instigator = GetInstigator();

AMyActor* Actor = GetWorld()->SpawnActor<AMyActor>(
    AMyActor::StaticClass(),
    Location,
    Rotation,
    SpawnParams
);

// 销毁
Actor->Destroy();
// 或延迟销毁
Actor->SetLifeSpan(5.f); // 5秒后自动销毁
```

## 常见陷阱与最佳实践

1. **CreateDefaultSubobject只能在构造函数中**: 运行时创建组件用NewObject+RegisterComponent
2. **Tick性能**: 大量Actor同时Tick会导致帧率下降，用TickInterval或事件驱动替代
3. **组件注册**: 运行时添加的组件必须调用RegisterComponent()才能生效
4. **RootComponent必须设置**: 没有RootComponent的Actor无法被正确附着
5. **SpawnActor时机**: BeginPlay中Spawn其他Actor可能产生顺序问题

## 与其他系统的关联

- **GamePlay框架**: PlayerController、Pawn等都是Actor的子类
- **物理系统**: UPrimitiveComponent（Mesh等）参与物理模拟
- **网络同步**: Actor是网络复制的基本单位
