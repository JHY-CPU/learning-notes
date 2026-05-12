# UObject与反射系统

## 核心概念

UObject是Unreal对象系统的基类，提供了反射(Reflection)、序列化(Serialization)、垃圾回收(GC)、蓝图暴露等核心能力。所有需要引擎管理的C++类都应继承UObject。

## UPROPERTY宏

UPROPERTY控制属性在编辑器、蓝图和序列化系统中的行为：

```cpp
UCLASS()
class YOURGAME_API AMyActor : public AActor
{
    GENERATED_BODY()

public:
    // === 可见性 ===
    // VisibleAnywhere - 细节面板可见，不可编辑（组件常用）
    // EditAnywhere - 随处可编辑
    // EditDefaultsOnly - 仅类默认值可编辑（蓝图类中）
    // EditInstanceOnly - 仅实例可编辑（场景中）

    // === 蓝图访问 ===
    // BlueprintReadWrite - 蓝图可读写
    // BlueprintReadOnly - 蓝图只读

    // === 分类 ===
    UPROPERTY(EditAnywhere, Category = "Combat")
    float Damage = 10.f;

    UPROPERTY(EditAnywhere, Category = "Combat|Advanced")
    float CriticalMultiplier = 2.f;

    // === 元数据 ===
    UPROPERTY(EditAnywhere, meta = (ClampMin = "0", ClampMax = "100", UIMin = "0", UIMax = "100"))
    float HealthPercent = 100.f;

    // 工具提示
    UPROPERTY(EditAnywhere, meta = (ToolTip = "角色移动速度"))
    float MoveSpeed = 600.f;

    // 按条件显示
    UPROPERTY(EditAnywhere, Category = "Effects")
    bool bUseGlow = false;

    UPROPERTY(EditAnywhere, Category = "Effects", meta = (EditCondition = "bUseGlow"))
    FLinearColor GlowColor = FLinearColor::White;

    // === 高级特性 ===

    // 复制（网络同步）
    UPROPERTY(Replicated, BlueprintReadOnly)
    int32 TeamID;

    // 数组
    UPROPERTY(EditAnywhere)
    TArray<FString> InventoryItems;

    // Map
    UPROPERTY(EditAnywhere)
    TMap<FName, float> StatsMap;

    // Set
    UPROPERTY()
    TSet<AActor*> OverlappingActors;

    // 软引用（延迟加载）
    UPROPERTY(EditAnywhere)
    TSoftObjectPtr<UTexture2D> IconTexture;

    UPROPERTY(EditAnywhere)
    TSoftClassPtr<AActor> EnemyClass;

    // 对象引用（编辑器中可直接赋值）
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    TObjectPtr<USoundBase> FireSound;
};
```

## UFUNCTION宏

UFUNCTION控制函数的蓝图暴露和网络行为：

```cpp
UCLASS()
class YOURGAME_API AMyActor : public AActor
{
    GENERATED_BODY()

public:
    // 蓝图可调用，显示为执行节点
    UFUNCTION(BlueprintCallable, Category = "Combat")
    void ApplyDamage(AActor* Target, float Amount);

    // 纯函数，无副作用，显示为数据节点
    UFUNCTION(BlueprintPure, Category = "State")
    float GetHealthRatio() const;

    // 蓝图可实现事件（蓝图中实现，C++中不可调用）
    UFUNCTION(BlueprintImplementableEvent, Category = "Events")
    void OnDeathEffect();

    // C++默认实现，蓝图可覆盖
    UFUNCTION(BlueprintNativeEvent, Category = "Events")
    float CalculateDamage(float BaseDamage, AActor* Target);
    virtual float CalculateDamage_Implementation(float BaseDamage, AActor* Target);

    // 网络RPC
    UFUNCTION(Server, Reliable, WithValidation)
    void ServerFire(FVector Direction);
    bool ServerFire_Validate(FVector Direction) { return true; }
    void ServerFire_Implementation(FVector Direction);

    UFUNCTION(Client, Reliable)
    void ClientPlayEffect(UParticleSystem* Effect);

    UFUNCTION(NetMulticast, Unreliable)
    void MulticastPlaySound(USoundBase* Sound);

    // 绑定到输入
    UFUNCTION()
    void OnFirePressed();

    // 委托回调
    UFUNCTION()
    void OnOverlapBegin(AActor* OverlappedActor, AActor* OtherActor);
};
```

## 垃圾回收(GC)

Unreal使用标记-清除GC，自动管理UObject的生命周期：

```cpp
// GC规则：
// 1. 没有任何UPROPERTY引用指向的对象会被回收
// 2. AddToRoot() / RemoveFromRoot() 手动控制根对象
// 3. 使用TObjectPtr替代裸指针（UE5推荐）

UCLASS()
class AMyActor : public AActor
{
    GENERATED_BODY()

    // UPROPERTY引用 - GC会追踪，不会回收目标对象
    UPROPERTY()
    TObjectPtr<AMyActor> TargetActor;

    // 裸指针 - GC不追踪，可能导致悬空指针（危险！）
    // AMyActor* RawPointer; // 避免使用

    // TWeakObjectPtr - 弱引用，不阻止GC
    TWeakObjectPtr<AMyActor> WeakRef;

    void CheckWeakRef()
    {
        // 使用前检查有效性
        if (WeakRef.IsValid())
        {
            AM yActor* Actor = WeakRef.Get();
            // 安全使用
        }
    }

    // TStrongObjectPtr - 强引用（非UPROPERTY场景）
    TStrongObjectPtr<UObject> StrongRef;

    // FSoftObjectPath - 软路径，不阻止GC也不加载
    FSoftObjectPath SoftPath;
};
```

## 反射系统运行时查询

```cpp
// 获取类信息
UClass* Class = MyActor->GetClass();
FString ClassName = Class->GetName();

// 遍历属性
for (TFieldIterator<FProperty> It(Class); It; ++It)
{
    FProperty* Property = *It;
    FString PropertyName = Property->GetName();
    FString PropertyType = Property->GetCPPType();

    // 获取属性值
    void* ValuePtr = Property->ContainerPtrToValuePtr<void>(MyActor);
    FString ValueStr;
    Property->ExportTextItem_Direct(ValueStr, ValuePtr, nullptr, nullptr, PPF_None);
}

// 通过名称获取/设置属性
FProperty* Prop = Class->FindPropertyByName(FName("Health"));
float HealthValue;
Prop->GetValue_InContainer(MyActor, &HealthValue);

// 调用函数
UFunction* Func = Class->FindFunctionByName(FName("ApplyDamage"));
MyActor->ProcessEvent(Func, &Params);
```

## 常见陷阱与最佳实践

1. **TObjectPtr替代裸指针**: UE5中应全面使用TObjectPtr，编辑器中支持拖拽赋值
2. **GC时机不可控**: 不要在析构函数中依赖UObject引用，用BeginDestroy
3. **UHT处理宏**: UCLASS/UPROPERTY/UFUNCTION由UHT预处理，不能用C++条件编译包裹
4. **引用类型选择**: 强引用(UPROPERTY)、弱引用(TWeakObjectPtr)、软引用(FSoftObjectPath)按需选择
5. **Delegate安全**: 绑定到委托的UFUNCTION在对象销毁前必须解绑

## 与其他系统的关联

- **蓝图系统**: 反射信息驱动蓝图节点生成
- **序列化**: UPROPERTY控制哪些属性被保存到资产文件
- **网络同步**: Replicated属性通过反射系统进行序列化和同步
