# 物理系统(Chaos)

## 核心概念

UE5使用Chaos Physics作为默认物理引擎，替代了UE4的PhysX。Chaos支持刚体模拟、布料、破坏系统、车辆等。物理系统通过PrimitiveComponent参与，通过碰撞通道和响应矩阵控制碰撞行为。

## 刚体模拟

```cpp
UCLASS()
class APhysicsObject : public AActor
{
    GENERATED_BODY()

public:
    APhysicsObject()
    {
        MeshComp = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
        RootComponent = MeshComp;

        // 启用物理模拟
        MeshComp->SetSimulatePhysics(true);

        // 物理属性
        MeshComp->SetMassOverrideInKg(NAME_None, 10.f); // 质量10kg
        MeshComp->SetLinearDamping(0.1f);    // 线性阻力
        MeshComp->SetAngularDamping(0.05f);  // 角阻力
        MeshComp->BodyInstance.bUseCCD = true; // 连续碰撞检测

        // 重力
        MeshComp->SetEnableGravity(true);

        // 约束（限制运动轴）
        MeshComp->BodyInstance.bLockXTranslation = false;
        MeshComp->BodyInstance.bLockYTranslation = false;
        MeshComp->BodyInstance.bLockZTranslation = true;  // 锁定Z轴移动
        MeshComp->BodyInstance.bLockXRotation = true;     // 锁定X轴旋转
        MeshComp->BodyInstance.bLockYRotation = true;
        MeshComp->BodyInstance.bLockZRotation = false;
    }

    // 施加力
    void ApplyForce()
    {
        // 力（受质量影响）
        MeshComp->AddForce(FVector(1000.f, 0, 0));

        // 冲量（瞬时）
        MeshComp->AddImpulse(FVector(0, 0, 500.f));

        // 在指定位置施加力
        MeshComp->AddForceAtLocation(FVector(0, 0, 1000.f), ImpactPoint);

        // 设置速度
        MeshComp->SetPhysicsLinearVelocity(FVector(500.f, 0, 0));
        MeshComp->SetPhysicsAngularVelocityInDegrees(FVector(0, 0, 180.f));
    }

private:
    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UStaticMeshComponent> MeshComp;
};
```

## 碰撞响应

碰撞通过碰撞通道(Collision Channel)和响应矩阵控制：

```cpp
// 碰撞预设 (Collision Presets)
// BlockAll - 阻挡一切
// OverlapAll - 与一切重叠
// NoCollision - 无碰撞
// Pawn - 角色预设
// PhysicsActor - 物理Actor预设
// Custom - 自定义通道和响应

void SetupCollision()
{
    // 设置碰撞预设
    MeshComp->SetCollisionProfileName(TEXT("PhysicsActor"));

    // 自定义碰撞响应
    MeshComp->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
    MeshComp->SetCollisionObjectType(ECC_PhysicsBody);

    // 对特定通道的响应
    MeshComp->SetCollisionResponseToChannel(ECC_Visibility, ECR_Block);
    MeshComp->SetCollisionResponseToChannel(ECC_Pawn, ECR_Overlap);
    MeshComp->SetCollisionResponseToChannel(ECC_WorldStatic, ECR_Block);

    // 对所有通道设置响应
    MeshComp->SetCollisionResponseToAllChannels(ECR_Block);
    MeshComp->SetCollisionResponseToChannel(ECC_Camera, ECR_Ignore);

    // 自定义碰撞通道（在Project Settings -> Collision中配置）
    // ECollisionChannel枚举值ECC_GameTraceChannel1等可自定义
}

// 碰撞事件绑定
UCLASS()
class ACollisionActor : public AActor
{
    GENERATED_BODY()

    ACollisionActor()
    {
        MeshComp = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
        RootComponent = MeshComp;

        // 开启碰撞事件生成
        MeshComp->SetNotifyRigidBodyCollision(true); // Hit事件
        MeshComp->SetGenerateOverlapEvents(true);    // Overlap事件
    }

    virtual void BeginPlay() override
    {
        Super::BeginPlay();

        // 绑定碰撞事件
        MeshComp->OnComponentHit.AddDynamic(this, &ACollisionActor::OnHit);
        MeshComp->OnComponentBeginOverlap.AddDynamic(this, &ACollisionActor::OnOverlapBegin);
        MeshComp->OnComponentEndOverlap.AddDynamic(this, &ACollisionActor::OnOverlapEnd);
    }

    // Hit事件（阻挡碰撞）
    UFUNCTION()
    void OnHit(UPrimitiveComponent* HitComp, AActor* OtherActor,
               UPrimitiveComponent* OtherComp, FVector NormalImpulse,
               const FHitResult& Hit)
    {
        UE_LOG(LogTemp, Log, TEXT("Hit: %s at %s"), *OtherActor->GetName(), *Hit.ImpactPoint.ToString());
    }

    // Overlap事件（重叠碰撞）
    UFUNCTION()
    void OnOverlapBegin(UPrimitiveComponent* OverlappedComp, AActor* OtherActor,
                        UPrimitiveComponent* OtherComp, int32 OtherBodyIndex,
                        bool bFromSweep, const FHitResult& SweepResult)
    {
        UE_LOG(LogTemp, Log, TEXT("Overlap: %s"), *OtherActor->GetName());
    }

    UFUNCTION()
    void OnOverlapEnd(UPrimitiveComponent* OverlappedComp, AActor* OtherActor,
                      UPrimitiveComponent* OtherComp, int32 OtherBodyIndex)
    {
    }

private:
    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UStaticMeshComponent> MeshComp;
};
```

## 物理材质

```cpp
// 创建物理材质 (UMaterialInterface -> Physics Material)
// 在Content Browser: Create -> Physics -> Physical Material

void ApplyPhysicalMaterial()
{
    // 物理材质属性
    // Friction - 摩擦力
    // Restitution - 弹性系数 (0-1)
    // Density - 密度

    UPhysicalMaterial* PhysMat = NewObject<UPhysicalMaterial>();
    PhysMat->Friction = 0.5f;
    PhysMat->Restitution = 0.8f; // 弹性

    // 应用到材质
    UMaterialInstanceDynamic* DynMat = UMaterialInstanceDynamic::Create(BaseMaterial, this);
    DynMat->SetPhysicalMaterial(PhysMat);
    MeshComp->SetMaterial(0, DynMat);
}
```

## 物理约束 (Constraint)

```cpp
// Physics Constraint (物理约束)用于连接两个刚体
void SetupConstraint()
{
    UPhysicsConstraintComponent* Constraint = CreateDefaultSubobject<UPhysicsConstraintComponent>(TEXT("Constraint"));
    Constraint->SetupAttachment(RootComponent);

    // 连接两个组件
    Constraint->SetConstrainedComponents(MeshA, NAME_None, MeshB, NAME_None);

    // 约束类型
    // Linear Limit - 线性限制（距离）
    Constraint->SetLinearXLimit(ELinearConstraintMotion::LCM_Limited, 100.f);
    Constraint->SetLinearYLimit(ELinearConstraintMotion::LCM_Locked, 0.f);
    Constraint->SetLinearZLimit(ELinearConstraintMotion::LCM_Limited, 50.f);

    // Angular Limit - 角度限制
    Constraint->SetAngularSwing1Limit(EAngularConstraintMotion::ACM_Limited, 45.f);
    Constraint->SetAngularTwistLimit(EAngularConstraintMotion::ACM_Limited, 30.f);

    // 阻尼和弹性
    Constraint->SetLinearDriveParams(100.f, 20.f, 0.f); // Stiffness, Damping, ForceLimit
}
```

## 射线检测

```cpp
// Line Trace（射线检测）
void PerformLineTrace()
{
    FVector Start = Camera->GetComponentLocation();
    FVector End = Start + Camera->GetForwardVector() * 10000.f;

    FHitResult HitResult;
    FCollisionQueryParams Params;
    Params.AddIgnoredActor(this); // 忽略自身

    bool bHit = GetWorld()->LineTraceSingleByChannel(
        HitResult, Start, End, ECC_Visibility, Params
    );

    if (bHit)
    {
        AActor* HitActor = HitResult.GetActor();
        FVector HitLocation = HitResult.ImpactPoint;
        FVector HitNormal = HitResult.ImpactNormal;
    }

    // 调试绘制
    DrawDebugLine(GetWorld(), Start, End, FColor::Green, false, 2.f);
}

// Shape Trace（形状检测）
void PerformSphereTrace()
{
    FCollisionShape Sphere = FCollisionShape::MakeSphere(50.f);
    FHitResult HitResult;

    bool bHit = GetWorld()->SweepSingleByChannel(
        HitResult, Start, End, FQuat::Identity,
        ECC_Pawn, Sphere, Params
    );
}
```

## 常见陷阱与最佳实践

1. **Chaos vs PhysX**: UE5.0+默认Chaos，部分PhysX API已废弃
2. **物理帧率**: Chaos使用固定时间步，与游戏帧率独立
3. **CCD对性能影响大**: 仅高速运动物体开启连续碰撞检测
4. **碰撞事件开销**: 大量物体开启碰撞事件会增加CPU开销
5. **物理线程**: Chaos物理在独立线程运行，注意线程安全

## 与其他系统的关联

- **动画系统**: 物理驱动的动画(Ragdoll)需要Physics Asset
- **AI系统**: EQS查询可考虑物理碰撞
- **Niagara**: 粒子系统可与物理系统交互
