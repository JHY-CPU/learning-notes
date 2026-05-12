# 动画系统(Animation Blueprint)

## 核心概念

Unreal动画系统基于Animation Blueprint (AnimBP)驱动。AnimBP包含AnimGraph（动画图）和EventGraph（事件图），通过状态机和动画节点控制骨骼动画的播放和混合。

## Animation Blueprint结构

```
Animation Blueprint
├── AnimGraph (动画图)
│   ├── State Machine (状态机)
│   │   ├── Idle State
│   │   ├── Walk State
│   │   └── Run State
│   ├── Blend Space (混合空间)
│   ├── Slot (动画插槽)
│   └── Output Pose (最终姿势)
├── EventGraph (事件图)
│   ├── BlueprintUpdateAnimation
│   ├── BlueprintInitializeAnimation
│   └── 自定义逻辑
└── Variables (变量)
    ├── Speed (float)
    ├── Direction (float)
    └── bIsInAir (bool)
```

## AnimGraph核心节点

```cpp
// C++ Animation Blueprint基类
UCLASS()
class UMyAnimInstance : public UAnimInstance
{
    GENERATED_BODY()

public:
    virtual void NativeInitializeAnimation() override
    {
        Super::NativeInitializeAnimation();
        // 初始化时获取角色引用
        OwningCharacter = Cast<AMyCharacter>(TryGetPawnOwner());
    }

    virtual void NativeUpdateAnimation(float DeltaSeconds) override
    {
        Super::NativeUpdateAnimation(DeltaSeconds);

        if (!OwningCharacter) return;

        // 更新AnimGraph使用的变量
        Speed = OwningCharacter->GetVelocity().Size();
        Direction = CalculateDirection(OwningCharacter->GetVelocity(),
                                       OwningCharacter->GetActorRotation());
        bIsInAir = OwningCharacter->GetCharacterMovement()->IsFalling();
        bIsCrouching = OwningCharacter->bIsCrouched;
        PitchAngle = OwningCharacter->GetBaseAimRotation().Pitch;
    }

    // AnimGraph中可访问的变量
    UPROPERTY(BlueprintReadOnly, Category = "Movement")
    float Speed;

    UPROPERTY(BlueprintReadOnly, Category = "Movement")
    float Direction;

    UPROPERTY(BlueprintReadOnly, Category = "Movement")
    bool bIsInAir;

    UPROPERTY(BlueprintReadOnly, Category = "Movement")
    bool bIsCrouching;

    UPROPERTY(BlueprintReadOnly, Category = "Aim")
    float PitchAngle;

private:
    UPROPERTY()
    TObjectPtr<AMyCharacter> OwningCharacter;
};
```

## Blend Space（混合空间）

Blend Space根据一个或两个参数混合多个动画：

```
// 1D Blend Space: Speed参数
// 0: Idle
// 150: Walk
// 375: Run
// 600: Sprint

// 2D Blend Space: Speed + Direction
// X轴: 速度 (0-600)
// Y轴: 方向 (-180 到 180)
// 放置8方向移动动画 + Idle
```

```cpp
// 运行时设置混合参数
void UMyAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
    // Blend Space由AnimGraph中的节点自动使用这些变量
    Speed = OwningCharacter->GetVelocity().Size();
    Direction = CalculateDirection(Velocity, Rotation);
    // AnimGraph中 [Speed] -> [LocomotionBlendSpace] -> [Output Pose]
}
```

## Montage（蒙太奇）

Montage用于播放一次性动画（攻击、技能、交互），可分段和设置通知：

```cpp
// 播放Montage
UCLASS()
class AMyCharacter : public ACharacter
{
    GENERATED_BODY()

    UFUNCTION(BlueprintCallable)
    void PlayAttackMontage()
    {
        if (AttackMontage && !GetMesh()->GetAnimInstance()->Montage_IsPlaying(AttackMontage))
        {
            // 播放Montage
            UAnimInstance* AnimInst = GetMesh()->GetAnimInstance();
            AnimInst->Montage_Play(AttackMontage);

            // 随机选择一个Section（不同的攻击动作）
            int32 SectionIndex = FMath::RandRange(0, 2);
            FName SectionName = *FString::Printf(TEXT("Attack_%d"), SectionIndex);
            AnimInst->Montage_JumpToSection(SectionName, AttackMontage);
        }
    }

    // Montage结束回调
    void OnMontageEnded(UAnimMontage* Montage, bool bInterrupted)
    {
        if (Montage == AttackMontage)
        {
            bIsAttacking = false;
        }
    }

    UPROPERTY(EditDefaultsOnly)
    TObjectPtr<UAnimMontage> AttackMontage;
};

// AnimNotify (动画通知)
// 在Montage的Timeline上添加通知点
// Notify: 命中判定、播放音效、生成粒子
// NotifyState: 持续效果（格挡、无敌帧）

// C++自定义Notify
UCLASS()
class UAnimNotify_AttackHit : public UAnimNotify
{
    GENERATED_BODY()

    virtual void Notify(USkeletalMeshComponent* MeshComp,
                        UAnimSequenceBase* Animation) override
    {
        AMyCharacter* Character = Cast<AMyCharacter>(MeshComp->GetOwner());
        if (Character)
        {
            Character->PerformAttackTrace(); // 执行攻击检测
        }
    }
};

// NotifyState (持续帧通知)
UCLASS()
class UAnimNotifyState_Invincible : public UAnimNotifyState
{
    GENERATED_BODY()

    virtual void NotifyBegin(USkeletalMeshComponent* MeshComp,
                             UAnimSequenceBase* Animation, float TotalDuration) override
    {
        // 开始无敌
    }

    virtual void NotifyEnd(USkeletalMeshComponent* MeshComp,
                           UAnimSequenceBase* Animation) override
    {
        // 结束无敌
    }
};
```

## IK（逆运动学）

```cpp
// FABRIK / Two Bone IK 用于手脚贴合目标
// 在AnimGraph中添加IK节点

// C++设置IK目标
void UMyAnimInstance::UpdateIK(float DeltaSeconds)
{
    // 脚部IK - 让脚适应地面高度
    FVector LeftFootLocation = GetOwningComponent()->GetSocketLocation(TEXT("foot_l"));
    FVector RightFootLocation = GetOwningComponent()->GetSocketLocation(TEXT("foot_r"));

    FHitResult LeftHit, RightHit;
    FVector Offset(0, 0, -90.f); // 骨骼偏移

    // 射线检测地面
    if (LineTraceForIK(LeftFootLocation + FVector::UpVector * 50.f, LeftFootLocation + FVector::DownVector * 50.f, LeftHit))
    {
        LeftFootIKOffset = LeftHit.ImpactPoint.Z - (LeftFootLocation.Z + Offset.Z);
    }

    // 在AnimGraph中将偏移应用到骨骼
}
```

## 常见陷阱与最佳实践

1. **AnimBP线程安全**: NativeUpdateAnimation在游戏线程，Thread Safe Update在动画线程
2. **Montage插槽**: 正确设置Montage的Slot，多个动画可共用Slot实现自动打断
3. **Blend Space采样点**: 均匀分布采样点避免混合突变
4. **IK性能**: 复杂IK链开销大，LOD远处禁用IK
5. **AnimBP复用**: 使用Linked Anim Graph在不同角色间复用动画逻辑

## 与其他系统的关联

- **物理系统**: Ragdoll动画需要Physics Asset配置
- **AI系统**: AI角色使用Animation Blueprint驱动行为表现
- **网络同步**: 动画状态通过Replication同步到其他客户端
