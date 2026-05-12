# AI系统

## 核心概念

Unreal AI系统基于AIController、Behavior Tree（行为树）、Blackboard（黑板）和EQS（环境查询系统）构建。AIController替代PlayerController控制NPC，行为树驱动决策逻辑。

## AIController

AIController是NPC的"大脑"，控制Pawn的AI行为：

```cpp
UCLASS()
class AMyAIController : public AAIController
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override
    {
        Super::BeginPlay();

        // 运行行为树
        if (BehaviorTreeAsset)
        {
            RunBehaviorTree(BehaviorTreeAsset);
        }
    }

    // 感知系统
    virtual void OnPossess(APawn* InPawn) override
    {
        Super::OnPossess(InPawn);

        // 配置AI感知
        if (AIPerception)
        {
            // 视觉感知
            UAISenseConfig_Sight* SightConfig = NewObject<UAISenseConfig_Sight>();
            SightConfig->SightRadius = 1500.f;
            SightConfig->LoseSightRadius = 2000.f;
            SightConfig->PeripheralVisionAngleDegrees = 60.f;
            SightConfig->DetectionByAffiliation.bDetectEnemies = true;
            AIPerception->ConfigureSense(*SightConfig);
        }
    }

    // 感知更新回调
    UFUNCTION()
    void OnPerceptionUpdated(AActor* Actor, FAIStimulus Stimulus)
    {
        if (Stimulus.WasSuccessfullySensed())
        {
            // 发现目标
            GetBlackboardComponent()->SetValueAsObject(TEXT("TargetActor"), Actor);
        }
    }

    UPROPERTY(EditDefaultsOnly)
    TObjectPtr<UBehaviorTree> BehaviorTreeAsset;

    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UAIPerceptionComponent> AIPerception;
};
```

## Blackboard（黑板）

黑板是AI的记忆存储，行为树通过读写黑板数据驱动决策：

```cpp
// 黑板键（Key）类型
// Object - 对象引用（TargetActor）
// Vector - 位置（TargetLocation）
// Float - 数值（Health）
// Bool - 标志（bCanSeePlayer）
// Enum - 枚举（AIState）
// Int - 整数（PatrolIndex）

// C++中操作黑板
void AMyAIController::UpdateBlackboard()
{
    UBlackboardComponent* BB = GetBlackboardComponent();
    if (BB)
    {
        // 写入
        BB->SetValueAsVector(TEXT("TargetLocation"), TargetPos);
        BB->SetValueAsObject(TEXT("TargetActor"), TargetPawn);
        BB->SetValueAsBool(TEXT("bHasWeapon"), true);
        BB->SetValueAsFloat(TEXT("Health"), CurrentHealth);

        // 读取
        FVector Location = BB->GetValueAsVector(TEXT("TargetLocation"));
        bool bCanSee = BB->GetValueAsBool(TEXT("bCanSeePlayer"));
    }
}
```

## Behavior Tree（行为树）

行为树由节点组成，从左到右、从上到下执行：

```
Root
└── Selector (选择器: 执行第一个成功的子节点)
    ├── Sequence (序列: 顺序执行所有子节点)
    │   ├── [Service] UpdateTarget
    │   ├── [Decorator] HasTarget?
    │   ├── MoveTo(Target)
    │   └── Attack(Target)
    ├── Sequence
    │   ├── [Decorator] HealthLow?
    │   └── FindCover
    └── Sequence
        ├── PatrolToNextPoint
        └── Wait(3s)
```

### 节点类型

| 节点 | 功能 | 说明 |
|------|------|------|
| Selector | 选择节点 | 执行第一个成功子节点（OR逻辑） |
| Sequence | 序列节点 | 顺序执行所有子节点（AND逻辑） |
| Service | 服务节点 | 挂在节点上，定期更新黑板 |
| Decorator | 装饰器/条件 | 条件判断，控制分支执行 |
| Task | 任务节点 | 执行具体行为 |

```cpp
// 自定义BT Task
UCLASS()
class UBTTask_FindPatrolPoint : public UBTTaskNode
{
    GENERATED_BODY()

    virtual EBTNodeResult::Type ExecuteTask(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override
    {
        AMyAIController* AICon = Cast<AMyAIController>(OwnerComp.GetAIOwner());
        if (!AICon) return EBTNodeResult::Failed;

        // 获取巡逻点
        int32 Index = OwnerComp.GetBlackboardComponent()->GetValueAsInt(TEXT("PatrolIndex"));
        FVector PatrolPoint = GetPatrolPoint(Index);

        // 写入黑板
        OwnerComp.GetBlackboardComponent()->SetValueAsVector(TEXT("MoveToLocation"), PatrolPoint);

        // 更新索引
        OwnerComp.GetBlackboardComponent()->SetValueAsInt(TEXT("PatrolIndex"), (Index + 1) % MaxPoints);

        return EBTNodeResult::Succeeded;
    }
};

// 自定义BT Decorator (条件检查)
UCLASS()
class UBTDecorator_CanSeeTarget : public UBTDecorator
{
    GENERATED_BODY()

    virtual bool CalculateRawConditionValue(UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override
    {
        AMyAIController* AICon = Cast<AMyAIController>(OwnerComp.GetAIOwner());
        AActor* Target = Cast<AActor>(OwnerComp.GetBlackboardComponent()->GetValueAsObject(TEXT("TargetActor")));

        if (!AICon || !Target) return false;

        // 简单可见性检查
        FVector AIlocation = AICon->GetPawn()->GetActorLocation();
        FVector TargetLocation = Target->GetActorLocation();

        FHitResult Hit;
        FCollisionQueryParams Params;
        Params.AddIgnoredActor(AICon->GetPawn());

        bool bBlocked = OwnerComp.GetWorld()->LineTraceSingleByChannel(
            Hit, AIlocation, TargetLocation, ECC_Visibility, Params);

        return !bBlocked; // 没有阻挡则可见
    }
};
```

## EQS（环境查询系统）

EQS用于AI在环境中寻找最优位置（掩体、攻击点、资源点）：

```cpp
// EQS查询由Generator、Test和Context组成
// Generator - 生成候选点（网格、圆形、路径）
// Test - 对每个点评分（距离、可见性、密度）
// Context - 查询的参考点（Querier自身、目标等）

// C++自定义EQS Generator
UCLASS()
class UEnvQueryGenerator_SurroundingPoints : public UEnvQueryGenerator
{
    GENERATED_BODY()

    virtual void GenerateItems(FEnvQueryInstance& QueryInstance) const override
    {
        FVector Origin = FVector::ZeroVector;
        QueryInstance.PrepareContext(OriginContext, Origin);

        // 在Origin周围生成候选点
        for (int32 i = 0; i < NumPoints; i++)
        {
            float Angle = (360.f / NumPoints) * i;
            FVector Point = Origin + FRotator(0, Angle, 0).Vector() * Radius;
            QueryInstance.AddItem<FEnvQueryItem>(Point);
        }
    }
};

// 在代码中执行EQS查询
void AMyAIController::RunEnvQuery()
{
    if (EnvQuery)
    {
        FEnvQueryRequest QueryRequest(EnvQuery, GetPawn());
        QueryRequest.Execute(
            EEnvQueryRunMode::SingleResult,
            FQueryFinishedSignature::CreateUObject(this, &AMyAIController::OnQueryFinished)
        );
    }
}

void AMyAIController::OnQueryFinished(TSharedPtr<FEnvQueryResult> Result)
{
    if (Result->IsSuccessful())
    {
        FVector BestLocation = Result->GetItemAsLocation(0);
        GetBlackboardComponent()->SetValueAsVector(TEXT("CoverLocation"), BestLocation);
    }
}
```

## 常见陷阱与最佳实践

1. **行为树不要过深**: 层级过深导致调试困难，建议不超过4层
2. **Service频率**: Service的Tick间隔影响性能，非关键数据降低频率
3. **黑板键命名统一**: 团队内统一命名规范（b前缀布尔值，Target前缀目标等）
4. **EQS性能**: 查询点数量和Test复杂度直接影响帧率
5. **多人AI**: 注意AIController在服务器端运行，客户端只看到表现

## 与其他系统的关联

- **导航系统**: MoveTo依赖NavMesh寻路
- **动画系统**: AI行为驱动动画状态机
- **感知系统**: AIPerception提供视觉/听觉输入
