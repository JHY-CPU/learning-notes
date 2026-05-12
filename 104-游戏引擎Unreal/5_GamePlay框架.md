# GamePlay框架

## 核心概念

Unreal的GamePlay框架定义了游戏中各核心对象的角色和关系。理解框架类的职责和生命周期是构建游戏逻辑的基础。框架类协同工作，管理玩家连接、游戏状态和规则。

## 框架类关系

```
GameInstance (进程级别，跨关卡持久)
    ↓
GameMode (关卡规则，服务器端)
    ├── GameState (游戏状态，同步到客户端)
    ├── PlayerController (玩家输入控制)
    │   └── Pawn/Character (玩家操控的角色)
    └── AIController (AI控制)
        └── Pawn/Character (AI角色)
```

## GameInstance

GameInstance是进程级别的单例，跨关卡持久存在：

```cpp
UCLASS()
class YOURGAME_API UMyGameInstance : public UGameInstance
{
    GENERATED_BODY()

public:
    virtual void Init() override
    {
        Super::Init();
        // 全局初始化（网络、子系统等）
        UE_LOG(LogTemp, Log, TEXT("GameInstance Initialized"));
    }

    // 跨关卡持久数据
    UPROPERTY(BlueprintReadWrite)
    FString PlayerName;

    UPROPERTY(BlueprintReadWrite)
    int32 TotalScore;

    // 存档加载
    UPROPERTY()
    TObjectPtr<USaveGame> CurrentSave;

    // 全局切换关卡
    UFUNCTION(BlueprintCallable)
    void TravelToLevel(FString LevelName)
    {
        UGameplayStatics::OpenLevel(this, FName(*LevelName));
    }
};
```

## GameMode

GameMode定义游戏规则，仅存在于服务器端：

```cpp
UCLASS()
class YOURGAME_API AMyGameMode : public AGameModeBase
{
    GENERATED_BODY()

public:
    AMyGameMode()
    {
        // 指定默认框架类
        DefaultPawnClass = AMyCharacter::StaticClass();
        PlayerControllerClass = AMyPlayerController::StaticClass();
        GameStateClass = AMyGameState::StaticClass();
        PlayerStateClass = AMyPlayerState::StaticClass();
    }

    // 玩家登录时调用
    virtual void PostLogin(APlayerController* NewPlayer) override
    {
        Super::PostLogin(NewPlayer);
        UE_LOG(LogTemp, Log, TEXT("Player joined: %s"), *NewPlayer->GetName());
    }

    // 玩家退出时调用
    virtual void Logout(AController* Exiting) override
    {
        Super::Logout(Exiting);
    }

    // 生成默认Pawn
    virtual APawn* SpawnDefaultPawnFor_Implementation(AController* NewPlayer, AActor* StartSpot) override
    {
        return Super::SpawnDefaultPawnFor_Implementation(NewPlayer, StartSpot);
    }

    // 选择玩家出生点
    virtual AActor* ChoosePlayerStart_Implementation(AController* Player) override
    {
        return Super::ChoosePlayerStart_Implementation(Player);
    }
};
```

## GameState

GameState存储同步到所有客户端的游戏状态：

```cpp
UCLASS()
class AMyGameState : public AGameStateBase
{
    GENERATED_BODY()

public:
    // 同步到所有客户端
    UPROPERTY(Replicated, BlueprintReadOnly)
    int32 RemainingTime;

    UPROPERTY(Replicated, BlueprintReadOnly)
    int32 TeamAScore;

    UPROPERTY(Replicated, BlueprintReadOnly)
    int32 TeamBScore;

    // 需要实现GetLifetimeReplicatedProps
    virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override
    {
        Super::GetLifetimeReplicatedProps(OutLifetimeProps);
        DOREPLIFETIME(AMyGameState, RemainingTime);
        DOREPLIFETIME(AMyGameState, TeamAScore);
        DOREPLIFETIME(AMyGameState, TeamBScore);
    }
};
```

## PlayerController

PlayerController处理玩家输入，是客户端与服务器之间的桥梁：

```cpp
UCLASS()
class AMyPlayerController : public APlayerController
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override
    {
        Super::BeginPlay();

        // 显示鼠标光标
        bShowMouseCursor = true;
        bEnableClickEvents = true;

        // 设置输入模式
        FInputModeGameAndUI InputMode;
        SetInputMode(InputMode);
    }

    // 处理输入
    virtual void SetupInputComponent() override
    {
        Super::SetupInputComponent();

        // 绑定动作映射
        InputComponent->BindAction("Jump", IE_Pressed, this, &AMyPlayerController::OnJumpPressed);
        InputComponent->BindAxis("MoveForward", this, &AMyPlayerController::MoveForward);
    }

    // 客户端到服务器的RPC
    UFUNCTION(Server, Reliable)
    void ServerRequestFire(FVector Direction);

    // 获取HUD
    UFUNCTION(BlueprintPure)
    AMyHUD* GetMyHUD() const { return Cast<AMyHUD>(GetHUD()); }
};
```

## Pawn与Character

Pawn是可被Controller控制的角色实体，Character是带Movement组件的Pawn：

```cpp
UCLASS()
class AMyCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    AMyCharacter()
    {
        // Capsule碰撞体（Character自带）
        GetCapsuleComponent()->InitCapsuleSize(42.f, 96.f);

        // Movement组件（Character自带）
        GetCharacterMovement()->bOrientRotationToMovement = true;
        GetCharacterMovement()->JumpZVelocity = 500.f;

        // 骨骼网格体
        GetMesh()->SetRelativeLocation(FVector(0, 0, -90.f));

        // 相机
        SpringArm = CreateDefaultSubobject<USpringArmComponent>(TEXT("SpringArm"));
        SpringArm->SetupAttachment(RootComponent);
        SpringArm->TargetArmLength = 300.f;

        Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
        Camera->SetupAttachment(SpringArm);
    }

private:
    UPROPERTY(VisibleAnywhere)
    TObjectPtr<USpringArmComponent> SpringArm;

    UPROPERTY(VisibleAnywhere)
    TObjectPtr<UCameraComponent> Camera;
};
```

## 常见陷阱与最佳实践

1. **GameMode仅服务器存在**: 客户端不能直接访问GameMode，用GameState或PlayerState同步
2. **PlayerController一对一**: 每个玩家有且只有一个PlayerController
3. **PlayerState同步**: 玩家的名称、分数等通过PlayerState同步到所有客户端
4. **GameInstance时机**: Init()中不要访问依赖世界(World)的对象
5. **框架类配置**: 在World Settings中可覆盖默认GameMode

## 与其他系统的关联

- **网络同步**: 框架类是多人游戏同步的基础结构
- **UI系统**: PlayerController是UMG的输入处理入口
- **AI系统**: AIController是行为树的驱动者
