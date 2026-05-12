# 状态模式（State）

## 核心概念

状态模式允许对象在内部状态改变时改变其行为，使对象看起来好像改变了其类。在游戏开发中，几乎所有涉及"不同阶段有不同行为"的系统都适用状态模式：角色行为、动画系统、UI流程、游戏阶段管理等。

### 为什么需要状态模式？原始方式的问题

```csharp
// 错误方式：用 if-else / switch 管理状态
class Player
{
    enum State { Idle, Running, Jumping, Attacking, Dead }
    State currentState;

    void Update(float dt)
    {
        switch (currentState)
        {
            case State.Idle:
                if (input.MovePressed) { currentState = State.Running; }
                if (input.JumpPressed) { currentState = State.Jumping; }
                // ...50行类似代码
                break;
            case State.Running:
                // ...又是50行
                break;
            // 10个状态 × 50行 = 500行 if-else 地狱
        }
    }
}
```

问题：
1. 每增加一个新状态，需要在每个 case 中添加转换条件
2. 状态进入/退出逻辑（如播放动画）和状态更新混在一起
3. 添加"任意状态都可切换到受击状态"需要在每个 case 中重复代码
4. 代码无法复用和测试

### 状态接口与具体状态

```csharp
// 状态接口定义
public interface ICharacterState
{
    void Enter(Character character);   // 进入状态时调用一次
    void Update(Character character, float dt); // 每帧调用
    void Exit(Character character);    // 退出状态时调用一次
    void OnTriggerEnter(Character character, Collider other); // 碰撞回调
    string GetStateName(); // 调试用
}

// ===== 具体状态实现 =====

public class IdleState : ICharacterState
{
    public void Enter(Character c)
    {
        c.Animator.CrossFade("Idle", 0.1f);
        c.Velocity = new Vector3(0, c.Velocity.y, 0); // 停止水平移动
    }

    public void Update(Character c, float dt)
    {
        // 转换条件判断——每个状态自己决定何时转换
        if (c.Health <= 0)
        {
            c.StateMachine.TransitionTo(new DeathState());
            return;
        }
        if (c.Input.MovePressed)
        {
            c.StateMachine.TransitionTo(new RunState());
            return;
        }
        if (c.Input.JumpPressed && c.IsGrounded)
        {
            c.StateMachine.TransitionTo(new JumpState());
            return;
        }
        if (c.Input.AttackPressed)
        {
            c.StateMachine.TransitionTo(new AttackState());
            return;
        }
    }

    public void Exit(Character c) { }
    public void OnTriggerEnter(Character c, Collider other) { }
    public string GetStateName() => "Idle";
}

public class RunState : ICharacterState
{
    public void Enter(Character c)
    {
        c.Animator.CrossFade("Run", 0.1f);
    }

    public void Update(Character c, float dt)
    {
        if (c.Health <= 0) { c.StateMachine.TransitionTo(new DeathState()); return; }
        if (!c.Input.MovePressed) { c.StateMachine.TransitionTo(new IdleState()); return; }
        if (c.Input.JumpPressed && c.IsGrounded) { c.StateMachine.TransitionTo(new JumpState()); return; }
        if (c.Input.AttackPressed) { c.StateMachine.TransitionTo(new AttackState()); return; }

        // 状态内的行为：移动
        Vector3 moveDir = c.Input.Direction.normalized;
        c.Velocity = new Vector3(moveDir.x * c.MoveSpeed, c.Velocity.y, moveDir.z * c.MoveSpeed);
        c.Transform.Position += c.Velocity * dt;

        // 朝向移动方向
        if (moveDir.magnitude > 0.1f)
            c.Transform.Rotation = Quaternion.LookRotation(moveDir);
    }

    public void Exit(Character c) { }
    public void OnTriggerEnter(Character c, Collider other) { }
    public string GetStateName() => "Run";
}

public class JumpState : ICharacterState
{
    private bool hasLanded = false;

    public void Enter(Character c)
    {
        c.Animator.CrossFade("Jump", 0.05f);
        c.Velocity = new Vector3(c.Velocity.x, c.JumpForce, c.Velocity.z);
        hasLanded = false;
    }

    public void Update(Character c, float dt)
    {
        if (c.Health <= 0) { c.StateMachine.TransitionTo(new DeathState()); return; }

        // 空中可以控制方向（但速度降低）
        Vector3 airControl = c.Input.Direction.normalized * c.MoveSpeed * 0.5f;
        c.Velocity = new Vector3(airControl.x, c.Velocity.y - 9.8f * dt, airControl.z);
        c.Transform.Position += c.Velocity * dt;

        // 落地检测
        if (c.Transform.Position.y <= 0)
        {
            c.Transform.Position = new Vector3(c.Transform.Position.x, 0, c.Transform.Position.z);
            if (c.Input.MovePressed)
                c.StateMachine.TransitionTo(new RunState());
            else
                c.StateMachine.TransitionTo(new IdleState());
        }
    }

    public void Exit(Character c)
    {
        c.Animator.CrossFade("Land", 0.1f);
    }

    public void OnTriggerEnter(Character c, Collider other) { }
    public string GetStateName() => "Jump";
}

public class AttackState : ICharacterState
{
    private float attackTimer;
    private float attackDuration = 0.5f;
    private bool hasHit = false;

    public void Enter(Character c)
    {
        c.Animator.CrossFade("Attack", 0.05f);
        attackTimer = 0f;
        hasHit = false;
        c.Velocity = new Vector3(0, c.Velocity.y, 0); // 攻击时不能移动
    }

    public void Update(Character c, float dt)
    {
        attackTimer += dt;

        // 在攻击窗口期内检测命中
        if (attackTimer > 0.15f && attackTimer < 0.35f && !hasHit)
        {
            hasHit = true;
            c.WeaponComponent.PerformAttack();
        }

        // 攻击动画结束
        if (attackTimer >= attackDuration)
        {
            c.StateMachine.TransitionTo(new IdleState());
        }
    }

    public void Exit(Character c) { }
    public void OnTriggerEnter(Character c, Collider other) { }
    public string GetStateName() => "Attack";
}

public class DeathState : ICharacterState
{
    public void Enter(Character c)
    {
        c.Animator.CrossFade("Death", 0.1f);
        c.Collider.enabled = false;
        c.Velocity = Vector3.Zero;
    }

    public void Update(Character c, float dt)
    {
        // 死亡状态不响应任何输入
        // 等待动画播放完后销毁或进入重生流程
    }

    public void Exit(Character c) { }
    public void OnTriggerEnter(Character c, Collider other) { }
    public string GetStateName() => "Death";
}

// 受击状态（可以从任何状态转换过来）
public class HurtState : ICharacterState
{
    private float hurtTimer;
    private float hurtDuration = 0.3f;
    private ICharacterState previousState; // 记录之前的状态

    public HurtState(ICharacterState prevState)
    {
        previousState = prevState;
    }

    public void Enter(Character c)
    {
        c.Animator.CrossFade("Hurt", 0.05f);
        hurtTimer = 0f;
        // 被击退
        c.Velocity = -c.transform.Forward * 3f;
    }

    public void Update(Character c, float dt)
    {
        hurtTimer += dt;
        if (hurtTimer >= hurtDuration)
        {
            // 回到之前的状态（或直接回到Idle）
            c.StateMachine.TransitionTo(previousState ?? new IdleState());
        }
    }

    public void Exit(Character c) { }
    public void OnTriggerEnter(Character c, Collider other) { }
    public string GetStateName() => "Hurt";
}
```

### 完整状态机框架

```csharp
public class StateMachine
{
    private ICharacterState currentState;
    private Character owner;
    private Dictionary<Type, ICharacterState> stateCache = new();
    private bool isTransitioning = false;
    private ICharacterState pendingState = null;

    // 状态变化事件（用于UI显示、日志等）
    public event Action<ICharacterState, ICharacterState> OnStateChanged;

    public ICharacterState CurrentState => currentState;
    public string CurrentStateName => currentState?.GetStateName() ?? "None";

    public StateMachine(Character owner)
    {
        this.owner = owner;
    }

    public void Initialize(ICharacterState initialState)
    {
        currentState = initialState;
        currentState.Enter(owner);
    }

    public void TransitionTo(ICharacterState newState)
    {
        // 避免重复进入同一状态
        if (currentState?.GetType() == newState.GetType()) return;

        if (isTransitioning)
        {
            // 正在转换中，暂存待处理状态
            pendingState = newState;
            return;
        }

        isTransitioning = true;

        var oldState = currentState;
        oldState?.Exit(owner);
        currentState = newState;
        currentState.Enter(owner);

        OnStateChanged?.Invoke(oldState, currentState);

        isTransitioning = false;

        // 处理转换期间被暂存的状态
        if (pendingState != null)
        {
            var pending = pendingState;
            pendingState = null;
            TransitionTo(pending);
        }
    }

    // 全局转换：从任何状态转换到某个状态（如受击、死亡）
    public void ForceTransition(ICharacterState newState)
    {
        currentState?.Exit(owner);
        var oldState = currentState;
        currentState = newState;
        currentState.Enter(owner);
        OnStateChanged?.Invoke(oldState, currentState);
    }

    public void Update(float dt)
    {
        currentState?.Update(owner, dt);
    }
}
```

### 推入式自动机（Pushdown Automaton）

普通FSM只能在同层级状态间切换。推入式自动机使用栈管理状态，支持"暂停当前状态、进入子状态、完成后恢复"：

```csharp
public class PushdownStateMachine
{
    private Stack<ICharacterState> stateStack = new();
    private Character owner;

    public ICharacterState CurrentState => stateStack.Count > 0 ? stateStack.Peek() : null;

    public PushdownStateMachine(Character owner) { this.owner = owner; }

    // 推入新状态（当前状态暂停但不退出）
    public void PushState(ICharacterState newState)
    {
        // 注意：不调用当前状态的 Exit
        stateStack.Push(newState);
        newState.Enter(owner);
    }

    // 弹出当前状态，恢复上一个状态
    public void PopState()
    {
        if (stateStack.Count == 0) return;
        var oldState = stateStack.Pop();
        oldState.Exit(owner);
        // 上一个状态自动恢复（不需要Enter，因为它从未真正Exit）
    }

    public void Update(float dt)
    {
        CurrentState?.Update(owner, dt);
    }
}

// 使用推入式状态机的示例
// 角色在"移动"状态中，突然进入对话（Push），
// 对话结束后弹出，自动恢复到"移动"状态
class GameplayState : ICharacterState
{
    public void Update(Character c, float dt)
    {
        if (Input.GetKeyDown(KeyCode.E) && c.NearNPC)
        {
            c.PushdownStateMachine.PushState(new DialogueState(c.NearNPC));
        }
        // 正常移动逻辑...
    }
}

class DialogueState : ICharacterState
{
    public void Update(Character c, float dt)
    {
        if (Input.GetKeyDown(KeyCode.Escape) || dialogueFinished)
        {
            c.PushdownStateMachine.PopState(); // 返回到之前的状态
        }
        // 对话UI逻辑...
    }
}
```

### 动画状态机

游戏引擎的动画系统本质上是复杂的FSM：

```
动画状态图：
[Idle] --MovePressed--> [Run]
[Idle] --JumpPressed--> [JumpStart]
[Run]  --!MovePressed--> [Idle]
[JumpStart] --AnimFinished--> [JumpLoop]
[JumpLoop]  --Landed--> [JumpLand]
[JumpLand]  --AnimFinished--> [Idle/Run]
[AnyState]  --HitReceived--> [Hurt] (全局转换)
[AnyState]  --HP<=0--> [Death]
```

```csharp
public class AnimationStateMachine
{
    private Dictionary<string, AnimationState> states = new();
    private AnimationState currentState;
    private Dictionary<string, float> parameters = new();

    public void AddState(string name, AnimationState state)
    {
        states[name] = state;
    }

    public void SetBool(string name, bool value) { parameters[name] = value ? 1f : 0f; CheckTransitions(); }
    public void SetFloat(string name, float value) { parameters[name] = value; CheckTransitions(); }
    public void SetTrigger(string name) { parameters[name] = 1f; CheckTransitions(); }

    private void CheckTransitions()
    {
        foreach (var transition in currentState.Transitions)
        {
            if (transition.Evaluate(parameters))
            {
                TransitionTo(transition.TargetStateName);
                break;
            }
        }
    }

    private void TransitionTo(string stateName)
    {
        if (currentState?.Name == stateName) return;
        currentState?.OnExit();
        currentState = states[stateName];
        currentState.OnEnter();
    }

    public void Update(float dt)
    {
        currentState?.Update(dt);
    }
}

public class AnimationTransition
{
    public string TargetStateName;
    public List<AnimCondition> Conditions;

    public bool Evaluate(Dictionary<string, float> parameters)
    {
        return Conditions.All(c => c.Evaluate(parameters));
    }
}

public class AnimCondition
{
    public string ParameterName;
    public CompareMode Mode;
    public float Threshold;

    public bool Evaluate(Dictionary<string, float> parameters)
    {
        if (!parameters.TryGetValue(ParameterName, out float value)) return false;
        return Mode switch
        {
            CompareMode.Greater => value > Threshold,
            CompareMode.Less => value < Threshold,
            CompareMode.Equals => Math.Abs(value - Threshold) < 0.01f,
            _ => false
        };
    }
}

public enum CompareMode { Greater, Less, Equals }
```

### 游戏流程状态机

```csharp
// 游戏整体流程管理
public class GameFlowStateMachine
{
    private StateMachine stateMachine;

    public void Initialize()
    {
        stateMachine = new StateMachine(this);
        stateMachine.Initialize(new MainMenuState());
    }
}

public class MainMenuState : ICharacterState
{
    public void Enter(Character c)
    {
        UIManager.Show("MainMenu");
        AudioManager.PlayBGM("menu_theme");
    }

    public void Update(Character c, float dt)
    {
        if (UIManager.GetButton("PlayButton").Clicked)
        {
            c.StateMachine.TransitionTo(new LoadingState("Level1"));
        }
        if (UIManager.GetButton("SettingsButton").Clicked)
        {
            c.StateMachine.TransitionTo(new SettingsState());
        }
    }

    public void Exit(Character c)
    {
        UIManager.Hide("MainMenu");
    }
}

public class LoadingState : ICharacterState
{
    private string levelName;
    private float loadProgress;

    public LoadingState(string level) { levelName = level; }

    public void Enter(Character c)
    {
        UIManager.Show("LoadingScreen");
        SceneManager.LoadSceneAsync(levelName, OnProgress, OnComplete);
    }

    public void Update(Character c, float dt)
    {
        UIManager.UpdateProgressBar(loadProgress);
    }

    void OnProgress(float progress) { loadProgress = progress; }
    void OnComplete() { /* 转换到 GameplayState */ }

    public void Exit(Character c)
    {
        UIManager.Hide("LoadingScreen");
    }
}

public class GameplayState : ICharacterState
{
    public void Enter(Character c)
    {
        UIManager.Show("InGameHUD");
        Time.timeScale = 1f;
        AudioManager.PlayBGM("level_theme");
    }

    public void Update(Character c, float dt)
    {
        if (Input.GetKeyDown(KeyCode.Escape))
            c.StateMachine.TransitionTo(new PauseState());
    }

    public void Exit(Character c)
    {
        UIManager.Hide("InGameHUD");
    }
}

public class PauseState : ICharacterState
{
    public void Enter(Character c)
    {
        UIManager.Show("PauseMenu");
        Time.timeScale = 0f;
    }

    public void Update(Character c, float dt)
    {
        if (Input.GetKeyDown(KeyCode.Escape))
            c.StateMachine.TransitionTo(new GameplayState());
        if (UIManager.GetButton("QuitButton").Clicked)
            c.StateMachine.TransitionTo(new MainMenuState());
    }

    public void Exit(Character c)
    {
        UIManager.Hide("PauseMenu");
        Time.timeScale = 1f;
    }
}
```

## 方案对比

| 方案 | 代码复杂度 | 状态数量上限 | 全局转换 | 嵌套状态 | 调试可视化 |
|------|-----------|-------------|---------|---------|-----------|
| if-else / switch | 低 | ~5 | 困难 | 不支持 | 无 |
| 状态类 + FSM | 中 | ~50 | 容易 | 不支持 | 可打印 |
| 推入式自动机 | 中 | ~30 | 容易 | 原生支持 | 可打印栈 |
| 行为树 | 高 | 无限 | N/A | 原生支持 | 可视化编辑器 |

## 常见陷阱与解决方案

1. **状态数量爆炸**：10个状态之间的转换连接可能有90条。解决方案：使用"全局转换"减少重复条件
2. **状态转换时序**：在 Enter 中触发另一个状态转换导致栈溢出。解决方案：延迟转换（pendingState）
3. **状态间数据传递**：攻击状态需要知道目标信息。解决方案：状态构造函数传参或共享上下文对象
4. **调试困难**：状态切换频繁难以追踪。解决方案：在 TransitionTo 中添加日志输出
5. **动画和状态不同步**：状态切换了但动画还在播放。解决方案：状态 Exit 中强制停止动画

## Unity 实现

```csharp
// Unity Animator Controller 就是可视化的状态机
// 也可以用纯代码实现
public class PlayerStateMachine : MonoBehaviour
{
    private ICharacterState currentState;

    void Start()
    {
        TransitionTo(new IdleState());
    }

    void Update()
    {
        currentState?.Update(this, Time.deltaTime);
    }

    public void TransitionTo(ICharacterState newState)
    {
        currentState?.Exit(this);
        currentState = newState;
        currentState?.Enter(this);
        Debug.Log($"State: {currentState.GetStateName()}");
    }
}
```

## Unreal Engine 实现

```cpp
// UE 的 Gameplay Ability System (GAS) 使用状态管理能力
// UE 的 AI 行为树也包含状态管理的特性

// 简单的 C++ 状态机
UCLASS()
class UMyStateMachine : public UObject
{
    GENERATED_BODY()
public:
    UPROPERTY()
    UMyStateBase* CurrentState;

    void TransitionState(UMyStateBase* NewState)
    {
        if (CurrentState)
            CurrentState->ExitState();
        CurrentState = NewState;
        CurrentState->EnterState();
    }

    void Tick(float DeltaTime)
    {
        if (CurrentState)
            CurrentState->TickState(DeltaTime);
    }
};
```

## 实际使用案例

- **《鬼泣》系列** 的角色状态系统管理数十种攻击、移动、特殊状态
- **Unity 的 Animator Controller** 是可视化的动画状态机编辑器
- **《文明》系列** 的游戏阶段（回合开始→玩家行动→AI行动→回合结束）使用状态模式
- **《空洞骑士》** 的角色状态精细到区分墙壁滑行、冲刺后硬直等细节状态
- **《只狼》** 的架势系统、弹刀窗口等精确到帧的状态管理
