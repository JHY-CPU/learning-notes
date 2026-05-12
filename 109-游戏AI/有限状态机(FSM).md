# 有限状态机（FSM）

## 1. 核心理论

### 1.1 什么是有限状态机

有限状态机（Finite State Machine，FSM）是一种计算模型，由有限数量的**状态**、**转换条件**和**事件**组成。在游戏AI中，FSM将实体的行为分解为离散的行为模式（状态），通过预定义规则在状态之间切换。

形式化定义：FSM是一个五元组 M = (S, S0, Σ, δ, F)
- **S**：有限状态集合
- **S0**：初始状态
- **Σ**：输入事件集合
- **δ**：状态转换函数 δ: S x Σ → S
- **F**：终止状态集合（游戏中通常没有终止状态，FSM持续运行）

### 1.2 状态机的设计原则

1. **原子性**：每个状态只负责单一、明确的行为
2. **完备性**：每个状态下对所有可能事件都有定义的转换或忽略
3. **确定性**：同一状态+同一事件产生唯一的下一个状态
4. **无环性**（理想情况）：避免状态间的瞬时循环转换

### 1.3 状态机的数学性质

FSM的一个重要性质是**可达性**：状态Si可达Sj，当且仅当存在一条从Si到Sj的转换路径。可达性分析可以发现：
- 孤立状态（无法到达或无法离开的状态）
- 死锁状态（进入后无法到达期望目标的状态）
- 活锁状态（在几个状态间无限循环但无法前进）

## 2. 基础FSM框架实现

### 2.1 状态基类

```cpp
#include <string>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <typeindex>
#include <memory>

// 前向声明
class Entity;

// 状态基类 - 使用CRTP模式避免虚函数开销
class IState {
public:
    virtual ~IState() = default;
    virtual void Enter(Entity* entity) = 0;
    virtual void Update(Entity* entity, float deltaTime) = 0;
    virtual void Exit(Entity* entity) = 0;
    virtual std::string GetName() const = 0;

    // 可选：处理外部事件
    virtual bool HandleEvent(Entity* entity, const std::string& eventName,
                              void* eventData = nullptr) { return false; }
};
```

### 2.2 实体类

```cpp
class Entity {
public:
    std::string name;
    float health = 100.0f;
    float maxHealth = 100.0f;
    float attackDamage = 10.0f;
    float walkSpeed = 3.0f;
    float runSpeed = 6.0f;
    float detectionRange = 15.0f;
    float attackRange = 2.0f;

    // 位置和朝向
    struct Vec3 { float x, y, z; };
    Vec3 position = {0, 0, 0};
    Vec3 forward = {0, 0, 1};

    Entity* targetEnemy = nullptr;
    Vec3 waypoint = {0, 0, 0};
    float healthRegenRate = 2.0f; // 每秒回血

    // FSM控制器
    class FSM* fsm = nullptr;

    bool IsAlive() const { return health > 0; }
    float GetHealthPercent() const { return health / maxHealth; }
    float DistanceTo(Entity* other) const {
        float dx = position.x - other->position.x;
        float dz = position.z - other->position.z;
        return sqrtf(dx*dx + dz*dz);
    }
    bool IsInDetectionRange(Entity* other) const { return DistanceTo(other) <= detectionRange; }
    bool IsInAttackRange(Entity* other) const { return DistanceTo(other) <= attackRange; }

    void TakeDamage(float dmg) { health = std::max(0.0f, health - dmg); }
    void Heal(float amount) { health = std::min(maxHealth, health + amount); }
    void MoveToward(const Vec3& target, float speed, float dt);
};
```

### 2.3 FSM控制器

```cpp
class FSM {
    Entity* owner;
    IState* currentState = nullptr;
    IState* previousState = nullptr;
    IState* globalState = nullptr;  // 全局状态（处理死亡等全局事件）

public:
    FSM(Entity* entity) : owner(entity) {}

    void SetState(IState* state) {
        if (currentState) currentState->Exit(owner);
        previousState = currentState;
        currentState = state;
        currentState->Enter(owner);
    }

    void SetGlobalState(IState* state) { globalState = state; }

    void Update(float dt) {
        // 全局状态先执行
        if (globalState) globalState->Update(owner, dt);
        // 当前状态执行
        if (currentState) currentState->Update(owner, dt);
    }

    void TransitionTo(IState* newState) {
        if (!newState || newState == currentState) return;
        if (currentState) currentState->Exit(owner);
        previousState = currentState;
        currentState = newState;
        currentState->Enter(owner);
    }

    // 回退到上一个状态
    void RevertToPreviousState() {
        if (previousState) TransitionTo(previousState);
    }

    IState* GetCurrentState() const { return currentState; }
    IState* GetPreviousState() const { return previousState; }

    bool IsInState(const std::string& stateName) const {
        return currentState && currentState->GetName() == stateName;
    }
};
```

## 3. 完整的敌人AI状态实现

### 3.1 巡逻状态

```cpp
class PatrolState : public IState {
public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 开始巡逻\n";
        e->waypoint = {rand() % 100, 0, rand() % 100}; // 随机巡逻点
    }

    void Update(Entity* e, float dt) override {
        // 移动到巡逻点
        e->MoveToward(e->waypoint, e->walkSpeed, dt);

        // 检测敌人
        if (e->targetEnemy && e->IsInDetectionRange(e->targetEnemy)) {
            e->fsm->TransitionTo(new ChaseState());
            return;
        }

        // 到达巡逻点，选择下一个
        // (简化判断)
        static float patrolTimer = 0;
        patrolTimer += dt;
        if (patrolTimer > 5.0f) {
            patrolTimer = 0;
            e->waypoint = {rand() % 100, 0, rand() % 100};
        }
    }

    void Exit(Entity* e) override {
        std::cout << e->name << " 停止巡逻\n";
    }

    std::string GetName() const override { return "Patrol"; }
};
```

### 3.2 追击状态

```cpp
class ChaseState : public IState {
    float chaseTimer = 0;

public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 发现敌人，开始追击!\n";
        chaseTimer = 0;
    }

    void Update(Entity* e, float dt) override {
        chaseTimer += dt;

        if (!e->targetEnemy || !e->targetEnemy->IsAlive()) {
            e->fsm->TransitionTo(new PatrolState());
            return;
        }

        // 追击目标
        e->MoveToward(e->targetEnemy->position, e->runSpeed, dt);

        // 进入攻击范围
        if (e->IsInAttackRange(e->targetEnemy)) {
            e->fsm->TransitionTo(new AttackState());
            return;
        }

        // 目标超出视野，失去目标
        if (!e->IsInDetectionRange(e->targetEnemy) || chaseTimer > 10.0f) {
            e->fsm->TransitionTo(new SearchState());
            return;
        }
    }

    void Exit(Entity* e) override {
        std::cout << e->name << " 停止追击\n";
    }

    std::string GetName() const override { return "Chase"; }
};
```

### 3.3 攻击状态

```cpp
class AttackState : public IState {
    float attackCooldown = 0;
    float attackRate = 1.0f; // 每秒攻击1次

public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 进入攻击状态!\n";
        attackCooldown = 0;
    }

    void Update(Entity* e, float dt) override {
        attackCooldown += dt;

        if (!e->targetEnemy || !e->targetEnemy->IsAlive()) {
            e->fsm->TransitionTo(new PatrolState());
            return;
        }

        // 血量过低，撤退
        if (e->GetHealthPercent() < 0.2f) {
            e->fsm->TransitionTo(new FleeState());
            return;
        }

        // 目标离开攻击范围
        if (!e->IsInAttackRange(e->targetEnemy)) {
            e->fsm->TransitionTo(new ChaseState());
            return;
        }

        // 攻击
        if (attackCooldown >= attackRate) {
            attackCooldown = 0;
            e->targetEnemy->TakeDamage(e->attackDamage);
            std::cout << e->name << " 攻击了 " << e->targetEnemy->name << "\n";
        }
    }

    void Exit(Entity* e) override {
        std::cout << e->name << " 停止攻击\n";
    }

    std::string GetName() const override { return "Attack"; }
};
```

### 3.4 撤退状态

```cpp
class FleeState : public IState {
public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 血量过低，开始撤退!\n";
    }

    void Update(Entity* e, float dt) override {
        // 远离敌人
        if (e->targetEnemy) {
            // 计算远离方向
            float dx = e->position.x - e->targetEnemy->position.x;
            float dz = e->position.z - e->targetEnemy->position.z;
            float len = sqrtf(dx*dx + dz*dz);
            if (len > 0) {
                // 向远离方向移动
                Entity::Vec3 fleeTarget = {
                    e->position.x + (dx/len) * 20.0f,
                    0,
                    e->position.z + (dz/len) * 20.0f
                };
                e->MoveToward(fleeTarget, e->runSpeed, dt);
            }
        }

        // 脱离追击范围后恢复
        if (!e->targetEnemy || !e->IsInDetectionRange(e->targetEnemy)) {
            e->fsm->TransitionTo(new HealState());
        }
    }

    void Exit(Entity* e) override {
        std::cout << e->name << " 停止撤退\n";
    }

    std::string GetName() const override { return "Flee"; }
};
```

### 3.5 回血状态

```cpp
class HealState : public IState {
public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 脱战，开始回血\n";
    }

    void Update(Entity* e, float dt) override {
        e->Heal(e->healthRegenRate * dt);

        if (e->GetHealthPercent() >= 0.8f) {
            e->fsm->TransitionTo(new PatrolState());
            return;
        }

        // 战斗中被打断
        if (e->targetEnemy && e->IsInDetectionRange(e->targetEnemy)) {
            e->fsm->TransitionTo(new ChaseState());
        }
    }

    void Exit(Entity* e) override {
        std::cout << e->name << " 结束回血\n";
    }

    std::string GetName() const override { return "Heal"; }
};
```

### 3.6 搜索状态

```cpp
class SearchState : public IState {
    float searchTimer = 0;
    Entity::Vec3 lastKnownPos;

public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 失去目标，开始搜索\n";
        searchTimer = 0;
        if (e->targetEnemy) lastKnownPos = e->targetEnemy->position;
    }

    void Update(Entity* e, float dt) override {
        searchTimer += dt;
        e->MoveToward(lastKnownPos, e->walkSpeed, dt);

        // 重新发现敌人
        if (e->targetEnemy && e->IsInDetectionRange(e->targetEnemy)) {
            e->fsm->TransitionTo(new ChaseState());
            return;
        }

        // 搜索超时，回到巡逻
        if (searchTimer > 8.0f) {
            e->fsm->TransitionTo(new PatrolState());
        }
    }

    void Exit(Entity* e) override {}

    std::string GetName() const override { return "Search"; }
};
```

### 3.7 死亡全局状态

```cpp
class DeadState : public IState {
public:
    void Enter(Entity* e) override {
        std::cout << e->name << " 死亡!\n";
    }
    void Update(Entity* e, float dt) override {
        // 死亡状态不做任何事
    }
    void Exit(Entity* e) override {}

    std::string GetName() const override { return "Dead"; }
};
```

## 4. 分层状态机（Hierarchical FSM）

### 4.1 设计动机

扁平FSM的致命问题是**状态爆炸**：N个状态需要O(N^2)条转换边。分层状态机通过将状态组织为层次结构来解决这个问题。

**层次结构**：
```
[顶层：行为模式]
├── [战斗模式]
│   ├── 攻击子状态
│   ├── 追击子状态
│   └── 掩护子状态
├── [探索模式]
│   ├── 巡逻子状态
│   ├── 搜索子状态
│   └── 检查子状态
└── [休整模式]
    ├── 回血子状态
    └── 交易子状态
```

**关键规则**：
- 父状态的Enter/Exit在子状态切换时会被调用
- 子状态未处理的事件向上传递给父状态
- 父状态的转换可以覆盖子状态

### 4.2 实现

```cpp
class HierarchicalState : public IState {
protected:
    HierarchicalState* parentState = nullptr;
    IState* childState = nullptr;

public:
    void SetParent(HierarchicalState* parent) { parentState = parent; }

    void SetChildState(IState* child) {
        if (childState) childState->Exit(nullptr); // entity传递简化
        childState = child;
        if (childState) childState->Enter(nullptr);
    }

    void Update(float dt) {
        if (childState) childState->Update(nullptr, dt);
        // 父状态逻辑（如果有）
    }

    // 事件冒泡：未处理的事件向上传递
    bool HandleEvent(const std::string& event) {
        // 先让子状态处理
        if (childState && childState->HandleEvent(nullptr, event)) return true;
        // 子状态无法处理，尝试自己处理
        if (OnEvent(event)) return true;
        // 自己也无法处理，向上传递
        if (parentState) return parentState->HandleEvent(event);
        return false;
    }

    virtual bool OnEvent(const std::string& event) { return false; }
};

// 战斗模式父状态
class CombatMode : public HierarchicalState {
public:
    void Enter(Entity* e) override {
        std::cout << "进入战斗模式\n";
        // 设置子状态为追击
        SetChildState(new ChaseState());
    }

    void Update(Entity* e, float dt) override {
        if (childState) childState->Update(e, dt);
    }

    void Exit(Entity* e) override {
        std::cout << "离开战斗模式\n";
        if (childState) { childState->Exit(e); delete childState; childState = nullptr; }
    }

    // 处理全局事件
    bool OnEvent(const std::string& event) override {
        if (event == "low_health") {
            // 切换到休整模式
            return true; // 处理了
        }
        return false;
    }

    std::string GetName() const override { return "CombatMode"; }
};
```

## 5. 基于栈的FSM（Stack FSM）

### 5.1 设计思想

栈FSM使用栈结构管理状态，支持状态的**压入**和**弹出**。常用于动画系统和行为中断。

**典型场景**：
1. 角色正常行走
2. 突然受到攻击 → 压入"受击动画"状态
3. 受击动画结束 → 弹出，恢复行走
4. 玩家按下技能键 → 压入"技能动画"
5. 技能动画结束 → 弹出，恢复行走

### 5.2 实现

```cpp
class StackFSM {
    std::vector<IState*> stateStack;

public:
    void PushState(IState* state, Entity* entity) {
        if (!stateStack.empty()) {
            stateStack.back()->Exit(entity); // 暂停当前状态
        }
        stateStack.push_back(state);
        state->Enter(entity);
    }

    void PopState(Entity* entity) {
        if (stateStack.empty()) return;
        stateStack.back()->Exit(entity);
        stateStack.pop_back();
        if (!stateStack.empty()) {
            stateStack.back()->Enter(entity); // 恢复上一个状态
        }
    }

    void ReplaceState(IState* state, Entity* entity) {
        if (!stateStack.empty()) {
            stateStack.back()->Exit(entity);
            stateStack.pop_back();
        }
        stateStack.push_back(state);
        state->Enter(entity);
    }

    void Update(Entity* entity, float dt) {
        if (!stateStack.empty()) {
            stateStack.back()->Update(entity, dt);
        }
    }

    IState* GetCurrentState() const {
        return stateStack.empty() ? nullptr : stateStack.back();
    }

    int GetStackSize() const { return (int)stateStack.size(); }

    // 清空栈到指定状态
    void ClearTo(IState* target, Entity* entity) {
        while (!stateStack.empty() && stateStack.back() != target) {
            stateStack.back()->Exit(entity);
            stateStack.pop_back();
        }
    }
};
```

### 5.3 动画栈示例

```cpp
class AnimatorFSM {
    StackFSM stack;
    Entity* owner;

public:
    AnimatorFSM(Entity* e) : owner(e) {}

    void Update(float dt) {
        stack.Update(owner, dt);
    }

    // 正常移动
    void StartWalking() {
        stack.ReplaceState(new WalkAnimState(), owner);
    }

    // 被打断：压入受击
    void OnHit() {
        stack.PushState(new HitReactState(), owner);
    }

    // 使用技能：压入技能动画
    void PlaySkillAnimation(const std::string& skillName) {
        stack.PushState(new SkillAnimState(skillName), owner);
    }

    // 动画结束回调
    void OnAnimationComplete() {
        stack.PopState(owner); // 弹出，恢复上层状态
    }
};
```

## 6. 基于表驱动的FSM

### 6.1 为什么需要表驱动

当状态和转换很多时，分散在各状态类中的转换逻辑难以维护。表驱动将所有转换规则集中在一张表中。

```cpp
// 转换表驱动FSM
enum class StateID {
    Patrol, Chase, Attack, Flee, Heal, Dead
};

enum class EventID {
    EnemyDetected, EnemyLost, InAttackRange,
    LowHealth, HealthRestored, EnemyKilled, TookDamage
};

class TableDrivenFSM {
    using TransitionFn = std::function<void(Entity*)>;

    struct Transition {
        StateID from;
        EventID event;
        StateID to;
        TransitionFn action; // 转换时执行的动作
    };

    Entity* owner;
    StateID currentState;
    std::vector<Transition> transitionTable;
    std::unordered_map<StateID, IState*> stateInstances;

public:
    TableDrivenFSM(Entity* e) : owner(e), currentState(StateID::Patrol) {
        // 定义转换表
        transitionTable = {
            {StateID::Patrol,  EventID::EnemyDetected,  StateID::Chase,  nullptr},
            {StateID::Chase,   EventID::InAttackRange,  StateID::Attack, nullptr},
            {StateID::Chase,   EventID::EnemyLost,      StateID::Search, nullptr},
            {StateID::Attack,  EventID::LowHealth,      StateID::Flee,   nullptr},
            {StateID::Attack,  EventID::EnemyLost,      StateID::Chase,  nullptr},
            {StateID::Flee,    EventID::EnemyLost,      StateID::Heal,   nullptr},
            {StateID::Heal,    EventID::HealthRestored,  StateID::Patrol, nullptr},
            {StateID::Heal,    EventID::EnemyDetected,   StateID::Chase, nullptr},
            // ... 更多转换规则
        };

        // 实例化所有状态
        stateInstances[StateID::Patrol] = new PatrolState();
        stateInstances[StateID::Chase] = new ChaseState();
        stateInstances[StateID::Attack] = new AttackState();
        // ...
    }

    void HandleEvent(EventID event) {
        for (const auto& t : transitionTable) {
            if (t.from == currentState && t.event == event) {
                if (stateInstances.count(currentState))
                    stateInstances[currentState]->Exit(owner);
                currentState = t.to;
                if (t.action) t.action(owner);
                if (stateInstances.count(currentState))
                    stateInstances[currentState]->Enter(owner);
                return;
            }
        }
        // 未定义的转换：忽略或记录警告
    }

    void Update(float dt) {
        if (stateInstances.count(currentState))
            stateInstances[currentState]->Update(owner, dt);
    }

    StateID GetCurrentStateID() const { return currentState; }
};
```

### 6.2 转换表的优势

| 维度 | 分散式FSM | 表驱动FSM |
|------|-----------|-----------|
| 可读性 | 需要遍历所有状态类 | 一张表看全部转换 |
| 可维护性 | 新增转换需修改多个类 | 仅修改表 |
| 调试 | 难以断点跟踪 | 可在表查询处断点 |
| 数据驱动 | 难以从文件加载 | 表可序列化为JSON/XML |
| 性能 | 分支预测友好 | 查表开销 |

## 7. FSM与动画系统的集成

### 7.1 UE动画蓝图状态机

UE的Animation Blueprint中内置了状态机系统：

```cpp
// UE5动画蓝图状态机示例
UCLASS()
class UMyAnimInstance : public UAnimInstance {
    GENERATED_BODY()

public:
    // 状态机参数
    UPROPERTY(BlueprintReadWrite, Category = "Locomotion")
    float Speed;

    UPROPERTY(BlueprintReadWrite, Category = "Locomotion")
    bool bIsInAir;

    UPROPERTY(BlueprintReadWrite, Category = "Locomotion")
    bool bIsCrouching;

    // 状态转换逻辑（在蓝图中编辑，此处为代码等价物）
    void UpdateAnimationState(float DeltaTime) {
        if (bIsInAir) {
            // 跳跃/下落状态
        } else if (Speed > 0.1f) {
            if (bIsCrouching) {
                // 蹲下行走
            } else if (Speed > 300.0f) {
                // 奔跑
            } else {
                // 行走
            }
        } else {
            // 待机
        }
    }
};
```

### 7.2 Unity Animator状态机

Unity的Animator Controller是可视化FSM编辑器：

```csharp
// Unity Animator参数驱动
public class CharacterAnimator : MonoBehaviour {
    private Animator animator;

    void Update() {
        animator = GetComponent<Animator>();

        // 设置状态机参数
        animator.SetFloat("Speed", currentSpeed);
        animator.SetBool("IsGrounded", isGrounded);
        animator.SetTrigger("Attack");
        animator.SetInteger("ComboStep", comboStep);

        // 状态转换由Animator Controller定义
        // 代码只需设置参数，状态机自动转换
    }
}
```

## 8. 性能分析

### 8.1 时间复杂度

| 操作 | 扁平FSM | 分层FSM | 栈FSM | 表驱动FSM |
|------|---------|---------|-------|-----------|
| Update | O(1) | O(D) | O(1) | O(1) |
| 转换 | O(1) | O(D) | O(1) | O(T) |
| 事件处理 | O(1) | O(D) | O(1) | O(T) |

D = 层级深度，T = 转换表大小

### 8.2 内存分析

- 每个状态实例：~32-64字节（取决于状态内部数据）
- 转换表：每条转换~32字节（状态ID x2 + 事件ID + 函数指针）
- 栈FSM额外栈空间：每层~8字节（指针）

## 9. 常见陷阱

1. **状态爆炸**：N个状态的完全连接FSM需要N*(N-1)条转换边，使用分层或表驱动缓解
2. **Enter/Exit不对称**：Enter中申请的资源在Exit中未释放，导致内存泄漏
3. **瞬时转换死循环**：A→B→A→B在一帧内无限循环，需要转换锁或最小驻留时间
4. **丢失状态**：删除状态时未更新所有引用该状态的转换
5. **全局状态滥用**：在全局状态中处理太多逻辑，变成"万能状态"
6. **事件时序问题**：同一帧内多个事件，处理顺序影响最终状态

## 10. 实际游戏案例

### 案例1：超级马里奥系列

马里奥的形态系统使用FSM：
- 小马里奥 → 蘑菇 → 大马里奥 → 火球马里奥
- 每种形态有独立的动作状态（站立、跑、跳、游泳、滑行）
- 形态转换通过FSM管理（受伤降级、吃道具升级）
- 动画状态机与角色形态FSM耦合

### 案例2：格斗游戏（街霸/拳皇）

格斗游戏的出招系统是典型的分层FSM：
- 顶层：站立/蹲下/跳跃
- 中层：普通攻击/特殊技/必杀技
- 底层：攻击动画帧序列
- 转换条件基于输入序列（→↓↘+P = 波动拳）

### 案例3：塔防游戏

塔防中的敌人AI使用FSM：
- 移动：沿预设路径前进
- 减速：被减速塔命中
- 眩晕：被控制技能命中
- 死亡：血量归零
- 抵达终点：扣除玩家生命值
