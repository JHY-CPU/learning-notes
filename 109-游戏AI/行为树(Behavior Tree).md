# 行为树（Behavior Tree）

## 1. 核心理论

### 1.1 什么是行为树

行为树是一种用于描述AI决策逻辑的分层节点树结构。每次执行（称为"Tick"）从根节点开始，自顶向下评估节点，直到叶节点执行具体行为。每个节点返回三种状态之一：
- **Success（成功）**：节点完成其任务
- **Failure（失败）**：节点未能完成任务
- **Running（运行中）**：节点需要更多时间完成

行为树最早由游戏行业在2000年代中期发展成熟，被Halo 2的AI团队系统性地应用于游戏开发。相比FSM，行为树具有更好的**组合性**、**可复用性**和**可维护性**。

### 1.2 Tick机制

行为树的Tick是定期驱动树执行的机制，类似于时钟脉冲：

```
Tick频率：
- 高频Tick（每帧/每0.1秒）：用于核心战斗AI
- 中频Tick（每0.5秒）：用于普通NPC行为
- 低频Tick（每1-2秒）：用于远距离/非关键AI

Tick流程：
1. 从根节点开始评估
2. 复合节点递归评估子节点
3. 叶节点执行动作或检查条件
4. 返回状态沿树向上传播
5. Running节点记录执行位置，下次Tick继续
```

### 1.3 行为树与FSM的本质区别

FSM通过**状态+转换边**描述行为，而行为树通过**树形组合逻辑**描述行为。关键区别：

- FSM的状态是扁平的（或有限层级），行为树天然分层
- FSM的状态转换是显式定义的（硬编码），行为树的"转换"由树结构隐式决定
- FSM难以复用子状态，行为树的子树可被多处引用
- 行为树更容易进行**数据驱动**设计（节点树可从文件加载）

## 2. 节点类型详解

### 2.1 复合节点（Composite Nodes）

复合节点是包含子节点的非叶节点，决定子节点的执行策略。

#### 选择器（Selector / Priority）

选择器按顺序执行子节点，**任一子节点成功即返回Success**，全部失败则返回Failure。类似于逻辑"或"（OR）。

```cpp
class Selector : public BTNode {
protected:
    std::vector<std::unique_ptr<BTNode>> children;

public:
    Selector(std::vector<std::unique_ptr<BTNode>> kids)
        : children(std::move(kids)) {}

    Status Tick(Blackboard& bb) override {
        for (auto& child : children) {
            Status s = child->Tick(bb);
            if (s == Status::Success) return Status::Success;
            if (s == Status::Running) return Status::Running;
            // Failure则继续执行下一个子节点
        }
        return Status::Failure;
    }
};
```

#### 序列（Sequence）

序列按顺序执行子节点，**任一子节点失败即返回Failure**，全部成功则返回Success。类似于逻辑"与"（AND）。

```cpp
class Sequence : public BTNode {
protected:
    std::vector<std::unique_ptr<BTNode>> children;
    int currentIndex = 0;

public:
    Sequence(std::vector<std::unique_ptr<BTNode>> kids)
        : children(std::move(kids)) {}

    Status Tick(Blackboard& bb) override {
        while (currentIndex < (int)children.size()) {
            Status s = children[currentIndex]->Tick(bb);
            if (s == Status::Failure) {
                currentIndex = 0;
                return Status::Failure;
            }
            if (s == Status::Running) {
                return Status::Running;
            }
            // Success，继续下一个
            currentIndex++;
        }
        currentIndex = 0;
        return Status::Success;
    }

    void Reset() override { currentIndex = 0; }
};
```

#### 并行节点（Parallel）

并行节点同时执行所有子节点，根据策略决定返回值。

```cpp
class Parallel : public BTNode {
    std::vector<std::unique_ptr<BTNode>> children;
    int successThreshold;  // 成功子节点数阈值
    int failureThreshold;  // 失败子节点数阈值
    bool resetOnComplete;  // 完成后是否重置所有子节点

public:
    Parallel(std::vector<std::unique_ptr<BTNode>> kids,
             int sucThresh = -1, int failThresh = 1, bool reset = true)
        : children(std::move(kids)),
          successThreshold(sucThresh == -1 ? (int)children.size() : sucThresh),
          failureThreshold(failThresh),
          resetOnComplete(reset) {}

    Status Tick(Blackboard& bb) override {
        int successCount = 0, failureCount = 0;

        for (auto& child : children) {
            Status s = child->Tick(bb);
            if (s == Status::Success) successCount++;
            else if (s == Status::Failure) failureCount++;
        }

        if (successCount >= successThreshold) {
            if (resetOnComplete) Reset();
            return Status::Success;
        }
        if (failureCount >= failureThreshold) {
            if (resetOnComplete) Reset();
            return Status::Failure;
        }
        return Status::Running;
    }

    void Reset() override {
        for (auto& child : children) child->Reset();
    }
};
```

#### 随机选择器（Random Selector）

随机排列子节点顺序后执行，用于增加AI行为的随机性。

```cpp
class RandomSelector : public Selector {
    std::vector<int> shuffledIndices;

public:
    RandomSelector(std::vector<std::unique_ptr<BTNode>> kids)
        : Selector(std::move(kids)) {
        ResetShuffle();
    }

    void ResetShuffle() {
        shuffledIndices.resize(children.size());
        for (int i = 0; i < (int)children.size(); i++)
            shuffledIndices[i] = i;
        std::random_shuffle(shuffledIndices.begin(), shuffledIndices.end());
    }

    Status Tick(Blackboard& bb) override {
        for (int idx : shuffledIndices) {
            Status s = children[idx]->Tick(bb);
            if (s == Status::Success) {
                ResetShuffle();
                return Status::Success;
            }
            if (s == Status::Running) return Status::Running;
        }
        ResetShuffle();
        return Status::Failure;
    }
};
```

### 2.2 装饰器节点（Decorator Nodes）

装饰器是只有一个子节点的节点，用于修改子节点的行为或返回值。

#### 反转装饰器（Inverter）

```cpp
class Inverter : public BTNode {
    std::unique_ptr<BTNode> child;

public:
    Inverter(std::unique_ptr<BTNode> c) : child(std::move(c)) {}

    Status Tick(Blackboard& bb) override {
        Status s = child->Tick(bb);
        if (s == Status::Success) return Status::Failure;
        if (s == Status::Failure) return Status::Success;
        return s; // Running保持不变
    }
};
```

#### 重复装饰器（Repeater / UntilSuccess / UntilFailure）

```cpp
// 重复N次或直到成功/失败
class Repeater : public BTNode {
    std::unique_ptr<BTNode> child;
    int maxRepeats;  // -1表示无限重复
    int currentCount = 0;

public:
    Repeater(std::unique_ptr<BTNode> c, int repeats = -1)
        : child(std::move(c)), maxRepeats(repeats) {}

    Status Tick(Blackboard& bb) override {
        while (true) {
            Status s = child->Tick(bb);
            if (s == Status::Running) return Status::Running;

            currentCount++;
            if (maxRepeats > 0 && currentCount >= maxRepeats) {
                currentCount = 0;
                return s; // 达到次数限制，返回最后一次结果
            }
            if (s == Status::Failure) {
                currentCount = 0;
                return Status::Failure;
            }
            // Success时重置子节点继续重复
            child->Reset();
        }
    }
};

class UntilSuccess : public BTNode {
    std::unique_ptr<BTNode> child;
public:
    UntilSuccess(std::unique_ptr<BTNode> c) : child(std::move(c)) {}
    Status Tick(Blackboard& bb) override {
        Status s = child->Tick(bb);
        if (s == Status::Success) return Status::Success;
        if (s == Status::Running) return Status::Running;
        child->Reset();
        return Status::Running; // 失败后继续尝试
    }
};
```

#### 冷却装饰器（Cooldown）

```cpp
class Cooldown : public BTNode {
    std::unique_ptr<BTNode> child;
    float cooldownTime;
    float lastExecTime = -999.0f;

public:
    Cooldown(std::unique_ptr<BTNode> c, float cd)
        : child(std::move(c)), cooldownTime(cd) {}

    Status Tick(Blackboard& bb) override {
        float currentTime = bb.GetFloat("gameTime");
        if (currentTime - lastExecTime < cooldownTime) {
            return Status::Failure; // 冷却中
        }
        Status s = child->Tick(bb);
        if (s == Status::Success) {
            lastExecTime = currentTime;
        }
        return s;
    }
};
```

#### 条件守卫（Guard / Check）

```cpp
class Guard : public BTNode {
    std::function<bool(Blackboard&)> condition;
    std::unique_ptr<BTNode> child;

public:
    Guard(std::function<bool(Blackboard&)> cond, std::unique_ptr<BTNode> c)
        : condition(std::move(cond)), child(std::move(c)) {}

    Status Tick(Blackboard& bb) override {
        if (!condition(bb)) return Status::Failure;
        return child->Tick(bb);
    }
};
```

### 2.3 叶节点（Leaf Nodes）

#### 条件节点（Condition）

```cpp
class Condition : public BTNode {
    std::string name;
    std::function<bool(Blackboard&)> check;

public:
    Condition(const std::string& n, std::function<bool(Blackboard&)> fn)
        : name(n), check(std::move(fn)) {}

    Status Tick(Blackboard& bb) override {
        return check(bb) ? Status::Success : Status::Failure;
    }

    std::string GetName() const override { return name; }
};
```

#### 动作节点（Action）

```cpp
class Action : public BTNode {
    std::string name;
    std::function<Status(Blackboard&)> action;

public:
    Action(const std::string& n, std::function<Status(Blackboard&)> fn)
        : name(n), action(std::move(fn)) {}

    Status Tick(Blackboard& bb) override {
        return action(bb);
    }

    std::string GetName() const override { return name; }
};
```

## 3. 黑板系统（Blackboard）

### 3.1 黑板设计

黑板是行为树的**共享数据存储**，所有节点通过黑板读写数据，实现松耦合的节点间通信。

```cpp
class Blackboard {
    // 类型安全的键值存储
    std::unordered_map<std::string, int> intData;
    std::unordered_map<std::string, float> floatData;
    std::unordered_map<std::string, bool> boolData;
    std::unordered_map<std::string, std::string> stringData;
    std::unordered_map<std::string, void*> pointerData; // 用于对象引用

public:
    // 整型
    void SetInt(const std::string& key, int val) { intData[key] = val; }
    int GetInt(const std::string& key, int defaultVal = 0) const {
        auto it = intData.find(key);
        return it != intData.end() ? it->second : defaultVal;
    }

    // 浮点型
    void SetFloat(const std::string& key, float val) { floatData[key] = val; }
    float GetFloat(const std::string& key, float defaultVal = 0.0f) const {
        auto it = floatData.find(key);
        return it != floatData.end() ? it->second : defaultVal;
    }

    // 布尔型
    void SetBool(const std::string& key, bool val) { boolData[key] = val; }
    bool GetBool(const std::string& key, bool defaultVal = false) const {
        auto it = boolData.find(key);
        return it != boolData.end() ? it->second : defaultVal;
    }

    // 字符串
    void SetString(const std::string& key, const std::string& val) { stringData[key] = val; }
    std::string GetString(const std::string& key, const std::string& def = "") const {
        auto it = stringData.find(key);
        return it != stringData.end() ? it->second : def;
    }

    // 对象引用（类型擦除）
    template<typename T>
    void SetPointer(const std::string& key, T* ptr) {
        pointerData[key] = static_cast<void*>(ptr);
    }
    template<typename T>
    T* GetPointer(const std::string& key) const {
        auto it = pointerData.find(key);
        return it != pointerData.end() ? static_cast<T*>(it->second) : nullptr;
    }

    // 批量操作
    void Clear() {
        intData.clear(); floatData.clear();
        boolData.clear(); stringData.clear(); pointerData.clear();
    }

    bool HasKey(const std::string& key) const {
        return intData.count(key) || floatData.count(key) ||
               boolData.count(key) || stringData.count(key) || pointerData.count(key);
    }
};
```

### 3.2 黑板层级

实际游戏中黑板常分多层：

```
全局黑板（Global Blackboard）
├── 存储世界状态：时间、天气、任务状态
├── 所有AI共享
└── 生命周期：游戏运行期间

AI实例黑板（Per-Agent Blackboard）
├── 存储个体数据：血量、弹药、目标
├── 每个AI独有
└── 生命周期：AI创建-销毁

子树黑板（Sub-Tree Blackboard）
├── 存储子树内部临时数据
├── 父子树不可访问
└── 生命周期：子树执行期间
```

```cpp
class HierarchicalBlackboard {
    Blackboard* global;     // 全局黑板（只读引用）
    Blackboard local;       // 实例黑板
    Blackboard* parent;     // 父级黑板（用于子树黑板）

public:
    HierarchicalBlackboard(Blackboard* globalBB = nullptr)
        : global(globalBB), parent(nullptr) {}

    // 查询时按优先级：local -> parent -> global
    float GetFloat(const std::string& key, float def = 0.0f) const {
        if (local.HasKey(key)) return local.GetFloat(key, def);
        if (parent && parent->HasKey(key)) return parent->GetFloat(key, def);
        if (global && global->HasKey(key)) return global->GetFloat(key, def);
        return def;
    }

    // 写入永远写到local
    void SetFloat(const std::string& key, float val) { local.SetFloat(key, val); }
};
```

## 4. UE行为树系统详解

### 4.1 UE行为树架构

UE的AI系统以**AIController**为核心，驱动**行为树**和**黑板**：

```
AIController
├── 黑板数据资产（UBlackboardData）
│   ├── 敌人位置（Vector）
│   ├── 是否看到敌人（Bool）
│   ├── 当前任务（Enum）
│   └── ...
├── 行为树数据资产（UBehaviorTree）
│   ├── 根节点
│   ├── 选择器 / 序列
│   ├── 装饰器（BTDecorator）
│   ├── 服务（BTService）
│   └── 任务（BTTask）
└── 感知系统（AIPerceptionComponent）
    ├── 视觉刺激
    ├── 听觉刺激
    └── 伤害刺激
```

### 4.2 UE特有节点

**服务（BTService）**：在后台定期运行，更新黑板数据。

```cpp
// UE服务示例：定期更新与目标的距离
UCLASS()
class UBTService_UpdateDistance : public UBTService {
    GENERATED_BODY()

public:
    UBTService_UpdateDistance() {
        NodeName = "Update Distance";
        Interval = 0.5f;       // 每0.5秒执行一次
        RandomDeviation = 0.1f;
    }

    virtual void TickNode(UBehaviorTreeComponent& OwnerComp,
                          uint8* NodeMemory, float DeltaSeconds) override {
        AAIController* Controller = OwnerComp.GetAIOwner();
        if (!Controller) return;

        AActor* Target = Cast<AActor>(
            OwnerComp.GetBlackboardComponent()->GetValueAsObject("TargetActor"));
        if (!Target) return;

        float Dist = FVector::Dist(
            Controller->GetPawn()->GetActorLocation(),
            Target->GetActorLocation());
        OwnerComp.GetBlackboardComponent()->SetValueAsFloat("DistanceToTarget", Dist);
    }
};
```

**任务（BTTask）**：执行具体行为的叶节点。

```cpp
// UE任务示例：移动到黑板标记的位置
UCLASS()
class UBTTask_MoveToTarget : public UBTTaskNode {
    GENERATED_BODY()

    virtual EBTNodeResult::Type ExecuteTask(
        UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) override
    {
        AAIController* Controller = OwnerComp.GetAIOwner();
        FVector TargetLoc = OwnerComp.GetBlackboardComponent()
            ->GetValueAsVector("TargetLocation");

        EPathFollowingRequestResult::Type Result =
            Controller->MoveToLocation(TargetLoc, 50.0f);

        if (Result == EPathFollowingRequestResult::RequestSuccessful) {
            return EBTNodeResult::InProgress; // 移动中
        }
        return EBTNodeResult::Failed;
    }

    // 移动完成时回调
    virtual void TickTask(UBehaviorTreeComponent& OwnerComp,
                          uint8* NodeMemory, float DeltaSeconds) override {
        if (OwnerComp.GetAIOwner()->GetMoveStatus() == EPathFollowingStatus::Idle) {
            FinishLatentTask(OwnerComp, EBTNodeResult::Succeeded);
        }
    }
};
```

**装饰器（BTDecorator）**：条件检查/流程控制。

```cpp
// UE装饰器示例：检查距离条件
UCLASS()
class UBTDecorator_CheckDistance : public UBTDecorator {
    GENERATED_BODY()

    UPROPERTY(EditAnywhere)
    float MaxDistance = 1000.0f;

    virtual bool CalculateRawConditionValue(
        UBehaviorTreeComponent& OwnerComp, uint8* NodeMemory) const override
    {
        float Dist = OwnerComp.GetBlackboardComponent()
            ->GetValueAsFloat("DistanceToTarget");
        return Dist <= MaxDistance;
    }
};
```

## 5. 行为树构建器模式

### 5.1 流畅接口构建行为树

```cpp
// 使用流式API构建行为树
class BTBuilder {
public:
    static std::unique_ptr<Selector> Selector_(
        std::initializer_list<std::unique_ptr<BTNode>> children)
    {
        std::vector<std::unique_ptr<BTNode>> v;
        for (auto& c : children) v.push_back(std::move(const_cast<std::unique_ptr<BTNode>&>(c)));
        return std::make_unique<Selector>(std::move(v));
    }

    static std::unique_ptr<Sequence> Sequence_(
        std::initializer_list<std::unique_ptr<BTNode>> children)
    {
        std::vector<std::unique_ptr<BTNode>> v;
        for (auto& c : children) v.push_back(std::move(const_cast<std::unique_ptr<BTNode>&>(c)));
        return std::make_unique<Sequence>(std::move(v));
    }

    static std::unique_ptr<Condition> Condition_(
        const std::string& name, std::function<bool(Blackboard&)> fn)
    {
        return std::make_unique<Condition>(name, std::move(fn));
    }

    static std::unique_ptr<Action> Action_(
        const std::string& name, std::function<Status(Blackboard&)> fn)
    {
        return std::make_unique<Action>(name, std::move(fn));
    }
};

// 使用示例
auto enemyAI = BTBuilder::Selector_({
    // 战斗分支
    BTBuilder::Sequence_({
        BTBuilder::Condition_("HasTarget", [](Blackboard& bb) {
            return bb.GetBool("hasTarget");
        }),
        BTBuilder::Condition_("InRange", [](Blackboard& bb) {
            return bb.GetFloat("distanceToTarget") < 2.0f;
        }),
        BTBuilder::Action_("Attack", [](Blackboard& bb) {
            std::cout << "攻击目标!\n";
            return Status::Success;
        })
    }),
    // 巡逻分支
    BTBuilder::Sequence_({
        BTBuilder::Condition_("HasWaypoint", [](Blackboard& bb) {
            return bb.GetBool("hasWaypoint");
        }),
        BTBuilder::Action_("Patrol", [](Blackboard& bb) {
            std::cout << "前往巡逻点\n";
            return Status::Success;
        })
    }),
    // 待机分支
    BTBuilder::Action_("Idle", [](Blackboard& bb) {
        std::cout << "待机中...\n";
        return Status::Success;
    })
});
```

## 6. 常见行为树模式

### 6.1 战斗AI行为树

```
[Selector: 战斗AI]
├── [Sequence: 躲避]
│   ├── [Condition: 血量 < 20%]
│   ├── [Condition: 有掩体]
│   └── [Action: 躲到掩体后]
├── [Sequence: 攻击]
│   ├── [Condition: 有目标]
│   ├── [Condition: 目标可见]
│   ├── [Selector: 攻击方式]
│   │   ├── [Sequence: 近战]
│   │   │   ├── [Condition: 距离 < 2m]
│   │   │   └── [Action: 近战攻击]
│   │   └── [Sequence: 远程]
│   │       ├── [Condition: 有弹药]
│   │       └── [Action: 射击]
│   └── [Cooldown: 1s]
├── [Sequence: 追击]
│   ├── [Condition: 有目标]
│   ├── [Condition: 目标不可达为假]
│   └── [Action: 移动到目标]
├── [Sequence: 搜索]
│   ├── [Condition: 曾看到目标]
│   └── [Action: 搜索最后已知位置]
└── [Action: 巡逻]
```

### 6.2 NPC日常行为树

```
[Selector: NPC日常]
├── [Sequence: 受伤处理]
│   ├── [Condition: 血量低]
│   └── [Action: 寻找医疗]
├── [Sequence: 危险响应]
│   ├── [Condition: 感知到威胁]
│   └── [Action: 躲避/逃跑]
├── [Sequence: 工作]
│   ├── [Condition: 是工作时间]
│   ├── [Action: 前往工作地点]
│   └── [Action: 执行工作]
├── [Sequence: 社交]
│   ├── [Condition: 附近有其他NPC]
│   └── [Selector: 社交方式]
│       ├── [Action: 打招呼]
│       ├── [Action: 聊天]
│       └── [Action: 交易]
└── [Action: 休息]
```

## 7. 性能优化

### 7.1 Tick频率控制

```cpp
class BehaviorTree {
    float tickInterval = 0.1f;  // 默认每0.1秒Tick一次
    float timeSinceLastTick = 0;
    float lastTickTime = 0;

    // 根据AI距离玩家调整Tick频率
    float GetAdaptiveTickInterval(float distanceToPlayer) const {
        if (distanceToPlayer < 20.0f) return 0.05f;   // 近距离：高频
        if (distanceToPlayer < 50.0f) return 0.1f;    // 中距离：中频
        if (distanceToPlayer < 100.0f) return 0.5f;   // 远距离：低频
        return 2.0f;                                     // 极远：极低频
    }
};
```

### 7.2 节点缓存与预热

```cpp
class CachedAction : public BTNode {
    Status lastResult = Status::Failure;
    float lastExecTime = 0;
    float cacheDuration = 0.5f;

public:
    Status Tick(Blackboard& bb) override {
        float now = bb.GetFloat("gameTime");
        if (now - lastExecTime < cacheDuration) return lastResult;

        lastExecTime = now;
        lastResult = ExecuteAction(bb);
        return lastResult;
    }

    virtual Status ExecuteAction(Blackboard& bb) = 0;
};
```

### 7.3 睡眠与唤醒

```cpp
class BehaviorTreeComponent {
    bool bIsSleeping = false;

public:
    void Sleep() { bIsSleeping = true; }
    void WakeUp() { bIsSleeping = false; }

    void Tick(float dt) {
        if (bIsSleeping) return; // 跳过休眠的AI
        root->Tick(blackboard);
    }
};
```

## 8. 性能分析

### 8.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| Tick（单次） | O(N) | N为树节点数 |
| 节点查找 | O(N) | 遍历树 |
| Running状态恢复 | O(D) | D为Running节点深度 |
| 黑板读写 | O(1) 均摊 | 哈希表操作 |

### 8.2 内存分析

| 组件 | 单实例大小 |
|------|-----------|
| 行为树节点 | 32-128字节（取决于节点类型） |
| 黑板条目 | 16-64字节（键+值） |
| 运行时状态 | 4-8字节/节点（Running状态等） |

## 9. 常见陷阱

1. **Running状态不当处理**：Running节点需要记录索引以便下次Tick继续，否则会重复执行
2. **黑板数据未清理**：AI销毁或切换行为树时黑板残留旧数据
3. **子树深度过大**：深层嵌套导致Tick递归开销增大
4. **并行节点停止策略**：并行节点被中断时，子节点的停止顺序影响资源释放
5. **服务更新频率**：服务（Service）的Tick间隔不当，导致数据过时或开销过高
6. **装饰器副作用**：装饰器中包含修改状态的逻辑，导致行为不可预测

## 10. 实际游戏案例

### 案例1：Halo 2 AI系统

Halo 2是行为树在游戏AI中的标志性应用：
- 敌人AI通过行为树管理掩护、射击、投掷手雷等战术
- 使用"AI指挥官"层协调小队行为
- 难度级别通过调整行为树参数（反应速度、精准度）实现

### 案例2：中土世界：魔多之影

兽人队长的AI系统基于行为树+个性系统：
- 每个兽人有独特的行为树变体（基于个性特质）
- 恐惧/弱点影响行为树分支权重
- 兽人等级提升时行为树节点解锁
- 复仇系统通过黑板与行为树联动

### 案例3：使命召唤系列

使命召唤的敌人AI使用行为树+感知系统：
- 视觉感知更新黑板中的"已知敌人位置"
- 行为树根据掩体可用性选择进攻/防守/撤退
- 小队长行为树协调小队火力压制与包抄
