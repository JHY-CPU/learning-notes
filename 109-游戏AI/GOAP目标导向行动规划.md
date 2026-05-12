# GOAP目标导向行动规划

## 1. 核心理论

### 1.1 GOAP的起源与定义

GOAP（Goal-Oriented Action Planning，目标导向行动规划）是一种AI规划系统，最早由Jeff Orkin在2003年的F.E.A.R.（First Encounter Assault Recon）游戏中提出。GOAP将AI的决策过程分为**目标设定**和**行动规划**两个阶段：

1. **目标设定**：AI根据当前世界状态选择一个目标（如"敌人被消灭"）
2. **行动规划**：规划器自动搜索从当前状态到达目标状态的最优行动序列

GOAP的核心洞察是：AI不需要被编程为"先做A，再做B，再做C"，而是只需要定义**每个行动的前置条件和效果**，以及**期望的目标状态**，规划器会自动找到从当前状态到目标的最优路径。

### 1.2 STRIPS规划模型

GOAP基于经典AI规划模型STRIPS（Stanford Research Institute Problem Solver）：

```
行动 = {
    名称（Name）,
    预条件（Preconditions）: 执行前必须满足的世界状态
    效果（Effects）: 执行后世界状态的改变
    代价（Cost）: 执行此行动的代价
}

世界状态 = {
    一系列键值对（属性名 -> 属性值）
}

目标 = {
    世界状态的部分描述（只需指定关键属性）
}
```

### 1.3 规划过程

GOAP的规划本质上是在**状态空间**中的**图搜索**：

- **节点**：世界状态（WorldState）
- **边**：行动（Action）——从一个状态转换到另一个状态
- **搜索算法**：A*搜索（以代价为g值，未满足目标数为h值）

```
起始状态 → [行动1] → 中间状态 → [行动2] → 中间状态 → ... → 目标状态
```

## 2. 世界状态表示

### 2.1 位集表示法

最高效的WorldState表示使用位集（BitSet），每个属性占1位：

```cpp
// 基于位集的WorldState（高效，最多支持64个属性）
struct WorldState {
    uint64_t values;    // 属性值的位集
    uint64_t dontCare;  // 忽略位集（1表示不关心此属性）

    WorldState() : values(0), dontCare(~0ULL) {}

    void SetFact(int index, bool value) {
        uint64_t mask = 1ULL << index;
        dontCare &= ~mask;         // 不再忽略此属性
        if (value) values |= mask;
        else values &= ~mask;
    }

    bool GetFact(int index) const {
        return (values >> index) & 1ULL;
    }

    bool DontCare(int index) const {
        return (dontCare >> index) & 1ULL;
    }

    // 判断此状态是否满足目标（goal是目标的部分状态描述）
    bool MeetsGoal(const WorldState& goal) const {
        uint64_t relevantBits = ~goal.dontCare;
        return (values & relevantBits) == (goal.values & relevantBits);
    }

    // 计算此状态与目标之间的不同属性数量（启发函数用）
    int HammingDistance(const WorldState& goal) const {
        uint64_t relevantBits = ~goal.dontCare;
        return __builtin_popcountll(
            (values & relevantBits) ^ (goal.values & relevantBits));
    }

    // 应用行动效果到此状态
    WorldState ApplyAction(const struct Action& action) const;
};
```

### 2.2 属性注册表

使用注册表管理属性名称到索引的映射：

```cpp
class FactRepository {
    std::unordered_map<std::string, int> factIndex;
    std::vector<std::string> factNames;
    int nextIndex = 0;

public:
    static FactRepository& Instance() {
        static FactRepository repo;
        return repo;
    }

    int Register(const std::string& name) {
        if (factIndex.count(name)) return factIndex[name];
        if (nextIndex >= 64) throw std::runtime_error("超过64个属性上限");
        factIndex[name] = nextIndex;
        factNames.push_back(name);
        return nextIndex++;
    }

    int GetIndex(const std::string& name) const {
        auto it = factIndex.find(name);
        return it != factIndex.end() ? it->second : -1;
    }

    const std::string& GetName(int index) const { return factNames[index]; }
};

// 便捷宏
#define FACT(name) FactRepository::Instance().Register(name)
```

## 3. 行动定义

### 3.1 Action结构

```cpp
struct Action {
    std::string name;
    float cost;
    WorldState preconditions;
    WorldState effects;

    // 执行回调（规划时不使用，执行时调用）
    std::function<bool()> execute;

    // 前置条件验证回调（用于无法用位集表示的复杂条件）
    std::function<bool()> validate;

    // 运行时前提检查（动态条件：如目标是否还活着）
    std::function<bool()> runtimeCheck;

    Action(const std::string& n, float c = 1.0f)
        : name(n), cost(c), execute(nullptr), validate(nullptr), runtimeCheck(nullptr) {}

    Action& Pre(const std::string& fact, bool value) {
        preconditions.SetFact(FACT(fact), value);
        return *this;
    }

    Action& Post(const std::string& fact, bool value) {
        effects.SetFact(FACT(fact), value);
        return *this;
    }

    Action& Cost(float c) { cost = c; return *this; }

    Action& Do(std::function<bool()> fn) { execute = std::move(fn); return *this; }
};
```

### 3.2 构建行动集合

```cpp
class ActionSet {
    std::vector<Action> actions;

public:
    ActionSet() {
        // 攻击行动
        actions.emplace_back("MeleeAttack", 1.0f)
            .Pre("hasWeapon", true)
            .Pre("enemyInRange", true)
            .Pre("enemyAlive", true)
            .Post("enemyHurt", true)
            .Do([]() { std::cout << "执行近战攻击\n"; return true; });

        // 射击行动
        actions.emplace_back("Shoot", 1.5f)
            .Pre("hasWeapon", true)
            .Pre("hasAmmo", true)
            .Pre("enemyInSight", true)
            .Post("enemyHurt", true)
            .Post("hasAmmo", false) // 射击消耗弹药
            .Do([]() { std::cout << "执行射击\n"; return true; });

        // 装弹行动
        actions.emplace_back("Reload", 2.0f)
            .Pre("hasWeapon", true)
            .Pre("hasAmmo", false)
            .Post("hasAmmo", true)
            .Do([]() { std::cout << "执行装弹\n"; return true; });

        // 搜索弹药
        actions.emplace_back("SearchAmmo", 3.0f)
            .Post("hasAmmo", true)
            .Do([]() { std::cout << "搜索弹药\n"; return true; });

        // 移动到目标
        actions.emplace_back("MoveToEnemy", 2.0f)
            .Pre("enemyAlive", true)
            .Post("enemyInRange", true)
            .Post("enemyInSight", true)
            .Do([]() { std::cout << "移动到敌人位置\n"; return true; });

        // 拾取武器
        actions.emplace_back("PickupWeapon", 2.0f)
            .Pre("hasWeapon", false)
            .Post("hasWeapon", true)
            .Do([]() { std::cout << "拾取武器\n"; return true; });
    }

    const std::vector<Action>& GetActions() const { return actions; }

    // 获取前置条件被当前状态满足的行动
    std::vector<const Action*> GetApplicableActions(const WorldState& state) const {
        std::vector<const Action*> result;
        for (const auto& action : actions) {
            if (state.MeetsGoal(action.preconditions)) {
                result.push_back(&action);
            }
        }
        return result;
    }
};
```

## 4. 完整GOAP规划器

### 4.1 A*搜索实现

```cpp
class GOAPPlanner {
public:
    struct PlanNode {
        WorldState state;
        float gCost;         // 从起点的实际代价
        float fCost;         // g + h
        const Action* action; // 到达此状态的行动（起始节点为null）
        PlanNode* parent;    // 父节点

        PlanNode(WorldState s, float g, float f, const Action* a, PlanNode* p)
            : state(s), gCost(g), fCost(f), action(a), parent(p) {}
    };

    struct PlanNodeCompare {
        bool operator()(const PlanNode* a, const PlanNode* b) const {
            return a->fCost > b->fCost;
        }
    };

    struct WorldStateHash {
        size_t operator()(const WorldState& ws) const {
            return std::hash<uint64_t>()(ws.values ^ (ws.dontCare << 1));
        }
    };

    struct WorldStateEqual {
        bool operator()(const WorldState& a, const WorldState& b) const {
            return a.values == b.values && a.dontCare == b.dontCare;
        }
    };

    // 规划入口
    std::vector<const Action*> Plan(
        const WorldState& start,
        const WorldState& goal,
        const std::vector<Action>& availableActions)
    {
        if (start.MeetsGoal(goal)) return {}; // 已经满足目标

        // OpenList（最小堆）
        std::priority_queue<PlanNode*, std::vector<PlanNode*>, PlanNodeCompare> openList;
        // CloseList
        std::unordered_set<WorldState, WorldStateHash, WorldStateEqual> closedSet;

        // 起始节点
        float h = (float)start.HammingDistance(goal);
        PlanNode* startNode = new PlanNode(start, 0.0f, h, nullptr, nullptr);
        openList.push(startNode);

        int nodesExplored = 0;
        const int MAX_NODES = 10000; // 防止搜索空间爆炸

        while (!openList.empty() && nodesExplored < MAX_NODES) {
            PlanNode* current = openList.top();
            openList.pop();
            nodesExplored++;

            // 检查是否到达目标
            if (current->state.MeetsGoal(goal)) {
                std::vector<const Action*> plan = ReconstructPlan(current);
                Cleanup(openList);
                return plan;
            }

            // 跳过已访问状态
            if (closedSet.count(current->state)) {
                delete current;
                continue;
            }
            closedSet.insert(current->state);

            // 扩展后继：尝试应用每个可行的行动
            for (const Action& action : availableActions) {
                // 检查行动的预条件是否被当前状态满足
                if (!current->state.MeetsGoal(action.preconditions)) continue;

                // 生成新状态
                WorldState newState = ApplyAction(current->state, action);

                // 跳过已访问的状态
                if (closedSet.count(newState)) continue;

                float newG = current->gCost + action.cost;
                float newH = (float)newState.HammingDistance(goal);
                float newF = newG + newH;

                PlanNode* newNode = new PlanNode(
                    newState, newG, newF, &action, current);
                openList.push(newNode);
            }
        }

        Cleanup(openList);
        return {}; // 无解
    }

private:
    // 应用行动效果到状态
    WorldState ApplyAction(const WorldState& state, const Action& action) {
        WorldState result = state;
        // 遍历效果中的每个属性
        for (int i = 0; i < 64; i++) {
            if (!action.effects.DontCare(i)) {
                result.SetFact(i, action.effects.GetFact(i));
            }
        }
        return result;
    }

    // 回溯规划路径
    std::vector<const Action*> ReconstructPlan(PlanNode* node) {
        std::vector<const Action*> plan;
        while (node && node->action) {
            plan.push_back(node->action);
            node = node->parent;
        }
        std::reverse(plan.begin(), plan.end());
        return plan;
    }

    // 清理内存
    void Cleanup(std::priority_queue<PlanNode*, std::vector<PlanNode*>, PlanNodeCompare>& q) {
        while (!q.empty()) { delete q.top(); q.pop(); }
    }
};
```

### 4.2 规划器使用示例

```cpp
int main() {
    // 定义当前世界状态
    WorldState currentState;
    currentState.SetFact(FACT("hasWeapon"), false);
    currentState.SetFact(FACT("hasAmmo"), false);
    currentState.SetFact(FACT("enemyAlive"), true);
    currentState.SetFact(FACT("enemyInRange"), false);
    currentState.SetFact(FACT("enemyInSight"), true);
    currentState.SetFact(FACT("enemyHurt"), false);

    // 定义目标状态
    WorldState goal;
    goal.SetFact(FACT("enemyHurt"), true);

    // 获取行动集合
    ActionSet actionSet;

    // 执行规划
    GOAPPlanner planner;
    auto plan = planner.Plan(currentState, goal, actionSet.GetActions());

    if (plan.empty()) {
        std::cout << "无法找到可行计划!\n";
    } else {
        std::cout << "规划结果:\n";
        float totalCost = 0;
        for (const Action* action : plan) {
            std::cout << "  -> " << action->name << " (代价: " << action->cost << ")\n";
            totalCost += action->cost;
        }
        std::cout << "总代价: " << totalCost << "\n";

        // 执行计划
        for (const Action* action : plan) {
            if (action->runtimeCheck && !action->runtimeCheck()) {
                std::cout << "行动前置条件失效，重新规划...\n";
                // 重新规划...
                break;
            }
            if (action->execute) action->execute();
        }
    }

    return 0;
}
```

**可能的输出**：
```
规划结果:
  -> PickupWeapon (代价: 2)
  -> MoveToEnemy (代价: 2)
  -> Shoot (代价: 1.5)
总代价: 5.5

拾取武器
移动到敌人位置
执行射击
```

## 5. 目标管理

### 5.1 目标优先级系统

AI通常有多个可能的目标，需要一个优先级系统来选择当前目标：

```cpp
struct Goal {
    std::string name;
    WorldState desiredState;
    float priority;          // 静态优先级
    std::function<float()> dynamicPriority; // 动态优先级函数
};

class GoalSelector {
    std::vector<Goal> goals;

public:
    void AddGoal(const std::string& name, const WorldState& state,
                 float priority, std::function<float()> dynPri = nullptr) {
        goals.push_back({name, state, priority, std::move(dynPri)});
    }

    const Goal* SelectGoal(const WorldState& currentState) const {
        const Goal* bestGoal = nullptr;
        float bestPriority = -1.0f;

        for (const auto& goal : goals) {
            // 跳过已满足的目标
            if (currentState.MeetsGoal(goal.desiredState)) continue;

            float pri = goal.priority;
            if (goal.dynamicPriority) pri += goal.dynamicPriority();

            if (pri > bestPriority) {
                bestPriority = pri;
                bestGoal = &goal;
            }
        }
        return bestGoal;
    }
};

// 使用示例
GoalSelector goalSelector;
goalSelector.AddGoal("KillEnemy",
    [](){ WorldState g; g.SetFact(FACT("enemyHurt"), true); return g; }(),
    10.0f,
    []() { return ai->IsInCombat() ? 5.0f : 0.0f; } // 战斗中优先级更高
);
goalSelector.AddGoal("Survive",
    [](){ WorldState g; g.SetFact(FACT("isHealthy"), true); return g; }(),
    8.0f,
    []() { return (1.0f - ai->GetHealthPercent()) * 10.0f; } // 血量越低优先级越高
);
goalSelector.AddGoal("Explore",
    [](){ WorldState g; g.SetFact(FACT("hasExplored"), true); return g; }(),
    3.0f
);
```

## 6. 增量规划与重规划

### 6.1 为什么需要重规划

游戏世界是动态的，行动执行过程中世界状态可能改变：
- 目标敌人死亡
- 出现新的威胁
- 行动执行失败（如装弹时被打断）
- 玩家干预改变环境

### 6.2 增量重规划

```cpp
class GOAPAgent {
    GOAPPlanner planner;
    ActionSet actionSet;
    GoalSelector goalSelector;

    WorldState currentState;
    std::vector<const Action*> currentPlan;
    int planStep = 0;

    float replanCooldown = 0.5f;
    float lastReplanTime = 0;

public:
    void Update(float gameTime, float dt) {
        // 如果当前计划为空或执行完毕，选择新目标并规划
        if (currentPlan.empty() || planStep >= (int)currentPlan.size()) {
            SelectNewGoalAndPlan();
            return;
        }

        // 执行当前步骤
        const Action* currentAction = currentPlan[planStep];

        // 检查运行时前提
        if (currentAction->runtimeCheck && !currentAction->runtimeCheck()) {
            // 条件失效，需要重规划
            if (gameTime - lastReplanTime > replanCooldown) {
                std::cout << "条件失效，重新规划...\n";
                SelectNewGoalAndPlan();
                lastReplanTime = gameTime;
            }
            return;
        }

        // 执行行动
        if (currentAction->execute && currentAction->execute()) {
            // 行动成功，更新世界状态
            for (int i = 0; i < 64; i++) {
                if (!currentAction->effects.DontCare(i)) {
                    currentState.SetFact(i, currentAction->effects.GetFact(i));
                }
            }
            planStep++;
        } else {
            // 行动执行失败，重规划
            std::cout << "行动 " << currentAction->name << " 失败，重新规划...\n";
            SelectNewGoalAndPlan();
            lastReplanTime = gameTime;
        }
    }

    void SelectNewGoalAndPlan() {
        const Goal* goal = goalSelector.SelectGoal(currentState);
        if (!goal) {
            currentPlan.clear();
            planStep = 0;
            return;
        }

        currentPlan = planner.Plan(currentState, goal->desiredState,
                                     actionSet.GetActions());
        planStep = 0;

        if (currentPlan.empty()) {
            std::cout << "目标 " << goal->name << " 无法找到可行计划\n";
        } else {
            std::cout << "为目标 " << goal->name << " 生成计划: ";
            for (const Action* a : currentPlan) std::cout << a->name << " -> ";
            std::cout << "完成\n";
        }
    }
};
```

## 7. GOAP与行为树混合架构

### 7.1 架构设计

GOAP和行为树并非互斥，可以混合使用：

```
[行为树根节点]
├── [Selector: 顶层决策]
│   ├── [Sequence: GOAP规划执行]
│   │   ├── [Condition: GOAP计划存在]
│   │   ├── [Action: 执行当前GOAP步骤]
│   │   └── [Action: 推进计划步骤]
│   ├── [Sequence: 紧急响应]
│   │   ├── [Condition: 被攻击]
│   │   └── [Action: 闪避]
│   └── [Sequence: 触发GOAP]
│       ├── [Condition: 需要规划]
│       └── [Action: 运行GOAP规划器]
```

**优势**：
- 行为树处理结构化的高频行为（移动、动画、紧急响应）
- GOAP处理需要长期规划的复杂任务
- 结合了行为树的可预测性和GOAP的灵活性

### 7.2 行为树节点调用GOAP

```cpp
class BTTask_RunGOAP : public BTNode {
    GOAPPlanner planner;
    std::vector<const Action*> plan;

public:
    Status Tick(Blackboard& bb) override {
        WorldState start = bb.GetPointer<WorldState>("currentWorldState")[0];
        WorldState goal = bb.GetPointer<WorldState>("currentGoal")[0];
        auto* actions = bb.GetPointer<std::vector<Action>>("availableActions");

        if (plan.empty()) {
            plan = planner.Plan(start, goal, *actions);
            if (plan.empty()) return Status::Failure;
            bb.SetInt("goapPlanStep", 0);
        }

        int step = bb.GetInt("goapPlanStep");
        if (step >= (int)plan.size()) {
            plan.clear();
            return Status::Success;
        }

        const Action* action = plan[step];
        if (action->execute && action->execute()) {
            bb.SetInt("goapPlanStep", step + 1);
            return Status::Running;
        }

        // 执行失败，清除计划以便重规划
        plan.clear();
        return Status::Failure;
    }
};
```

## 8. 性能分析

### 8.1 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| 规划（最好） | O(A) | A为行动数，直接匹配目标 |
| 规划（平均） | O(A^D * log(A^D)) | D为最优计划深度 |
| 规划（最差） | O(2^S) | S为状态空间大小 |
| 状态比较 | O(1) | 位集操作 |
| 行动应用 | O(1) | 位集操作 |

### 8.2 内存分析

| 组件 | 大小 |
|------|------|
| WorldState | 16字节（两个uint64_t） |
| Action | 32字节 + 回调大小 |
| PlanNode | 40字节 |
| 搜索过程 | O(N) N为扩展的节点数 |

### 8.3 性能优化

1. **行动预排序**：按与目标的关联度排序行动，提高A*搜索效率
2. **启发函数优化**：使用更精确的启发函数（考虑行动效果与目标的重合度）
3. **缓存规划结果**：相同起始状态和目标的规划结果缓存
4. **限制搜索深度**：设置最大规划步数，防止搜索空间爆炸
5. **分层规划**：高层规划抽象行动，低层规划细化

## 9. 常见陷阱

1. **行动集合膨胀**：可用行动过多导致搜索空间爆炸。解决方案：预过滤不相关的行动
2. **预条件/效果不一致**：行动A的预条件检查"有武器=true"但效果不设"有武器=true"。解决方案：使用单元测试验证每个行动
3. **规划死循环**：行动A的效果抵消行动B的前置条件，反之亦然。解决方案：在状态中添加"已执行X"标记
4. **执行失败但世界状态未更新**：行动执行失败时必须回滚状态更新
5. **缺少有效启发函数**：h(n)=0退化为Dijkstra。解决方案：至少使用"未满足目标数"作为启发
6. **实时开销**：每帧都重新规划过于昂贵。解决方案：使用增量重规划+冷却期

## 10. 实际游戏案例

### 案例1：F.E.A.R.（2005）

F.E.A.R.是GOAP的开创性应用：
- 敌人有5种基本行动：射击、投掷手雷、躲避、追击、搜索
- 目标："消灭玩家"、"存活"、"发现玩家"
- 行动通过预条件/效果组合，自动生成掩护射击、包抄进攻等复杂战术
- 敌人AI"看起来聪明"是因为行动组合自然产生了灵活行为

### 案例2：中土世界：魔多之影

兽人AI使用GOAP生成个性化行为：
- 每个兽人有不同的行动集合（基于种族、等级、个性）
- 兽人之间的社交互动通过GOAP行动建模（挑衅、结盟、背叛）
- 玩家行为影响兽人的世界状态（如击败兽人使其产生"恐惧"属性）
- GOAP让兽人行为看起来"有机"而非脚本化

### 案例3：Tomb Raider（2013重启版）

敌人AI根据环境自适应战术：
- 行动包括：射击、包抄、使用掩体、投掷瓶罐、近战冲锋
- 环境因素影响行动选择（掩体位置、弹药分布）
- 敌人会根据玩家位置和行为动态调整战术
- GOAP确保敌人行为多样化且看起来有"战术意识"

### 案例4：XCOM 2

XCOM 2的AI Mod支持基于GOAP：
- Mod作者可以定义新的行动和效果
- 外星人AI的决策通过GOAP规划
- 不同敌人类型有不同的行动集合和目标优先级
- 开发者通过调整行动代价和优先级来微调AI难度
