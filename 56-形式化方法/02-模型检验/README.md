# 02-模型检验

## 1. 模型检验的原理

模型检验（Model Checking）是一种自动化的形式化验证方法，通过穷举搜索系统的所有可能状态来验证系统是否满足给定的规约。

### 基本流程

```
系统模型 M + 时序逻辑规约 φ
         ↓
   模型检验算法
         ↓
   ┌─────┴─────┐
   ↓           ↓
  满足       不满足
(系统正确)  (提供反例路径)
```

### 核心思想

- 将系统建模为有限状态转换系统
- 用时序逻辑（LTL/CTL）表达系统应满足的性质
- 穷举搜索状态空间，验证性质是否在所有可达状态上成立

### 优势

- **完全自动化**：无需人工干预
- **反例生成**：验证失败时提供具体的错误路径
- **无需测试用例**：覆盖所有可能的执行路径

### 适用领域

- 硬件电路验证
- 通信协议验证
- 并发系统验证
- 安全协议分析

## 2. 状态空间爆炸问题

模型检验的最大挑战是状态空间爆炸：系统的状态数随组件数呈指数增长。

### 问题示例

```
N个并发进程，每个有k个状态 → 总状态数 = k^N

例如：20个进程，每个10个状态 → 10^20 个状态
```

### 缓解技术

| 技术 | 原理 |
|------|------|
| **符号模型检验** | 用BDD（二元决策图）表示状态集合，避免逐个枚举 |
| **偏序归约** | 利用并发操作的独立性减少需要探索的状态 |
| **对称性归约** | 利用系统的对称结构合并等价状态 |
| **抽象** | 用更抽象的模型替代具体模型，减少状态数 |
| **有界模型检验** | 限制搜索深度，将问题转化为SAT求解 |
| **on-the-fly检验** | 按需生成状态，而非预先展开全部状态 |

## 3. 时序逻辑（LTL、CTL）

时序逻辑是表达系统随时间变化的行为性质的逻辑语言。

### 线性时序逻辑（LTL）

LTL将时间视为线性路径，表达路径上的性质。

#### 基本算子

| 算子 | 含义 | 示例 |
|------|------|------|
| **X p** | 下一个时刻p成立 | X(request → X grant) |
| **F p** | 将来某个时刻p成立 | F (response) |
| **G p** | 所有时刻p都成立 | G (no_crash) |
| **p U q** | p一直成立直到q成立 | (waiting) U (served) |
| **p R q** | q一直成立直到p成立（或永远） | (safe) R (no_error) |

#### 经典性质

```
-- 安全性：坏的事情永远不会发生
G ¬(deadlock)

-- 活性：好的事情最终会发生
G (request → F response)

-- 公平性：无限次执行某个动作
G F (process_runs)
```

### 计算树逻辑（CTL）

CTL将时间视为分支树，每个状态可能有多个后继。

#### 路径量词

- **A**：所有路径上（For All paths）
- **E**：存在一条路径上（Exists a path）

#### 状态算子

与LTL相同：X、F、G、U、R

#### CTL公式必须交替使用量词和算子

```
-- 安全性：所有路径上永远没有死锁
AG ¬deadlock

-- 可达性：存在一条路径最终到达目标状态
EF goal_reached

-- 活性：所有路径上，请求最终会被响应
AG (request → AF response)

-- 可能性：存在一条路径，在某个时刻满足条件
EF (error_state)
```

### LTL vs CTL

| 维度 | LTL | CTL |
|------|-----|-----|
| 时间观 | 线性（路径） | 分支（树） |
| 量词 | 无显式量词 | A（所有）、E（存在） |
| 表达能力 | 有些CTL性质无法表达 | 有些LTL性质无法表达 |
| 两者不可比较 | 存在对方无法表达的性质 |

### CTL* 

CTL*是LTL和CTL的超集，允许自由混合路径量词和时序算子。

## 4. 模型检验工具

### SPIN

SPIN由贝尔实验室开发，专为并发系统和通信协议的验证。

#### 特点

- 建模语言：Promela（Process Meta Language）
- 验证性质：LTL公式
- 技术：on-the-fly检验、偏序归约

#### Promela示例

```promela
mtype = { req, ack };
chan channel = [1] of { mtype };

active proctype Sender() {
    channel!req;
    channel?ack;
}

active proctype Receiver() {
    channel?req;
    channel!ack;
}

ltl response { [](req -> <>(ack)) }
```

### NuSMV

NuSMV是SMV的重新实现，支持符号模型检验。

#### 特点

- 建模语言：SMV语言
- 验证性质：CTL和LTL
- 技术：BDD-based符号模型检验、有界模型检验（BMC）

#### SMV示例

```smv
MODULE main
VAR
    state : { idle, requesting, active, releasing };
    request : boolean;
    grant : boolean;

ASSIGN
    init(state) := idle;
    next(state) := case
        state = idle & request : requesting;
        state = requesting & grant : active;
        state = active & !request : releasing;
        state = releasing : idle;
        TRUE : state;
    esac;

SPEC AG (request -> AF (state = active))
```

### TLA+

TLA+由Leslie Lamport开发，适合并发和分布式系统的规约和验证。

#### 特点

- 基于集合论和时序逻辑
- 模型检验工具：TLC
- 工业应用：Amazon Web Services、微软

#### TLA+示例

```tla
VARIABLE pc, x

Init == pc = "start" /\ x = 0

Step ==
    \/ pc = "start" /\ x' = x + 1 /\ pc' = "done"
    \/ pc = "done" /\ UNCHANGED <<pc, x>>

Spec == Init /\ [][Step]_<<pc, x>>

Safety == pc = "done" => x = 1
```

### 工具对比

| 工具 | 建模语言 | 支持逻辑 | 主要技术 | 适用场景 |
|------|---------|---------|---------|---------|
| SPIN | Promela | LTL | on-the-fly | 并发、协议 |
| NuSMV | SMV | CTL、LTL | BMC、BDD | 硬件、控制系统 |
| TLA+ | TLA+ | TLA | 状态枚举 | 分布式系统 |
| UPPAAL | Timed Automata | TCTL | 符号执行 | 实时系统 |
| PRISM | 概率模型 | PCTL | 概率验证 | 随机系统 |

## 5. 符号模型检验与有界模型检验

### 符号模型检验（Symbolic Model Checking）

不用逐个枚举状态，而是用符号（通常是BDD）表示状态集合。

#### BDD（Binary Decision Diagram）

- 用有向无环图表示布尔函数
- 支持高效的集合运算（交、并、补、前像、后像）
- 避免显式存储所有状态

#### 流程

```
初始状态集合 S₀（用BDD表示）
    ↓
计算后继：S₁ = Post(S₀)
    ↓
合并：S₀ ∪ S₁
    ↓
重复直到不动点或发现违反性质
```

### 有界模型检验（BMC）

将模型检验问题转化为布尔可满足性（SAT）问题。

#### 核心思想

```
问题：是否存在长度为k的执行路径，使得系统违反性质φ？

转化为：找一个SAT公式 F(k) 的解
F(k) = Init(s₀) ∧ T(s₀,s₁) ∧ ... ∧ T(sₖ₋₁,sₖ) ∧ ¬φ(sₖ)
```

#### 优势

- SAT求解器极其高效
- 特别擅长寻找反例（短路径）
- 适用于大规模系统

#### 局限性

- 只能检查有界长度
- 无法证明性质（只能证伪）
- 无法保证系统不存在更长的违反路径

## 6. 抽象与精化

### 抽象（Abstraction）

用更简单的模型替代具体的系统模型，减少状态空间。

#### 常用抽象技术

- **数据抽象**：将具体值映射到抽象域（如将具体数字映射为{zero, positive, negative}）
- **谓词抽象**：用谓词集合表示状态
- **行为抽象**：忽略不相关的操作

#### 抽象的正确性

抽象模型必须是具体模型的**过近似（Over-approximation）**：
- 抽象模型包含具体模型的所有行为
- 如果性质在抽象模型上成立，则在具体模型上也成立
- 如果性质在抽象模型上不成立，可能需要精化

### 精化（Refinement）

当抽象模型过于粗糙导致假阳性（spurious counterexample）时，需要精化。

#### CEGAR（Counterexample-Guided Abstraction Refinement）

```
1. 构造抽象模型
2. 在抽象模型上验证性质
3. 如果满足 → 性质在具体模型上成立，结束
4. 如果不满足 → 检查反例是否在具体模型中可行
   a. 如果可行 → 真正的反例，结束
   b. 如果不可行（假反例） → 精化抽象模型，回到步骤2
```

#### CEGAR流程图

```
抽象模型 → 模型检验 → 反例
                         ↓
                    具体化验证
                    ↓        ↓
               可行        不可行
                ↓            ↓
            报告错误     精化抽象模型 → 重新验证
```

### 实践建议

- 从最粗糙的抽象开始
- 自动化精化过程（CEGAR）
- 结合领域知识选择抽象域
- 对关键组件使用更精确的抽象
