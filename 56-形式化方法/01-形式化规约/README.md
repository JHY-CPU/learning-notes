# 01-形式化规约

## 1. 形式化方法概述与动机

形式化方法（Formal Methods）是基于数学的技术，用于软件系统和硬件系统的规约、开发和验证。

### 动机

- **消除歧义**：自然语言规约常有歧义，形式化语言精确无歧义
- **早期发现缺陷**：在编码前发现逻辑错误
- **自动化验证**：通过工具自动检查正确性
- **安全关键系统**：航空航天、医疗设备、核能控制等领域要求极高可靠性

### 分类

| 类别 | 方法 | 说明 |
|------|------|------|
| 形式化规约 | Z、VDM、B、Alloy | 用数学语言描述系统"做什么" |
| 模型检验 | SPIN、NuSMV、TLA+ | 自动验证有限状态系统 |
| 定理证明 | Coq、Isabelle、ACL2 | 交互式证明系统正确性 |
| 合约式设计 | Eiffel、JML、ACSL | 在代码中嵌入形式化约束 |

### 应用领域

- 安全关键系统（DO-178C、IEC 61508）
- 密码协议验证
- 并发和分布式系统验证
- 芯片设计验证

## 2. Z语言基础

Z语言（Z notation）由牛津大学Jean-Raymond Abrial于1970年代提出，是最早的形式化规约语言之一。

### 基本元素

#### 集合

```
[PERSON]          -- 给定集合声明
S : ℙ PERSON      -- S是PERSON的幂集（子集集合）
```

#### 关系

```
enrolled : STUDENT ↔ COURSE    -- enrolled是从STUDENT到COURSE的关系
```

#### 函数

```
score : STUDENT → ℕ            -- score是从STUDENT到自然数的函数
```

### 模式（Schema）

模式是Z语言的核心，用于描述系统的状态和操作。

#### 状态模式

```
┌─────────────────────────┐
│        PhoneBook         │
├─────────────────────────┤
│ names : ℙ NAME          │
│ numbers : ℙ NUMBER      │
│ book : NAME ⇸ NUMBER    │
├─────────────────────────┤
│ dom book ⊆ names        │
│ ran book ⊆ numbers      │
└─────────────────────────┘
```

#### 操作模式

```
┌─────────────────────────┐
│      AddEntry            │
├─────────────────────────┤
│ ΔPhoneBook              │
│ name? : NAME            │
│ number? : NUMBER        │
├─────────────────────────┤
│ name? ∉ names           │
│ names' = names ∪ {name?}│
│ book' = book ∪ {name? ↦ number?}│
└─────────────────────────┘
```

其中 `?` 表示输入，`'` 表示操作后的状态。

### Z语言的特点

- 基于集合论和一阶谓词逻辑
- 模式提供结构化的规约组织方式
- 增量式精化支持从抽象规约到具体实现

## 3. VDM（维也纳开发方法）

VDM（Vienna Development Method）由IBM维也纳实验室开发，是另一种经典的形式化规约方法。

### 核心概念

#### 基本类型

```
Token                        -- 未指定的原子类型
nat = {n: int | n >= 0}     -- 自然数
char                        -- 字符
```

#### 复合类型

```
-- 记录类型
Account :: owner : Token
           balance : nat
           status : [<Active> | <Closed>]

-- 可选类型
[opt]value : [nat]           -- 可选的自然数

-- 联合类型
Id = nat | string
```

#### 操作规约

```
withdraw: Account * nat ==> Account
withdraw(acc, amount) == 
    acc(balance) := acc(balance) - amount
pre acc(balance) >= amount ∧ acc(status) = <Active>
post acc'(balance) = acc(balance) - amount
```

### VDM-SL

VDM-SL（VDM Specification Language）是VDM的标准语言，已被ISO标准化（ISO/IEC 13817-1）。

### VDM的特点

- 强调操作规约（前置/后置条件）
- 支持数据精化和操作精化
- 工具支持较好（VDMTools、VDM VSCode插件）

## 4. B方法（精化与证明）

B方法由Z语言的发明者Jean-Raymond Abrial进一步发展，强调从规约到代码的完整精化链。

### 核心思想

```
抽象机（Abstract Machine）
    ↓ 精化（Refinement）
精化机（Refinement Machine）
    ↓ 精化
实现（Implementation）
```

### 抽象机示例

```
MACHINE Counter
VARIABLES count
INVARIANT count ∈ ℕ
INITIALISATION count := 0
OPERATIONS
  inc =
    BEGIN
      count := count + 1
    END;
  val = 
    BEGIN
      result := count
    END
END
```

### 精化过程

每一步精化需要证明：
- **可实施性**（Feasibility）：操作在状态下可执行
- **精化关系**（Refinement Relation）：精化后的状态与抽象状态一致
- **不变量保持**（Invariant Preservation）：操作后不变量仍然成立

### Event-B

Event-B是B方法的现代扩展，使用事件驱动的方式建模。

```
MACHINE TrafficLight
VARIABLES light
INVARIANT light ∈ {<RED>, <YELLOW>, <GREEN>}
EVENTS
  Initialisation: light := <RED>
  Go: WHEN light = <RED> THEN light := <GREEN> END
  Warn: WHEN light = <GREEN> THEN light := <YELLOW> END
  Stop: WHEN light = <YELLOW> THEN light := <RED> END
END
```

### B方法的工业应用

- 巴黎地铁14号线信号系统
- 阿丽亚娜5号火箭控制软件
- 智能卡操作系统

## 5. Alloy语言基础

Alloy由MIT的Daniel Jackson开发，基于关系逻辑，适合建模和分析软件结构。

### 核心概念

```alloy
-- 签名（类）
sig Person {
    friends: set Person
}

-- 事实（约束）
fact {
    no p: Person | p in p.friends        -- 不能是自己的朋友
    friends = ~friends                    -- 朋友关系对称
}

-- 断言（需要验证的性质）
assert NoSelfLoop {
    no p: Person | p in p.^friends       -- 不能通过朋友链到达自己
}

-- 谓词（可执行的规约）
pred show() {
    #Person = 5
    some p: Person | #p.friends > 2
}

-- 运行谓词，生成满足条件的实例
run show for 6
```

### 关系运算符

| 运算符 | 含义 |
|--------|------|
| `.` | 关系连接（类似函数组合） |
| `&` | 交集 |
| `+` | 并集 |
| `-` | 差集 |
| `->` | 笛卡尔积 |
| `~` | 转置 |
| `*` | 传递闭包 |
| `^` | 自反传递闭包 |

### Alloy Analyzer

Alloy的核心工具是Alloy Analyzer，提供：
- **模型实例生成**：生成满足约束的具体实例
- **反例查找**：验证断言时自动寻找反例
- **可视化**：图形化展示生成的实例

### 适用场景

- 软件架构设计验证
- 安全策略分析
- 数据模型验证
- 协议设计验证

## 6. OCL（对象约束语言）

OCL（Object Constraint Language）是UML的组成部分，用于对UML模型添加精确约束。

### 基本结构

```
context Person::adult() : Boolean
    -- 前置条件
    pre: self.age.isDefined()
    -- 后置条件
    post: result = (self.age >= 18)
```

### 常用表达式

#### 不变量（Invariant）

```ocl
context Person
    inv: self.age >= 0
    inv: self.name.size() > 0
```

#### 集合操作

```ocl
-- 选择
self.employees->select(e | e.salary > 5000)

-- 映射
self.orders->collect(o | o.totalAmount)

-- 存在性
self.employees->exists(e | e.manager = true)

-- 全称
self.employees->forAll(e | e.salary > 0)

-- 排序
self.orders->sortedBy(o | o.date)
```

#### 导航

```ocl
-- 通过关联导航
self.department.company.ceo

-- 遍历关联
self.manages.employees
```

### OCL的限制

- 只能表达约束，不能表达计算
- 不是可执行语言（不能直接运行）
- 依赖于UML模型的存在
