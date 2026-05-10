# 代数规约 (Algebraic Specification)


## 1. 代数规约基本概念

**核心思想：**
不描述"怎么做"，而是通过操作之间的等式关系描述"做什么"。数据类型被看作代数结构（集合 + 操作 + 公理）。

#### 规约组成


- **签名 (Signature)**
   ：类型名 + 操作签名
- **公理 (Axioms)**
   ：操作间的等式关系
- **类型参数**
   ：泛型/多态支持
- **导入**
   ：复用已有规约


#### 关键特性


- 声明式（what, not how）
- 独立于实现
- 可通过等式推理验证性质
- 支持模块化与组合


### 签名 (Signature) 定义


$$
SIG Stack
            Sorts: Stack, Element, Bool
            Operations:
              empty : → Stack
              push : Element × Stack → Stack
              top : Stack → Element ∪ {error}
              pop : Stack → Stack
              isEmpty : Stack → Bool
$$


## 2. 初始代数语义 (Initial Algebra)

**核心思想：**
在所有满足公理的代数中，"初始代数"是最小的、无多余元素的代数。它将语法上不同的项映射为不同的值（除非公理要求它们相等）。

### 等式公理示例（栈 Stack）


$$
AXIOMS Stack
            ∀ e : Element, s : Stack
            (1) isEmpty(empty) = true
            (2) isEmpty(push(e, s)) = false
            (3) top(empty) = error
            (4) top(push(e, s)) = e
            (5) pop(empty) = empty
            (6) pop(push(e, s)) = s
$$


### 等式推理 (Equational Reasoning)


```
// 证明：push(top(s), pop(s)) = s  当 s 非空时

// 设 s = push(e, rest)，由公理：
push(top(push(e, rest)), pop(push(e, rest)))
    = push(e, rest)            // 公理(4): top(push(e,s)) = e
                                // 公理(6): pop(push(e,s)) = s
    = s                        // 定义

// 因此 push(top(s), pop(s)) = s 对所有非空 s 成立
```


### 初始代数 vs 其他语义


| 语义方式 | 核心特征 | 适用场景 |
| --- | --- | --- |
| 初始代数 (Initial) | 不同项 = 不同值（最小模型） | 规范正确性、等式推理 |
| 终代数 (Terminal/Final) | 观察等价的项 = 相同值 | 实现等价性 |
| 松散语义 (Loose) | 允许非初始的模型 | 约束更宽松的实现 |


## 3. OBJ 语言族


OBJ 是代数规约的可执行语言，支持模块化规约和等式求值。


```
*** OBJ3 栈规约 ***
obj STACK is
  sort Stack Element Bool .
  op empty : -> Stack .
  op push  : Element Stack -> Stack .
  op top   : Stack -> Element .
  op pop   : Stack -> Stack .
  op isEmpty : Stack -> Bool .

  var E : Element .
  var S : Stack .

  eq isEmpty(empty) = true .
  eq isEmpty(push(E, S)) = false .
  eq top(push(E, S)) = E .
  eq pop(push(E, S)) = S .
endo

*** 使用规约进行等式求值 ***
red top(push(3, push(2, empty))) .
*** 结果: 3 ***

red isEmpty(pop(push(3, empty))) .
*** 结果: true ***
```


### OBJ 模块系统


```
*** 模块继承：用栈实现队列 ***
obj QUEUE is
  protecting STACK .
  sort Queue .
  op newQ : -> Queue .
  op enqueue : Element Queue -> Queue .
  op dequeue : Queue -> Queue .
  op front : Queue -> Element .
  *** 用两个栈实现队列 ***
  *** 省略具体实现 ***
endo
```


## 4. CASL (Common Algebraic Specification Language)

**CASL**
是 ISO 标准化的代数规约语言（ISO/IEC 15148:2003），由 CoFI 项目开发。支持部分函数、子类型、多态等高级特性。

```
--- CASL 栈规约 ---
spec STACK = sort Stack, Element

    ops
        empty : Stack;
        push : Element * Stack -> Stack;
        top : Stack ->? Element;      --- 部分函数 (? 表示可能无定义)
        pop : Stack -> Stack;
        isEmpty : Stack -> Bool

    forall s : Stack; e : Element

    then
        top(empty) undefined;         --- top(empty) 无定义
        not isEmpty(push(e, s));      --- push 后不为空
        top(push(e, s)) = e;
        pop(push(e, s)) = s;
        isEmpty(empty)
end

--- 规约扩展 ---
spec STACK_WITH_COUNT = STACK then
    ops count : Stack -> Nat
    forall s : Stack; e : Element
    then
        count(empty) = 0;
        count(push(e, s)) = count(s) + 1
end
```


### CASL 特性对比


| 特性 | OBJ3 | CASL |
| --- | --- | --- |
| 部分函数 | 需手动处理 | 原生支持 (->?) |
| 子类型 | 不支持 | 原生支持 |
| 多态 | 有限支持 | 原生支持 |
| 可执行 | 是 | 部分可执行 |
| 标准化 | 否 | ISO 标准 |
| 工具支持 | OBJ3 解释器 | Hets, CASL Tools |


## 5. 应用与总结


#### 应用场景


- API 行为精确规约
- 编译器中抽象语法定义
- 数据库 schema 验证
- 协议行为的形式化描述


#### 优势与局限


- 优势：精确、无歧义、可推理
- 优势：模块化、可组合
- 局限：学习曲线陡峭
- 局限：大型系统规约复杂


<!-- Converted from: 02_Algebra规约.html -->
