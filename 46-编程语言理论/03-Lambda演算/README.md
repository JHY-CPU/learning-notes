# 03-Lambda演算

Lambda演算（λ-Calculus）是由Alonzo Church在1930年代提出的**形式系统**，用于研究函数定义、函数应用和递归。它是所有函数式编程语言的理论基础，也是计算理论的核心模型之一。

## λ演算的语法

Lambda演算的语法极其精简，仅包含三个构造：

```
e ::= x | λx.e | e1 e2
```

- **变量**（Variable）：如 x, y, z，标识一个符号
- **抽象**（Abstraction）：`λx.e` 定义一个以x为参数、e为函数体的匿名函数
- **应用**（Application）：`e1 e2` 将函数e1作用于参数e2

### 约定与简化

- 函数应用**左结合**：`f g h` 等价于 `((f g) h)`
- 抽象的**体向右延伸尽可能远**：`λx. λy. x y` 等价于 `λx. (λy. (x y))`
- 可省略多余括号：`λx. (λy. x)` 写作 `λx. λy. x`

### 自由变量与约束变量

在 `λx.e` 中，x是**约束变量**（Bound Variable），e中未被λ绑定的变量是**自由变量**（Free Variable）。**封闭项**（Closed Term）是没有自由变量的λ表达式，如 `λx. x`（恒等函数）。

## α转换、β归约、η转换

### α转换（Alpha Conversion）

重命名约束变量，不改变函数含义。类比数学中 ∫f(x)dx = ∫f(y)dy。

```
λx. x  ≡α  λy. y           -- 恒等函数，参数名无关紧要
λx. λx. x  →α  λx. λy. x  -- 避免变量名冲突
```

形式化：`λx.e ≡α λy.e[y/x]`，其中y不在e中出现且不被其他λ捕获。

### β归约（Beta Reduction）

β归约是λ演算的核心计算规则，定义了函数应用的求值：

```
(λx.e1) e2  →β  e1[x := e2]
```

即：将函数体e1中所有自由出现的x替换为e2。

**示例：**

```
(λx. x + 1) 3  →β  3 + 1  =  4

(λf. λx. f (f x)) (λy. y + 1)
→β λx. (λy. y + 1) ((λy. y + 1) x)
→β λx. (λy. y + 1) (x + 1)
→β λx. (x + 1) + 1  =  λx. x + 2
```

**替换的捕获避免：** 替换 `e1[x := e2]` 时，必须确保e2中的自由变量不被e1中的λ意外捕获。

```
-- 错误：(λx. λy. x) y  →  λy. y    （y被外层λy捕获了）
-- 正确：(λx. λy. x) y  →α (λx. λz. x) y  →β  λz. y
```

### η转换（Eta Conversion）

η转换刻画了"外延等价"——如果两个函数对所有输入产生相同输出，则它们相等。

```
λx. f x  ≡η  f      （当x不在f中自由出现时）
```

## Church编码

Church编码展示了如何用纯λ表达式编码各种数据结构和运算，证明了λ演算的图灵完备性。

### Church布尔值

```
TRUE  = λt. λf. t      -- 选择第一个参数
FALSE = λt. λf. f      -- 选择第二个参数

AND = λp. λq. p q p
OR  = λp. λq. p p q
NOT = λp. p FALSE TRUE
```

### Church自然数（Church数）

Church数用"应用次数"表示自然数：

```
ZERO  = λf. λx. x              -- f应用0次
ONE   = λf. λx. f x            -- f应用1次
TWO   = λf. λx. f (f x)        -- f应用2次
THREE = λf. λx. f (f (f x))    -- f应用3次

SUCC    = λn. λf. λx. f (n f x)
PLUS    = λm. λn. λf. λx. m f (n f x)
MULT    = λm. λn. λf. m (n f)
ISZERO  = λn. n (λx. FALSE) TRUE
```

### Church对（有序对）

```
PAIR  = λa. λb. λf. f a b
FST   = λp. p TRUE
SND   = λp. p FALSE
```

### Church递归（Y组合子）

λ演算中没有内置递归，需用不动点组合子实现：`Y = λf. (λx. f (x x)) (λx. f (x x))`，满足 `Y f = f (Y f)`。使用Y组合子可定义阶乘：`FACT = Y (λf. λn. ISZERO n ONE (MULT n (f (PRED n))))`。

## Church-Rosser定理（合流性）

Church-Rosser定理是λ演算的基石性质：

**定理：** 若 e1 ← e → e2，则存在 e3 使得 e1 → e3 且 e2 → e3。

```
        e
       / \
    e1    e2
      \   /
       e3
```

**推论：**
- 如果 e →β v1 且 e →β v2，且v1和v2都是正规形式，则 v1 = v2（经α等价）
- 每个λ表达式至多有一个正规形式

**意义：** 合流性保证了求值顺序不影响最终结果（只要结果存在）。这为惰性求值和并行求值提供了理论基础。

**注意：** 并非所有λ表达式都有正规形式。`(λx. x x) (λx. x x)` 会无限归约（Ω发散）。

## 归约策略

### 正规序（Normal Order）

总是选择**最左边最外层**的可归约表达式（redex）。参数延迟求值。

```
(λx. λy. x) ((λz. z z) (λz. z z))
→ λy. (λz. z z) (λz. z z)    -- 外层redex被归约，内层发散项未被求值
```

正规序保证：若表达式有正规形式，正规序一定能找到它。

### 应用序（Applicative Order）

总是选择**最左边最内层**的redex。参数先求值。

```
(λx. λy. x) ((λz. z z) (λz. z z))
→ (λx. λy. x) (...)  -- 内层归约，陷入无限循环
```

### CBV与CBN

- **CBV**（Call by Value）：函数调用时先求值参数到值，对应应用序的受限版本
- **CBN**（Call by Name）：参数在需要时才求值，对应正规序的受限版本

| 策略 | 求值顺序 | 参数求值 | 代表语言 |
|------|---------|---------|---------|
| 正规序 | 最外层 | 不提前 | Haskell（惰性） |
| 应用序 | 最内层 | 提前 | Scheme, ML |
| CBV | 最内层受限 | 调用时 | Java, Python |
| CBN | 最外层受限 | 使用时 | Algol（历史） |

## 有类型λ演算

### 简单类型λ演算（STLC）

在无类型λ演算基础上添加类型标注，消除无限归约的可能性。

**类型语法：** `τ ::= τ → τ | Bool | Nat | ...`

**类型判断规则：**

```
Γ(x) = τ                        Γ, x:τ1 ⊢ e : τ2
──────── (Var)                   ────────────────── (Abs)
Γ ⊢ x : τ                       Γ ⊢ λx:τ1. e : τ1 → τ2

Γ ⊢ e1 : τ1 → τ2    Γ ⊢ e2 : τ1
───────────────────────────────── (App)
Γ ⊢ e1 e2 : τ2
```

STLC满足**强规范化**（Strong Normalization）：所有良类型表达式都有正规形式。这意味着STLC不是图灵完备的——这是类型安全的代价。

### System F（多态λ演算）

System F引入了**类型抽象**和**类型应用**，支持参数多态：

```
e ::= x | λx:τ.e | e1 e2 | Λα.e | e [τ]
τ ::= α | τ → τ | ∀α.τ
```

示例：多态恒等函数 `id = Λα. λx:α. x : ∀α. α → α`，使用时 `id [Nat] 5 →β 5`。

System F是Haskell和Rust中泛型的理论基础。

## Currying与Uncurrying

**Currying**将多参数函数转换为一系列单参数函数链：

```
-- 未Curry化：add : (Nat, Nat) → Nat
-- Curry化：  add : Nat → (Nat → Nat)
add = λm. λn. m + n
-- 调用：add 3 5 等价于 (add 3) 5
```

**数学基础：** 在范畴论中，Currying对应于笛卡尔闭范畴中的伴随关系：

```
Hom(A × B, C)  ≅  Hom(A, Hom(B, C))
```

所有函数式语言都天然支持Currying（Haskell中所有函数本质上都是单参数的）。

## 与函数式编程的关系

| λ演算 | Haskell | Scheme | ML |
|--------|---------|--------|-----|
| `λx.e` | `\x -> e` | `(lambda (x) e)` | `fn x => e` |
| `e1 e2` | `e1 e2` | `(e1 e2)` | `e1 e2` |
| Church编码 | 代数数据类型 | cons对 | datatype |
| Y组合子 | `fix` | 递归定义 | `let rec` |

**关键洞察：** λ演算证明了函数抽象和应用足以表达所有计算。现代函数式语言在此基础上添加了类型系统、模式匹配、惰性求值等便利特性，但核心计算模型始终是λ演算。

## 延伸阅读

- Hendrik Pieter Barendregt, *The Lambda Calculus: Its Syntax and Semantics*
- Benjamin C. Pierce, *Types and Programming Languages*, Chapters 5, 9, 11, 23
- Simon Peyton Jones, *The Implementation of Functional Programming Languages*, Chapter 2
- J. Roger Hindley, *Basic Simple Type Theory*
