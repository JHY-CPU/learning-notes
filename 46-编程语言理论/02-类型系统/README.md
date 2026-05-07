# 02-类型系统

类型系统是编程语言中用于对程序值和表达式进行分类的规则集合，旨在通过分类来保证程序的正确性和安全性。

## 类型的概念与作用

**类型**（Type）是一组值的集合以及可在这些值上执行的操作的总体。类型系统的核心作用包括：

1. **错误检测**：在编译期或运行时捕获类型不匹配的错误
2. **抽象**：通过类型接口隐藏实现细节
3. **文档化**：类型签名即文档，说明函数的输入输出
4. **优化**：编译器可根据类型信息生成更高效的代码
5. **安全性**：防止非法操作（如对字符串执行算术运算）

类型判断的形式化表达：`Γ ⊢ e : τ`，表示在类型环境Γ下，表达式e具有类型τ。

## 静态类型 vs 动态类型

### 静态类型（Static Typing）

在编译时确定所有表达式的类型。变量一旦声明，类型不可更改。

**优势：** 编译时捕获类型错误、编译器可优化、IDE支持好、类型即文档。
**代表语言：** C, C++, Java, Haskell, Rust, TypeScript, Go

### 动态类型（Dynamic Typing）

类型在运行时与值关联，变量可以持有任意类型的值。

**优势：** 编写灵活、适合快速原型、鸭子类型（Duck Typing）提供多态性。
**代表语言：** Python, JavaScript, Ruby, Lisp, PHP

```python
# Python - 动态类型
x = 42
x = "hello"  # 合法，x的类型在运行时改变
```

### 混合策略

现代语言常采用混合方式：Python（mypy）、JavaScript（TypeScript）、C++（auto）、Rust/Kotlin（局部类型推导）。

## 强类型 vs 弱类型

### 强类型（Strong Typing）

语言不允许或极少进行隐式类型转换，类型之间有明确的边界。

```python
# Python - 强类型
"hello" + 42  # TypeError: can only concatenate str to str
```

### 弱类型（Weak Typing）

语言允许广泛的隐式类型转换，不同类型的值可混用。

```javascript
// JavaScript - 弱类型
"hello" + 42    // "hello42" （数字隐式转为字符串）
"3" - 1         // 2 （字符串隐式转为数字）
[] + []         // "" （空数组转为空字符串）
```

**注意：** 强/弱类型是一个连续光谱而非二元对立。C语言在某些方面也表现出弱类型特征（如void指针的隐式转换）。

## 类型推导

类型推导允许编译器自动推断表达式的类型，无需程序员显式标注。

### Hindley-Milner类型系统

HM类型系统是多态类型推导的经典框架，广泛应用于ML家族语言（Standard ML, OCaml, Haskell）。

**核心思想：**
- 每个表达式有一个最一般（most general）的类型
- 通过**合一算法**（Unification）求解类型约束
- 支持let多态（let-polymorphism）

**类型推导规则示例：**

```
Γ ⊢ e1 : τ1 → τ2    Γ ⊢ e2 : τ1
──────────────────────────────── (App)
Γ ⊢ e1 e2 : τ2

Γ, x:τ1 ⊢ e : τ2
────────────────── (Abs)
Γ ⊢ λx.e : τ1 → τ2
```

### W算法

W算法是HM类型系统中实现类型推导的具体算法，输入类型环境和表达式，输出最一般类型替换和类型。

```haskell
W :: TypeEnv -> Expr -> Maybe (Subst, Type)
W Γ (Var x)     = instantiate(Γ(x))
W Γ (App e1 e2) = do
    (s1, τ1) <- W Γ e1
    (s2, τ2) <- W (s1·Γ) e2
    v <- fresh;  u <- unify(s2(τ1), τ2 → v)
    return (u·s2·s1, u(v))
W Γ (Lam x e)   = do
    v <- fresh
    (s, τ) <- W (Γ, x:v) e
    return (s, s(v) → τ)
```

**时间复杂度：** W算法最坏情况O(2^n)，但实践中几乎总是线性。

## 多态

多态（Polymorphism）指同一段代码可操作多种类型的能力。

### 参数多态（Parametric Polymorphism）

通过类型变量实现，又称"泛型"。

```haskell
length :: [a] -> Int       -- 对任意类型a的列表
map :: (a -> b) -> [a] -> [b]
```

### 特设多态（Ad-hoc Polymorphism）

同一函数名对不同类型有不同实现，又称"重载"。

```cpp
// C++ 函数重载
int add(int a, int b) { return a + b; }
double add(double a, double b) { return a + b; }
```

Haskell通过**类型类**（Type Classes）实现更优雅的特设多态。

### 子类型多态（Subtype Polymorphism）

子类型的对象可在期望父类型的上下文中使用，面向对象语言的核心特性。

```java
Animal a = new Dog();   // Dog是Animal的子类型
a.speak();              // 动态分派，调用Dog的speak
```

## 类型类（Type Classes）

类型类由Wadler和Blott在1989年提出，是Haskell中特设多态的机制。

```haskell
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool

instance Eq Int where
    x == y = primEqualInt x y

elem :: (Eq a) => a -> [a] -> Bool
elem x (y:ys) = x == y || elem x ys
```

类型类的优点：开放性（可随时为已有类型添加实例）、类型安全、无运行时开销。

## 代数数据类型（ADT）

代数数据类型通过"和"与"积"两种基本组合方式构造复杂数据类型。

### 积类型（Product Types）

积类型的值需要**同时**包含所有组成部分，对应逻辑的"与"（AND）。

```haskell
type Pair = (Int, String)           -- 值域大小：|Int| × |String|
data Person = Person { name :: String, age :: Int }
```

### 和类型（Sum Types）

和类型的值是**其中一个**分支的值，对应逻辑的"或"（OR）。

```haskell
data Shape = Circle Double | Rectangle Double Double   -- 值域：|Circle| + |Rectangle|
data Maybe a = Nothing | Just a
data Either a b = Left a | Right b
```

### 模式匹配

代数数据类型配合模式匹配提供了类型安全的解构方式：

```haskell
area :: Shape -> Double
area (Circle r)      = pi * r * r
area (Rectangle w h) = w * h
```

## 依赖类型基础

**依赖类型**（Dependent Type）允许类型依赖于值，将类型系统提升到更强的表达能力。

```idris
-- Idris中的向量，长度是类型的一部分
data Vect : Nat -> Type -> Type where
    Nil  : Vect 0 a
    (::) : a -> Vect n a -> Vect (S n) a

-- 类型保证：append的结果长度是两个输入长度之和
append : Vect n a -> Vect m a -> Vect (n + m) a
```

依赖类型可表达极其精确的规范（如排序算法返回有序列表），是形式化验证的强大工具。代表语言：Idris, Agda, Coq, Lean。

## 线性类型与仿射类型

**线性类型**要求每个值**恰好使用一次**；**仿射类型**允许每个值**最多使用一次**（可以丢弃）。Rust的所有权系统是仿射类型，Haskell从GHC 9.0起通过`LinearTypes`扩展支持线性类型。

## 类型安全与类型擦除

### 类型安全（Type Safety）

类型安全保证程序不会在运行时出现"未定义行为"。经典的类型安全定理：

- **Progress**：类型良好的表达式要么是值，要么可以继续求值
- **Preservation**：求值步骤保持类型不变

### 类型擦除（Type Erasure）

某些语言在编译后擦除类型信息，类型仅在编译时发挥作用。

```java
// Java泛型擦除
List<String> strings = new ArrayList<>();
List<Integer> ints = new ArrayList<>();
// 运行时 strings.getClass() == ints.getClass() 为 true
```

C++的模板实例化和Haskell的类型类在编译时生成特化代码，不存在运行时类型信息。

## 延伸阅读

- Benjamin C. Pierce, *Types and Programming Languages* (TAPL)
- Robert Harper, *Practical Foundations for Programming Languages*, Chapters 4-11
- Luca Cardelli, "Type Systems", *ACM Computing Surveys*, 1996
- Simon Peyton Jones, *The Implementation of Functional Programming Languages*, Chapter 7
