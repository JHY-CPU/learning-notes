# 04-编程范式

编程范式（Programming Paradigm）是一种组织和构造程序的思维方式与方法论。不同的范式代表了对计算的不同理解角度。

## 命令式编程

命令式编程将计算描述为一系列改变程序状态的指令，核心概念是**可变状态**和**赋值**。

### 过程式编程

将程序组织为过程（函数/子程序）的序列，通过顺序、分支、循环控制流程。

```c
// C语言过程式编程 - 冒泡排序
void sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
}
```

**特征：** 自顶向下设计、全局/局部变量、副作用驱动。Dijkstra的结构化编程原则强调顺序、选择、迭代三种控制结构，禁止goto。

## 函数式编程

函数式编程将计算视为**函数的求值**，强调不可变数据和纯函数。

### 纯函数与高阶函数

纯函数满足：相同输入必然产生相同输出，且无副作用。高阶函数可作为参数和返回值传递。

```haskell
-- 纯函数与高阶函数
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs

filter :: (a -> Bool) -> [a] -> [a]
filter p (x:xs) | p x       = x : filter p xs
                | otherwise  = filter p xs

(.) :: (b -> c) -> (a -> b) -> a -> c
(f . g) x = f (g x)
```

### 不可变性与模式匹配

不可变数据结构避免了共享可变状态带来的复杂性。模式匹配提供声明式的值解构方式。

```haskell
-- 不可变：所有"修改"操作都返回新值
addToList :: a -> [a] -> [a]
addToList x xs = x : xs

-- 模式匹配
data Tree a = Leaf a | Node (Tree a) (Tree a)

depth :: Tree a -> Int
depth (Leaf _)   = 1
depth (Node l r) = 1 + max (depth l) (depth r)
```

**不可变性的工程优势：** 线程安全、可预测性、可缓存性、可回溯性。

## 逻辑式编程

逻辑式编程将计算视为在关系上的**逻辑推导**。程序员声明"什么是真的"，系统自动推导如何求解。

### Prolog基础与归结原理

Prolog（Programming in Logic）是最著名的逻辑式语言：

```prolog
% 事实（Facts）与规则（Rules）
parent(tom, bob).  parent(tom, liz).  parent(bob, ann).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% 查询：?- grandparent(tom, Who).  % Who = ann
```

归结原理是推理引擎核心：通过**合一**（Unification）匹配目标和子句头部，通过**回溯**（Backtracking）搜索所有可能的解。

**局限：** 效率问题（搜索空间大）、非确定性、算术表达不便。

## 面向对象核心概念详解

### 封装、继承与多态

```java
// 封装：隐藏实现细节
public class BankAccount {
    private double balance;
    public void deposit(double amount) { if (amount > 0) balance += amount; }
    public double getBalance() { return balance; }
}

// 继承：子类复用父类
class Dog extends Animal {
    @Override void speak() { System.out.println("Woof!"); }
}

// 多态：动态分派
Animal a = new Dog();
a.speak();  // 调用Dog的speak
```

### 消息传递

面向对象的核心隐喻：对象通过发送消息（方法调用）交互。Alan Kay最初的OOP概念强调的是消息传递和动态绑定，而非类和继承。

```smalltalk
"Smalltalk - 纯消息传递风格"
animal speak.    "向animal对象发送speak消息"
```

## 委托与原型链

某些语言不使用类继承，而是通过**委托**（Delegation）实现复用：对象将未处理的消息转发给其原型。

```javascript
// JavaScript原型链
const animal = { speak() { console.log(this.sound); } };
const dog = Object.create(animal);  // dog的原型是animal
dog.sound = "Woof!";
dog.speak();  // dog自身没有speak方法，委托给animal
```

**与类继承的区别：** 类继承在创建时复制，原型委托在运行时委托。原型链的动态性更强，可在运行时修改。

## 并发编程模型

### 共享内存模型

多个线程访问同一块内存区域，通过锁、信号量等机制同步。问题：死锁、竞态条件、难以推理。

```java
synchronized (lock) { counter++; }  // 临界区
```

### 消息传递与CSP

C.A.R. Hoare提出的CSP模型将并发视为独立进程通过通道通信：

```go
// Go基于CSP
func producer(ch chan<- int) { for i := 0; i < 10; i++ { ch <- i }; close(ch) }
func consumer(ch <-chan int) { for v := range ch { fmt.Println(v) } }
```

### Actor模型

Actor是独立的计算单元，通过异步消息通信，每个Actor有自己的邮箱和私有状态。关键特征：无共享状态、异步消息传递、串行处理消息、天然容错（"let it crash"）。代表实现：Erlang/OTP、Scala/Akka。

## 函数式响应式编程（FRP）

FRP将时间变化的值（信号/行为）和离散事件流作为一等公民。

```haskell
-- 原始FRP（Conal Elliott）
type Behavior a = Time -> a
type Event a = [(Time, a)]
```

现代FRP实现（基于流）：

```javascript
// RxJS - 响应式流
const positions$ = fromEvent(button, 'click').pipe(
    map(e => ({ x: e.clientX, y: e.clientY })),
    debounceTime(300), distinctUntilChanged()
);
```

**应用场景：** GUI事件处理、实时数据流、动画系统、异步编程。

## 元编程与宏系统

元编程（Metaprogramming）是"编写操作程序的程序"。

### Lisp卫生宏

```scheme
(define-syntax when
  (syntax-rules ()
    ((when test body ...)
     (if test (begin body ...)))))
```

### Rust过程式宏

```rust
#[derive(Debug, Clone, Serialize)]  // 派生宏
struct Point { x: f64, y: f64; }
```

### C++模板元编程

```cpp
template<int N> struct Factorial {
    static const int value = N * Factorial<N-1>::value;
};
template<> struct Factorial<0> { static const int value = 1; };
static_assert(Factorial<5>::value == 120, "");
```

## 领域特定语言（DSL）设计

DSL是为特定问题领域设计的专用语言，分为两类：

**外部DSL：** 独立于宿主语言，需自己的解析器。如SQL、正则表达式。

**内部DSL（嵌入式DSL）：** 利用宿主语言的语法构造。如Ruby on Rails路由DSL、Haskell Parsec组合子库。

**设计原则：** 嵌入式优先（复用工具链）、组合性（小块可组合）、领域映射（语法直觉）、领域友好的错误报告。

## 各范式的比较与选择

| 范式 | 核心抽象 | 优势 | 典型场景 | 代表语言 |
|------|---------|------|---------|---------|
| 命令式 | 赋值与状态 | 直接映射硬件、性能可控 | 系统编程、嵌入式 | C, Assembly |
| 面向对象 | 对象与消息 | 模块化、复用性、可维护 | 大型应用、GUI | Java, C# |
| 函数式 | 函数与不可变 | 并发安全、可组合、可验证 | 数据处理、编译器 | Haskell, OCaml |
| 逻辑式 | 规则与推导 | 声明式问题求解 | AI、专家系统 | Prolog, Datalog |
| 响应式 | 信号与事件流 | 异步处理、实时响应 | GUI、数据流 | RxJS, Elm |

### 混合范式趋势

现代语言多为多范式：Scala（OO+FP）、Rust（命令式+FP+系统级）、Python（命令式+OO+FP）、Swift（OO+FP+协议导向）。

**选择建议：** 数据变换适合函数式，状态密集型适合面向对象；考虑团队经验和生态工具；同一项目中不同层次可采用不同范式。

## 延伸阅读

- Harold Abelson & Gerald Jay Sussman, *Structure and Interpretation of Computer Programs* (SICP)
- Peter Van Roy & Seif Haridi, *Concepts, Techniques, and Models of Computer Programming*
- Michael Scott, *Programming Language Pragmatics*, Chapters 1, 3, 9, 11
- Richard P. Gabriel, *Patterns of Software*
