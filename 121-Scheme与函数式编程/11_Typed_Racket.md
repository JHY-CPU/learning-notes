# Typed Racket

## 概述

Typed Racket 是 Racket 的类型化变体，为 Racket 程序提供静态类型检查。它由 Sam Tobin-Hochhardt 开发，设计目标是与普通 Racket 代码无缝互操作，允许逐步将无类型代码迁移到有类型代码。

### 设计动机

动态类型语言提供了极大的灵活性，但随着代码规模增长，缺乏类型检查会导致：
- 运行时才暴露的类型错误难以调试
- 重构时缺乏编译器辅助，难以确定改动的影响范围
- 代码即文档的能力不足——没有类型签名的函数难以理解

Typed Racket 的设计理念：

- **渐进类型化**（Gradual Typing）：可以逐步添加类型，不必一次性全部转换。这是 Typed Racket 最核心的设计理念。
- **高精度类型**：类型系统足够精确，能表达 Racket 的动态特性（如联合类型、出现类型）
- **类型推断**：大多数类型可以自动推断，不必手动标注
- **健全性**（Soundness）：类型检查通过的程序不会有类型错误（在互操作边界除外）
- **与 Racket 互操作**：可以在同一项目中混用有类型和无类型模块

### 渐进类型化的意义

渐进类型化意味着你不需要在"动态类型"和"静态类型"之间做二选一：

```
阶段 1: 全部无类型代码（#lang racket）
阶段 2: 关键模块添加类型（#lang typed/racket + require/typed）
阶段 3: 大部分代码有类型（#lang typed/racket）
阶段 4: 全部有类型，但仍可依赖无类型的第三方库
```

这种策略不同于 TypeScript 的结构化类型系统——Typed Racket 使用更严格的名义类型和出现类型（occurrence typing）来保证健全性。

```scheme
#lang typed/racket
;; Typed Racket 的基本使用

;; 类型注解语法: [变量 : 类型]
(: greet : String -> String)
(define (greet name)
  (string-append "Hello, " name "!"))

(greet "Alice")  ; => "Hello, Alice!"
;; (greet 42)    ; 类型错误！

;; 类型注解可以省略（通过类型推断）
(define x 42)       ; x : Integer
(define y 3.14)     ; y : Flonum
(define z "hello")  ; z : String
```

## 类型系统架构

Typed Racket 的类型系统非常丰富，支持多态、递归类型、依赖类型（有限形式）等高级特性。

### 类型层次结构

```
Top (Any)
├── Number
│   ├── Integer
│   ├── Flonum
│   ├── Exact-Positive-Integer
│   ├── Exact-Nonnegative-Integer
│   └── Real
├── String
├── Symbol
├── Boolean
│   ├── True (#t)
│   └── False (#f)
├── Char
├── Bytes
├── Void
├── (Listof T)
├── (Vectorof T)
├── (HashTable K V)
├── (-> T1 T2 ... R)     ; 函数类型
├── (U T1 T2 ...)         ; 联合类型
├── (Pair A B)            ; 序对类型
├── (Option T)            ; 选项类型 (= (U T False))
└── Bottom (Nothing)
```

## 基本类型

Typed Racket 提供了丰富的基本类型，比大多数静态类型语言更精细：

```scheme
#lang typed/racket

;; 基本类型
: Integer        ; 精确整数: 42, -7, 0
: Flonum         ; 浮点数: 3.14, -2.5
: Number         ; 任意数字
: String         ; 字符串: "hello"
: Symbol         ; 符号: 'foo
: Boolean        ; #t 或 #f
: Char           ; 字符: #\a
: Bytes          ; 字节串
: Void           ; 无返回值
: Any            ; 任意类型（Top 类型）
: Nothing        ; 不存在的类型（Bottom 类型，用于标记永不返回的函数）

;; 更精确的数字类型
: Exact-Positive-Integer      ; 精确正整数
: Exact-Nonnegative-Integer   ; 精确非负整数
: Natural                     ; 同 Exact-Nonnegative-Integer
: Index                       ; 用作向量/字符串索引的非负精确整数
: Positive-Integer            ; 正整数（可能非精确）
: Nonnegative-Integer         ; 非负整数（可能非精确）
: Negative-Integer            ; 负整数
: Zero                        ; 精确 0

;; 字面量类型（Singleton Types）
: 42              ; 类型就是值 42
: "hello"         ; 类型就是值 "hello"
: 'red            ; 类型就是符号 'red
: #t              ; 类型就是布尔值 #t

;; 字面量类型用于定义枚举类型
(define-type Color (U 'red 'green 'blue))
(define-type Weekday (U 'mon 'tue 'wed 'thu 'fri))
(define-type Weekend (U 'sat 'sun))
(define-type Day (U Weekday Weekend))

;; 使用类型别名
(: color-name : Color -> String)
(define (color-name c)
  (case c
    [(red)   "红色"]
    [(green) "绿色"]
    [(blue)  "蓝色"]))

(color-name 'red)  ; => "红色"
;; (color-name 'yellow)  ; 类型错误！'yellow 不是 Color
```

### 类型之间的子类型关系

```
Integer <: Exact-Positive-Integer <: Exact-Nonnegative-Integer <: Real <: Number
'tes <: String
#t <: Boolean

;; 联合类型是其成员类型的超类型
Integer <: (U Integer String)
String  <: (U Integer String)

;; 字面量类型是其所属类型的子类型
'red <: Symbol
'red <: (U 'red 'green 'blue)
42  <: Integer
42  <: Positive-Integer
```

## 函数类型

```scheme
#lang typed/racket

;; 函数类型注解: (参数类型 ... -> 返回类型)
(: square : Number -> Number)
(define (square x) (* x x))

;; 多参数函数
(: add : Number Number -> Number)
(define (add x y) (+ x y))

;; 高阶函数类型
(: twice : (Number -> Number) Number -> Number)
(define (twice f x) (f (f x)))

(twice add1 5)  ; => 7

;; 多态函数（参数多态 / 泛型）
(: my-map : (All (A B) (A -> B) (Listof A) -> (Listof B)))
(define (my-map f lst)
  (if (null? lst)
      '()
      (cons (f (car lst)) (my-map f (cdr lst)))))

(my-map add1 '(1 2 3))           ; => (2 3 4)
(my-map number->string '(1 2 3)) ; => ("1" "2" "3")

;; 可选参数
(: greet : String [#:prefix String] -> String)
(define (greet name #:prefix [p "Hello"])
  (string-append p ", " name "!"))

(greet "Alice")                ; => "Hello, Alice!"
(greet "Alice" #:prefix "Hi") ; => "Hi, Alice!"

;; 可变参数
(: sum : Number * -> Number)
(define (sum . nums)
  (apply + nums))

(sum 1 2 3 4 5)  ; => 15

;; 关键字参数
(: connect : String #:port Integer [#:timeout Real] -> Void)
(define (connect host #:port port #:timeout [t 30.0])
  (printf "连接 ~a:~a (超时: ~as)~%" host port t))

;; 函数类型箭头的语法:
;; A -> B             ; 一元函数
;; A B -> C           ; 二元函数
;; A * -> B           ; 可变参数（同类型）
;; A [B] -> C         ; 可选参数
;; (All (T) T -> T)   ; 多态函数
```

### 函数类型的子类型规则

```scheme
;; 函数类型是逆变的（contravariant）在参数位置，协变的（covariant）在返回位置
;; 如果 Dog <: Animal，则：
;; (Animal -> Cat) <: (Dog -> Cat)    ; 参数逆变
;; (Animal -> Dog) <: (Animal -> Cat) ; 返回值协变

;; 组合起来：
;; (Animal -> Dog) <: (Dog -> Cat) 的错误！
;; 但 (Animal -> Dog) <: (Animal -> Dog) 是正确的
```

## 复合类型

### 列表和序对

```scheme
#lang typed/racket

;; 元组类型（固定长度的列表）
(define-type Point3D (List Real Real Real))
(: make-point : Real Real Real -> Point3D)
(define (make-point x y z) (list x y z))

;; 元组的精确类型
(: get-x : Point3D -> Real)
(define (get-x p) (car p))

;; 序对类型
(define-type KV (Pair String Integer))
(: make-kv : String Integer -> KV)
(define (make-kv k v) (cons k v))

;; 同构列表（所有元素同一类型）
(: nums : (Listof Integer))
(define nums '(1 2 3 4 5))

;; 异构列表（每个位置不同类型）
(: mixed : (List String Integer Boolean))
(define mixed (list "hello" 42 #t))
```

### 向量和哈希表

```scheme
#lang typed/racket

;; 向量类型
(: vec : (Vector Integer Integer Integer))
(define vec (vector 1 2 3))

;; 可变长度向量
(: dyn-vec : (Vectorof Integer))
(define dyn-vec (vector 1 2 3 4 5))

(vector-ref dyn-vec 0)  ; => 1
(vector-set! dyn-vec 0 100)

;; 哈希表类型
(: scores : (HashTable String Integer))
(define scores (hash "Alice" 95 "Bob" 87))

(hash-ref scores "Alice")  ; => 95
;; (hash-ref scores "Eve") ; 运行时错误（应提供默认值）

;; 带默认值的安全查询
(hash-ref scores "Eve" (lambda () 0))  ; => 0

;; 不可变 vs 可变哈希表
(: immutable-scores : (Immutable-HashTable String Integer))
(define immutable-scores (make-immutable-hash '(("Alice" . 95))))

;; 更新不可变哈希表会返回新表
(define new-scores (hash-set immutable-scores "Bob" 87))
```

## 结构体类型

```scheme
#lang typed/racket

;; 有类型结构体
(struct Person ([name : String]
                [age : Integer]
                [email : String]))

(: alice : Person)
(define alice (Person "Alice" 30 "alice@example.com"))

(Person-name alice)   ; => "Alice"
(Person-age alice)    ; => 30

;; 透明结构体（可以打印内容）
(struct Point ([x : Real] [y : Real]) #:transparent)
(Point 3 4)  ; => (Point 3 4)

;; 泛型结构体
(struct (A) Pair ([first : A] [second : A]) #:transparent)

(: int-pair : (Pair Integer))
(define int-pair (Pair 1 2))
(Pair-first int-pair)   ; => 1

(: str-pair : (Pair String))
(define str-pair (Pair "a" "b"))

;; 递归类型（用于树等数据结构）
(struct Leaf ([value : Integer]) #:transparent)
(struct Node ([left : Tree] [right : Tree]) #:transparent)
(define-type Tree (U Leaf Node))

(: tree-sum : Tree -> Integer)
(define (tree-sum t)
  (cond
    [(Leaf? t) (Leaf-value t)]
    [(Node? t) (+ (tree-sum (Node-left t))
                  (tree-sum (Node-right t)))]))

;; 构建一棵树
(define my-tree
  (Node (Leaf 1)
        (Node (Leaf 2) (Leaf 3))))

(tree-sum my-tree)  ; => 6
```

### 结构体子类型

```scheme
#lang typed/racket

;; 结构体支持单继承
(struct Animal ([name : String]) #:transparent)
(struct Dog Animal ([breed : String]) #:transparent)
(struct Cat Animal ([color : String]) #:transparent)

(: describe : Animal -> String)
(define (describe a)
  (string-append "Animal: " (Animal-name a)))

;; Dog 和 Cat 都是 Animal 的子类型
(describe (Dog "Buddy" "Golden Retriever"))  ; => "Animal: Buddy"
(describe (Cat "Whiskers" "Orange"))         ; => "Animal: Whiskers"

;; 类型细化：使用谓词函数
(: dog? : Any -> Boolean : Dog)
(define (dog? x) (Dog? x))

(: get-breed : Animal -> (U String False))
(define (get-breed a)
  (if (Dog? a)
      (Dog-breed a)    ; a 被细化为 Dog
      #f))
```

## 联合类型与出现类型

Typed Racket 的**出现类型**（occurrence typing）是其最强大的特性之一：类型检查器可以利用条件判断来细化变量类型。这与 TypeScript 的类型守卫类似，但更加系统化。

### 出现类型的原理

出现类型的核心思想：在条件分支中，类型检查器会跟踪条件如何影响变量的类型。

```scheme
#lang typed/racket

;; 出现类型：类型检查器利用条件判断来细化变量类型
(: process : (U String Integer) -> String)
(define (process x)
  (cond
    [(string? x) (string-append x "!")]  ; x 被细化为 String
    [(integer? x) (number->string x)]    ; x 被细化为 Integer
    ))

;; 在 (string? x) 为 #t 的分支中，
;; x 的类型从 (U String Integer) 细化为 String
;; 在 (string? x) 为 #f 的分支中，
;; x 的类型从 (U String Integer) 细化为 Integer
```

### 出现类型的详细示例

```scheme
#lang typed/racket

;; 复杂的出现类型
(: describe : (U String Integer False) -> String)
(define (describe x)
  (if x
      (if (string? x)
          (string-append "String: " x)    ; x : String
          (number->string x))             ; x : Integer
      "nothing"))                         ; x : False

;; 嵌套条件中的类型细化
(: process-value : (U (Listof Integer) String Integer False) -> String)
(define (process-value x)
  (cond
    [(not x) "nothing"]                              ; x : False
    [(string? x) (string-append "str:" x)]           ; x : String
    [(integer? x) (number->string x)]                ; x : Integer
    [else                                            ; x : (Listof Integer)
     (string-append "list:"
                    (string-join (map number->string x) ","))]))

;; 谓词函数的类型细化
(: positive-integer? : Any -> Boolean : Positive-Integer)
(define (positive-integer? x)
  (and (integer? x) (positive? x)))

(: safe-divide : Real Real -> (U Real False))
(define (safe-divide x y)
  (if (and (not (zero? y)) (real? y))
      (/ x y)
      #f))
```

### 联合类型的常见使用模式

```scheme
#lang typed/racket

;; 选项类型（Maybe 模式）
(define-type (Option A) (U A False))

(: find : (All (A) (A -> Boolean) (Listof A) -> (Option A)))
(define (find pred lst)
  (cond
    [(null? lst) #f]
    [(pred (car lst)) (car lst)]
    [else (find pred (cdr lst))]))

(find even? '(1 3 5 7))  ; => #f
(find even? '(1 2 3 4))  ; => 2

;; 结果类型（Either 模式）
(struct (A) Ok ([value : A]) #:transparent)
(struct (B) Err ([error : B]) #:transparent)
(define-type (Result A B) (U (Ok A) (Err B)))

(: safe-parse-int : String -> (Result Integer String))
(define (safe-parse-int s)
  (with-handlers ([exn:fail? (lambda (e) (Err "解析失败"))])
    (Ok (assert (string->number s) exact-integer?))))

(safe-parse-int "42")    ; => (Ok 42)
(safe-parse-int "abc")   ; => (Err "解析失败")
```

## 多态与类型变量

### 参数多态

```scheme
#lang typed/racket

;; 参数多态（parametric polymorphism）
(: identity : (All (T) T -> T))
(define (identity x) x)

;; 多态数据结构
(struct (A) Stack ([items : (Listof A)]) #:transparent)

(: stack-empty : (All (A) -> (Stack A)))
(define (stack-empty) (Stack '()))

(: push : (All (A) A (Stack A) -> (Stack A)))
(define (push item s)
  (Stack (cons item (Stack-items s))))

(: pop : (All (A) (Stack A) -> (Option (Pair A (Stack A)))))
(define (pop s)
  (let ([items (Stack-items s)])
    (if (null? items)
        #f
        (cons (car items) (Stack (cdr items))))))

(: stack-size : (All (A) (Stack A) -> Integer))
(define (stack-size s)
  (length (Stack-items s)))

;; 使用
(define s0 (stack-empty))
(define s1 (push 1 s0))
(define s2 (push 2 s1))
(define p (pop s2))
(if p
    (printf "弹出: ~a~n" (car p))
    (printf "栈为空~n"))
;; 输出: 弹出: 2
```

### 有限制的多态（Bounded Polymorphism）

```scheme
#lang typed/racket

;; 使用约束类型变量
(: sort-numbers : (Listof Number) -> (Listof Number))
(define (sort-numbers lst)
  (sort lst <))

;; 有限制的多态：T 必须实现 <
;; Typed Racket 不直接支持 Haskell 风格的 typeclass
;; 但可以用 case-lambda 和联合类型模拟

;; 实际应用：比较函数
(: compare : (All (A B) (A -> B) A A -> (U 'less 'equal 'greater)
                  #:when (B B -> Boolean)))
;; 注：Typed Racket 中更常见的是直接使用具体类型
(: my-sort : (All (A) (A A -> Boolean) (Listof A) -> (Listof A)))
(define (my-sort cmp lst)
  ;; 排序实现...
  (sort lst cmp))

(my-sort < '(3 1 4 1 5))     ; => (1 1 3 4 5)
(my-sort string<? '("c" "a" "b"))  ; => ("a" "b" "c")
```

## Typed Racket 的类型检查器工作原理

Typed Racket 的类型检查器执行以下步骤：

### 1. 类型推断

```scheme
#lang typed/racket

;; 以下代码不需要类型注解——类型完全由推断得出
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

;; 类型推断过程：
;; 1. n 用于 (<= n 1)，推断 n : Real（因为 <= 接受 Real）
;; 2. 分支 1: 1 是 Integer
;; 3. 分支 2: (* n (factorial (- n 1)))
;;    - (- n 1) 返回 Real
;;    - (factorial (- n 1)) 递归调用，返回 Integer（已推断）
;;    - (* n Integer)，如果 n 是 Integer 则返回 Integer
;; 4. 两个分支的联合: Integer
;; 最终推断: factorial : Integer -> Integer
```

### 2. 出现类型（Occurrence Typing）

这是 Typed Racket 最独特的特性。类型检查器为每个变量维护一个"类型流"——在条件分支中跟踪类型如何变化：

```scheme
#lang typed/racket

;; 类型检查器为 x 维护的类型流:
(: example : (U String Integer) -> String)
(define (example x)
  ;; 入口: x : (U String Integer)
  (if (string? x)
      ;; true 分支: x 已通过 string? 检查
      ;; (U String Integer) ∩ String = String
      ;; 因此 x : String
      (string-append x " world")
      ;; false 分支: x 未通过 string? 检查
      ;; (U String Integer) \ String = Integer
      ;; 因此 x : Integer
      (number->string x)))
```

### 3. 类型等价和子类型

```scheme
#lang typed/racket

;; Typed Racket 使用结构化的子类型检查
;; (U 1 2 3) 等价于 (U 3 1 2)（联合类型无序）
;; (List Integer String) 是 (Listof (U Integer String)) 的子类型

;; 但 Typed Racket 不支持完整的子类型推断
;; 有时需要显式类型转换
(: coerce : Any -> String)
(define (coerce x)
  (if (string? x)
      x
      (error "不是字符串")))

;; 或者使用 cast
(: my-value : Any)
(define my-value "hello")
(define typed-value (cast my-value String))  ; 强制转换（有风险）
```

## Typed Racket 与无类型 Racket 互操作

### 基本互操作

```scheme
;; 无类型模块 (utils.rkt)
#lang racket
(provide random-list)
(define (random-list n)
  (for/list ([i (in-range n)])
    (random 100)))

;; 有类型模块使用无类型代码
#lang typed/racket
(require/typed "utils.rkt"
  [random-list (Integer -> (Listof Integer))])

(define nums (random-list 10))
(displayln nums)
```

### 互操作的工作原理：契约（Contracts）

Typed Racket 在有类型和无类型代码之间插入**运行时契约**（runtime contracts）来保证类型安全：

```
无类型代码  ──→  契约检查  ──→  有类型代码
  (动态)         (运行时)        (静态保证)

有类型代码  ──→  契约检查  ──→  无类型代码
  (静态保证)      (运行时)        (动态)
```

这意味着：
- 如果无类型代码返回了错误类型的值，契约检查会在运行时捕获
- 性能开销：每个跨边界调用都有一次契约检查
- 健全性：有类型代码中的类型保证不会被无类型代码破坏

```scheme
#lang typed/racket

;; require/typed 自动生成契约
(require/typed "untyped.rkt"
  [get-data (-> String (Listof Integer))])
;; Typed Racket 实际生成：
;; (define get-data
;;   (contract (-> string? (listof integer?))
;;             (untyped-get-data)
;;             ...))

;; 对于复杂类型，契约生成可能很昂贵
;; 使用 #:no-optimize 可以减少开销
(require/typed "heavy.rkt"
  [process (-> (HashTable String Integer) String)]
  #:no-optimize)
```

### 反向互操作：在无类型代码中使用有类型模块

```scheme
;; 有类型模块 (typed-math.rkt)
#lang typed/racket
(provide factorial)
(: factorial : Integer -> Integer)
(define (factorial n)
  (if (<= n 1) 1 (* n (factorial (- n 1)))))

;; 无类型模块使用有类型代码
#lang racket
(require "typed-math.rkt")  ; 自动添加契约
(factorial 10)  ; => 3628800
;; (factorial "hello")  ; 运行时契约错误
```

## Typed Racket vs TypeScript 对比

| 特性 | Typed Racket | TypeScript |
|------|-------------|-----------|
| 基础语言 | Racket（Lisp/Scheme） | JavaScript |
| 类型系统 | 名义类型为主 | 结构化类型为主 |
| 健全性 | 静态保证（互操作边界除外） | 不保证（有 `any` 和类型断言） |
| 出现类型 | 内建且系统化 | 有限支持（类型守卫） |
| 渐进类型化 | 完整支持 | 完整支持 |
| 类型推断 | Hindley-Milner 变体 | 局部推断 |
| 联合类型 | 原生支持 + 出现类型 | 原生支持 + 类型守卫 |
| 递归类型 | 原生支持 | 原生支持 |
| 泛型 | 参数多态 + 有限制的多态 | 参数多态 + 条件类型 |
| 宏系统 | 与宏完全集成 | 无宏 |

### 代码对比

```scheme
#lang typed/racket
;; Typed Racket：出现类型自动细化
(: describe : (U String Integer False) -> String)
(define (describe x)
  (if x
      (if (string? x)
          (string-append "String: " x)
          (number->string x))
      "nothing"))
```

```typescript
// TypeScript：需要类型守卫
type Result = string | number | false;

function describe(x: Result): string {
    if (x === false) {
        return "nothing";
    } else if (typeof x === "string") {
        return `String: ${x}`;
    } else {
        return x.toString();  // x 已被细化为 number
    }
}
```

Typed Racket 的出现类型更强大，因为它可以自动从谓词函数中推断类型细化，而 TypeScript 需要用户显式编写类型守卫。

## 完整示例：类型安全的表达式树

以下是一个综合运用 Typed Racket 各特性的完整示例：

```scheme
#lang typed/racket

;; 定义表达式树的类型
(struct Num ([val : Real]) #:transparent)
(struct Add ([left : Expr] [right : Expr]) #:transparent)
(struct Mul ([left : Expr] [right : Expr]) #:transparent)
(struct Neg ([operand : Expr]) #:transparent)
(struct Var ([name : Symbol]) #:transparent)
(define-type Expr (U Num Add Mul Neg Var))

;; 求值函数
(: eval-expr : Expr (HashTable Symbol Real) -> Real)
(define (eval-expr expr env)
  (cond
    [(Num? expr) (Num-val expr)]
    [(Var? expr) (hash-ref env (Var-name expr)
                           (lambda () (error "未定义变量" (Var-name expr))))]
    [(Add? expr) (+ (eval-expr (Add-left expr) env)
                    (eval-expr (Add-right expr) env))]
    [(Mul? expr) (* (eval-expr (Mul-left expr) env)
                    (eval-expr (Mul-right expr) env))]
    [(Neg? expr) (- (eval-expr (Neg-operand expr) env))]))

;; 表达式转字符串
(: expr->string : Expr -> String)
(define (expr->string expr)
  (cond
    [(Num? expr) (number->string (Num-val expr))]
    [(Var? expr) (symbol->string (Var-name expr))]
    [(Add? expr) (string-append "("
                  (expr->string (Add-left expr)) " + "
                  (expr->string (Add-right expr)) ")")]
    [(Mul? expr) (string-append "("
                  (expr->string (Mul-left expr)) " * "
                  (expr->string (Mul-right expr)) ")")]
    [(Neg? expr) (string-append "-(" (expr->string (Neg-operand expr)) ")")]))

;; 简单的常量折叠优化
(: optimize : Expr -> Expr)
(define (optimize expr)
  (cond
    [(Num? expr) expr]
    [(Var? expr) expr]
    [(Add? expr)
     (let ([left (optimize (Add-left expr))]
           [right (optimize (Add-right expr))])
       (cond
         [(and (Num? left) (Num? right))
          (Num (+ (Num-val left) (Num-val right)))]
         [(and (Num? left) (zero? (Num-val left))) right]
         [(and (Num? right) (zero? (Num-val right))) left]
         [else (Add left right)]))]
    [(Mul? expr)
     (let ([left (optimize (Mul-left expr))]
           [right (optimize (Mul-right expr))])
       (cond
         [(and (Num? left) (Num? right))
          (Num (* (Num-val left) (Num-val right)))]
         [(or (and (Num? left) (zero? (Num-val left)))
              (and (Num? right) (zero? (Num-val right))))
          (Num 0)]
         [(and (Num? left) (= 1 (Num-val left))) right]
         [(and (Num? right) (= 1 (Num-val right))) left]
         [else (Mul left right)]))]
    [(Neg? expr)
     (let ([operand (optimize (Neg-operand expr))])
       (if (Num? operand)
           (Num (- (Num-val operand)))
           (Neg operand)))]))

;; 测试
(define my-expr
  (Add (Mul (Num 3) (Var 'x))
       (Neg (Add (Var 'x) (Num 0)))))

(define env (hash 'x 5))
(expr->string my-expr)                          ; => "((3 * x) + -(x + 0))"
(eval-expr my-expr env)                         ; => 10
(expr->string (optimize my-expr))               ; => "((3 * x) + -(x))"
(eval-expr (optimize my-expr) env)              ; => 10

;; 更多优化示例
(optimize (Add (Num 0) (Var 'x)))               ; => (Var 'x)
(optimize (Mul (Num 1) (Var 'x)))               ; => (Var 'x)
(optimize (Mul (Num 0) (Var 'x)))               ; => (Num 0)
(optimize (Add (Num 3) (Num 4)))                ; => (Num 7)
```

## 常见陷阱

### 1. 联合类型需要所有分支

```scheme
#lang typed/racket

;; 错误：忘记了 False 分支
(: process : (U String False) -> String)
(define (process x)
  (if x
      (string-upcase x)
      ;; 缺少 else 分支会导致类型错误！
      "default"))  ; 必须处理 False 的情况
```

### 2. `cast` 是不安全的

```scheme
#lang typed/racket

;; cast 不做运行时检查，只是告诉编译器相信你
(define x : Any "hello")
(define y (cast x Integer))  ; 编译通过，但运行时会出错
(+ y 1)  ; 运行时错误！

;; 更安全的做法：使用类型守卫
(: safe-to-int : Any -> (Option Integer))
(define (safe-to-int x)
  (if (exact-integer? x) x #f))
```

### 3. 函数类型的参数逆变

```scheme
#lang typed/racket

;; 错误理解：以为 (Integer -> Integer) 可以传给 (Number -> Number)
;; 实际上：函数参数是逆变的

(: call-with-number : (Number -> Number) Number -> Number)
(define (call-with-number f x) (f x))

(: double : Integer -> Integer)
(define (double x) (* x 2))

;; (call-with-number double 5) 类型错误！
;; 因为 double : Integer -> Integer
;; 但 call-with-number 期望 Number -> Number
;; Integer -> Integer 不是 Number -> Number 的子类型（参数逆变！）

;; 解决方案：放宽 double 的类型
(: double-better : Number -> Number)
(define (double-better x) (* x 2))
(call-with-number double-better 5)  ; OK
```

### 4. 类型细化在 let 绑定中丢失

```scheme
#lang typed/racket

;; 陷阱：类型细化不能跨 let 绑定传递
(: example : (U String Integer) -> String)
(define (example x)
  (let ([y x])          ; y : (U String Integer)
    (if (string? y)
        (string-append y "!")
        (number->string y))))  ; y 被细化为 Integer，OK

;; 但这个不行：
(: example2 : (U String Integer) -> String)
(define (example2 x)
  (let ([y x])
    (if (string? x)     ; 细化的是 x，不是 y
        y               ; y 仍然是 (U String Integer)
        ;; (string-append y "!") 类型错误！y 可能是 Integer
        (number->string y))))
```

### 5. 无类型代码的契约开销

```scheme
#lang typed/racket

;; 频繁调用无类型代码可能导致性能问题
(require/typed "slow.rkt"
  [heavy-fn : (-> (Vectorof Integer) (Vectorof Integer))])

;; 每次调用 heavy-fn 都会触发契约检查
;; 对于大向量，契约检查的开销可能很大

;; 解决方案：
;; 1. 批量调用，减少跨边界次数
;; 2. 将热路径代码迁移到 Typed Racket
;; 3. 使用 #:no-optimize 减少契约生成（有风险）
```

## 练习题

### 练习 1：类型安全的字典

实现一个类型安全的字典数据结构，支持 `get`、`set`、`delete` 和 `keys` 操作：

```scheme
#lang typed/racket
;; 参考答案
(struct (K V) Dict ([data : (Listof (Pair K V))]) #:transparent)

(: dict-empty : (All (K V) -> (Dict K V)))
(define (dict-empty) (Dict '()))

(: dict-set : (All (K V) (Dict K V) K V -> (Dict K V)))
(define (dict-set d k v)
  (Dict (cons (cons k v)
              (filter (lambda ([p : (Pair K V)])
                        (not (equal? (car p) k)))
                      (Dict-data d)))))

(: dict-get : (All (K V) (Dict K V) K -> (Option V)))
(define (dict-get d k)
  (let ([pair (assoc k (Dict-data d))])
    (if pair (cdr pair) #f)))

(: dict-keys : (All (K V) (Dict K V) -> (Listof K)))
(define (dict-keys d)
  (map (lambda ([p : (Pair K V)]) (car p)) (Dict-data d)))

;; 测试
(define d0 (dict-empty))
(define d1 (dict-set d0 "Alice" 95))
(define d2 (dict-set d1 "Bob" 87))
(dict-get d2 "Alice")    ; => (Ok 95)
(dict-get d2 "Eve")      ; => #f
(dict-keys d2)           ; => ("Bob" "Alice")
```

### 练习 2：类型安全的状态机

使用联合类型和结构体实现一个类型安全的有限状态机（门锁）：

```scheme
#lang typed/racket
;; 参考答案
(struct Locked () #:transparent)
(struct Unlocked () #:transparent)
(define-type LockState (U Locked Unlocked))

(struct LockMachine ([state : LockState] [attempts : Integer]) #:transparent)

(: initial-lock : LockMachine)
(define initial-lock (LockMachine (Locked) 0))

(: unlock : LockMachine String -> (U LockMachine String))
(define (unlock machine code)
  (if (Locked? (LockMachine-state machine))
      (if (string=? code "1234")
          (LockMachine (Unlocked) 0)
          (let ([new-attempts (+ 1 (LockMachine-attempts machine))])
            (if (>= new-attempts 3)
                "锁定！尝试次数过多"
                (LockMachine (Locked) new-attempts))))
      "已经解锁了"))

(: lock : LockMachine -> LockMachine)
(define (lock machine)
  (if (Unlocked? (LockMachine-state machine))
      (LockMachine (Locked) 0)
      machine))

;; 测试
(define m0 initial-lock)
(define m1 (unlock m0 "0000"))        ; 尝试失败
(define m2 (if (LockMachine? m1) (unlock m1 "0000") m1))  ; 再次失败
(define m3 (if (LockMachine? m2) (unlock m2 "1234") m2))  ; 成功解锁
```

### 练习 3：类型安全的 JSON 值

定义一个表示 JSON 值的递归类型，并实现序列化函数：

```scheme
#lang typed/racket
;; 参考答案
(struct JNull () #:transparent)
(struct JBool ([val : Boolean]) #:transparent)
(struct JNum ([val : Real]) #:transparent)
(struct JStr ([val : String]) #:transparent)
(struct JArr ([items : (Listof JSON)]) #:transparent)
(struct JObj ([pairs : (Listof (Pair String JSON))]) #:transparent)
(define-type JSON (U JNull JBool JNum JStr JArr JObj))

(: json->string : JSON -> String)
(define (json->string j)
  (cond
    [(JNull? j) "null"]
    [(JBool? j) (if (JBool-val j) "true" "false")]
    [(JNum? j) (number->string (JNum-val j))]
    [(JStr? j) (string-append "\"" (JStr-val j) "\"")]
    [(JArr? j) (string-append "["
                 (string-join (map json->string (JArr-items j)) ", ") "]")]
    [(JObj? j)
     (string-append "{"
       (string-join
         (map (lambda ([p : (Pair String JSON)])
                (string-append "\"" (car p) "\": " (json->string (cdr p))))
              (JObj-pairs j))
         ", ") "}")]))

;; 测试
(define data
  (JObj (list (cons "name" (JStr "Alice"))
              (cons "age" (JNum 30))
              (cons "active" (JBool #t))
              (cons "scores" (JArr (list (JNum 95) (JNum 87)))))))

(json->string data)
;; => "{\"name\": \"Alice\", \"age\": 30, \"active\": true, \"scores\": [95, 87]}"
```
