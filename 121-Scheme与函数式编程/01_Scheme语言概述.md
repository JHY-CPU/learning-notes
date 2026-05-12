# Scheme 语言概述

## Lisp 家族与 Scheme 的设计哲学

Scheme 是 Lisp 家族中最具学术影响力的方言之一，由 **Guy L. Steele** 和 **Gerald Jay Sussman** 于 1975 年在 MIT 人工智能实验室开发。Scheme 的设计目标是将 **Lambda 演算** 直接映射到编程语言中，追求极简主义和正交性。

### 设计动机：为什么需要 Scheme？

1970 年代，AI 研究大量使用 Lisp，但当时的 MacLisp 和 InterLisp 等实现充满了特殊规则和历史包袱。Steele 和 Sussman 想要回答一个根本性的问题：**一个编程语言最少需要哪些概念才能具有完整的表达力？**

他们的答案就是 Scheme——一个基于 Lambda 演算的形式系统，只有大约 5 条核心规则。这种极简性并非偷懒，而是一种哲学立场：语言应该提供最少量的正交原语，让程序员通过组合这些原语构建复杂系统。

### 与 Common Lisp 的对比

| 特性 | Scheme | Common Lisp |
|------|--------|-------------|
| 设计哲学 | 极简、正交 | 实用、"厨房水槽" |
| 作用域 | 词法作用域（原生） | 词法为主，可选动态 |
| 尾调用优化 | 标准要求必须优化 | 不要求 |
| 标准特殊形式 | ~26 个 | ~100+ 个（含宏） |
| 多重返回值 | 可选（R7RS） | 内建 |
| 面向对象 | 无标准（有 SRFI） | CLOS 内建 |
| 适用场景 | 教学、研究、嵌入 | 大型工业项目 |

```scheme
;; Scheme 的极简语法：所有表达式都是 (操作数 参数...)
(+ 1 2 3)                    ; => 6
(define (square x) (* x x))  ; 定义一个求平方的函数
(square 5)                    ; => 25

;; Lambda 表达式：Scheme 的核心构造
((lambda (x y) (+ x y)) 3 4) ; => 7
```

Scheme 与 Common Lisp 的根本区别可以用一句话概括：**Scheme 问"最少需要什么？"，Common Lisp 问"实际编程需要什么？"**

### Scheme 的核心原则

Scheme 遵循以下设计原则：

- **极简核心**：语言的核心构造尽可能少，但表达力极强。Scheme 的全部特殊形式可以用一只手数完：`lambda`、`if`、`define`、`set!`、`quote`。
- **一等过程**：函数是一等公民，可以作为参数传递、作为返回值返回、存储在数据结构中。这是 Scheme 表达力的根基。
- **尾递归优化**：标准要求实现必须对尾调用进行优化，使得递归可以替代循环，且不消耗额外栈空间。
- **词法作用域**：从诞生之初就采用了词法作用域（lexical scoping），这在 1975 年是相当先进的选择。
- **同像性（Homoiconicity）**：代码即数据——Scheme 程序本身的数据表示就是 Scheme 的列表结构，这意味着 Scheme 程序可以生成和操作 Scheme 程序。

Scheme 的语法极其简单——整个语言建立在 S-表达式（symbolic expression）之上，代码和数据使用相同的表示形式。这意味着 Scheme 程序本身就可以被 Scheme 程序处理，这是宏系统的根基。

## 从 Lambda 演算到 Scheme

Scheme 的核心就是 Lambda 演算的直接实现。理解 Lambda 演算有助于理解 Scheme 为何如此设计。

Alonzo Church 在 1930 年代提出的 Lambda 演算只有三个概念：

1. **变量**（Variable）：`x`
2. **抽象**（Abstraction）：`λx.M`（定义一个接受参数 x 并返回 M 的函数）
3. **应用**（Application）：`(M N)`（将函数 M 应用于参数 N）

Scheme 几乎是逐字对应地实现了这些概念：

```
Lambda 演算              Scheme
─────────────────────────────────────
λx.x                    (lambda (x) x)
λx.λy.x                 (lambda (x) (lambda (y) x))
(λx.x) 5               ((lambda (x) x) 5)
(λx.λy.x) 3 4          (((lambda (x) (lambda (y) x)) 3) 4)
```

Sussman 和 Steele 在他们 1975 年的论文 "Scheme: An Interpreter for Extended Lambda Calculus" 中明确指出，Scheme 的目标是创建一个可以直接执行 Lambda 演算的解释器。`define`、`if` 等形式只是为了方便使用而添加的语法糖。

```scheme
;; 在纯 Lambda 演算中，所有东西都可以用 lambda 表达
;; Church 布尔值
(define TRUE  (lambda (x) (lambda (y) x)))
(define FALSE (lambda (x) (lambda (y) y)))

;; Church 数：自然数的函数式编码
;; n = λf.λx. f^n(x)
(define zero  (lambda (f) (lambda (x) x)))
(define one   (lambda (f) (lambda (x) (f x))))
(define two   (lambda (f) (lambda (x) (f (f x)))))

;; 后继函数 succ(n) = n + 1
(define (church-succ n)
  (lambda (f) (lambda (x) (f ((n f) x)))))

;; 验证：将 Church 数转换为普通整数
(define (church->int n) ((n add1) 0))
(church->int (church-succ two))  ; => 3

;; Church 对
(define (church-cons a b)
  (lambda (f) ((f a) b)))
(define (church-pair-first p)  (p TRUE))
(define (church-pair-second p) (p FALSE))

(church-pair-first (church-cons 3 4))   ; => 3
(church-pair-second (church-cons 3 4))  ; => 4
```

这揭示了一个深刻的结论：**Scheme 的全部计算能力来自 `lambda` 这一个概念**。其他一切（数字运算、条件判断、数据结构）都可以在 lambda 之上构建。这也是 SICP 第 2 章的核心主题之一。

## Scheme 标准：R5RS、R6RS 与 R7RS

Scheme 由一系列技术报告（Revised^n Report on the Algorithmic Language Scheme）规范，每个版本都有不同的社区接受度。

### R5RS（1998）

R5RS 是最广泛使用的 Scheme 标准，被 SICP 等经典教材采用。它定义了一个非常小的核心语言，刻意保持简洁：

- 约 26 个特殊形式和语法关键字
- 没有标准的模块系统
- 没有标准的异常处理
- 实现之间差异较大

R5RS 的极简性既是优点也是缺点。优点是语言小到可以完全装进脑子里；缺点是"可移植的 R5RS 程序"几乎没有实用价值——你需要依赖 SRFI（Scheme Requests for Implementation）来获取基本功能，如命令行参数解析、正则表达式、哈希表等。

```scheme
;; R5RS 风格的代码示例
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

(display (factorial 10))  ; => 3628800

;; R5RS 没有内建的哈希表、异常处理或模块系统
;; 需要依赖 SRFI 或实现特定的扩展
```

### R6RS（2007）

R6RS 试图标准化更多特性，引入了库系统、字节向量、更严格的语义等。它在社区中引发了一些争议——许多人认为 R6RS 太复杂，背离了 Scheme 的极简精神。

R6RS 的争议点包括：
- 强制不可变的 `pair` 语义（默认 pair 不可 `set-car!`/`set-cdr!`）
- 复杂的库和导入系统
- 与 R5RS 的大量不兼容
- 许多实现选择不完整支持 R6RS

```scheme
;; R6RS 风格：使用库（library）系统
(library (my-math)
  (export square cube)
  (import (rnrs))

  (define (square x) (* x x))
  (define (cube x) (* x x x)))
```

### R7RS（2013）与 R7RS-large

R7RS 分为 small 和 large 两部分。R7RS-small 回归简洁，修复了 R6RS 的一些争议，同时吸收了有用的新特性：

- 改进的库系统（`define-library`）
- 例外和条件系统
- 字节向量
- 动态绑定参数（`parameterize`）
- 保持了与 R5RS 的较高兼容性

R7RS-large 正在进行中，试图在 R7RS-small 之上构建一个实用的标准库。

```scheme
;; R7RS 风格：使用 define-library
(define-library (my-math)
  (export square cube)
  (import (scheme base))
  (begin
    (define (square x) (* x x))
    (define (cube x) (* x x x))))

;; R7RS 特有功能
(import (scheme write))
(define current-log-level (make-parameter 'info))
(parameterize ([current-log-level 'debug])
  (displayln (current-log-level)))  ; => debug
```

## Scheme 实现与开发环境

Scheme 有众多实现，各有特色。选择合适的实现取决于你的目标：学习、研究还是生产。

### Racket

Racket 是最流行的 Scheme 方言实现，拥有丰富的生态系统。严格来说，Racket 已经发展成一门独立的语言（支持 `#lang racket`、`#lang typed/racket`、`#lang lazy` 等多种语言变体），但其核心仍保留着 Scheme 的基因。

Racket 的独特优势：
- **DrRacket**：为教学设计的集成开发环境，具有逐步求值器（Stepper）、语法绘图等功能
- **#lang 系统**：可以定义自己的语言，用 `#lang` 声明切换
- **宏系统**：拥有 `syntax-rules` 和更强大的 `syntax-parse`
- **包管理**：`raco pkg install` 提供了方便的包管理

```scheme
#lang racket
;; Racket 中的简单程序
(define (fibonacci n)
  (if (< n 2)
      n
      (+ (fibonacci (- n 1)) (fibonacci (- n 2)))))

(for ([i (in-range 10)])
  (printf "fib(~a) = ~a~n" i (fibonacci i)))

;; Racket 的面向对象系统
(require racket/class)
(define circle%
  (class object%
    (init-field radius)
    (define/public (area)
      (* pi radius radius))
    (super-new)))

(define c (new circle% [radius 5]))
(send c area)  ; => 78.5398...
```

### GNU Guile

Guile 是 GNU 项目的官方扩展语言，被许多 GNU 工具（如 GDB、GIMP）使用：

```scheme
;; Guile 中的模块定义
(define-module (my utils)
  #:export (greet))

(define (greet name)
  (format #t "Hello, ~a!~%" name))

;; Guile 的 Web 编程
(use-modules (web server)
             (web response))
(define (handler request body)
  (values '((content-type . (text/plain)))
          "Hello from Guile!"))
;; (run-server handler 'http '(#:port 8080))
```

### MIT/GNU Scheme

MIT Scheme 是 SICP 的"原配"实现，最接近书中描述的语言子集。如果你正在跟随 SICP 学习，MIT Scheme 或 Racket（配合 SICP 语言包）都是很好的选择。

### Chicken Scheme

Chicken Scheme 将 Scheme 编译为 C 代码，拥有独特的 CPS（Continuation-Passing Style）编译策略。它有丰富的 Eggs（包）生态系统，适合实际部署。

### Chez Scheme

Chez Scheme 被认为是最快的 Scheme 实现之一，由 Cisco 开源。它被 Racket 的 Chez Scheme 后端采用，证明了 Scheme 可以拥有工业级的性能。

Chez Scheme 的编译器直接生成优化的机器码，而非先编译到 C。它的垃圾回收器也经过高度优化。

### Gambit Scheme

Gambit 同样将 Scheme 编译到 C，性能出色，支持多线程。Gerbil Scheme 基于 Gambit 构建，提供了更现代的开发体验。

### 实现选择建议

| 目标 | 推荐实现 | 理由 |
|------|----------|------|
| SICP 学习 | Racket (SICP lang) 或 MIT/GNU Scheme | 教材原配或最佳兼容 |
| 函数式编程教学 | Racket | 最好的 IDE 和文档 |
| 系统编程/性能 | Chez Scheme | 最快的编译器 |
| GNU 工具扩展 | GNU Guile | GNU 官方语言 |
| 嵌入式/生产 | Chicken 或 Gambit | 编译到 C，部署方便 |

## REPL 交互式开发

REPL（Read-Eval-Print Loop）是 Scheme 开发的核心工作流。在 REPL 中可以即时测试表达式、探索 API、调试代码。

REPL 的四个阶段不仅是名字，更是 Scheme 解释器架构的直接映射：

```
用户输入: (* (+ 2 3) (- 7 4))
        │
        ▼
┌─────────────────────────────────┐
│  READ（读取）                    │
│  解析 S-表达式                   │
│  输入 → 内部树结构               │
│  "(* (+ 2 3) (- 7 4))"          │
│       → (* (+ 2 3) (- 7 4))     │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  EVAL（求值）                    │
│  递归求值表达式树                │
│  (+ 2 3) → 5                    │
│  (- 7 4) → 3                    │
│  (* 5 3) → 15                   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  PRINT（打印）                   │
│  将求值结果格式化为字符串        │
│  15 → "15"                      │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  LOOP（循环）                    │
│  回到 READ，等待下一个表达式     │
└─────────────────────────────────┘
```

```scheme
;; REPL 会话示例
> (+ 1 2 3)
6
> (define x 42)
> (* x x)
1764
> (define (hello) (display "Hello, Scheme!\n"))
> (hello)
Hello, Scheme!
> (map (lambda (x) (* x x)) '(1 2 3 4 5))
(1 4 9 16 25)

;; REPL 中的调试技巧
> (define (buggy n)
    (if (= n 0)
        "done"
        (buggy (- n 1))))
> (buggy 3)  ; 可以在 REPL 中逐步修改和重新测试
```

SICP 第 4 章的核心内容就是从零开始实现一个 Scheme 求值器——你需要实现 `eval`（处理特殊形式）和 `apply`（执行过程调用），这两个函数相互递归，构成了整个语言的核心。

## Scheme 与其它语言的对比

### 语法对比

```scheme
;; Scheme
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))
```

```python
# Python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

```javascript
// JavaScript
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

对比可以看到：Scheme 的括号嵌套虽然初看密集，但它是**完全统一的**——每个表达式都是 `(操作 参数...)` 的形式。Python 和 JavaScript 需要额外的语法糖来处理不同的语句类型（`if/else`、`for`、`while`、`return` 等）。

### 一等函数的对比

Scheme 从 1975 年就原生支持一等函数，而 JavaScript（1995）和 Python（1991）是后来才加入这一特性的。

```scheme
;; Scheme：函数天然是一等公民
(define (compose f g)
  (lambda (x) (f (g x))))

(define add1-squared (compose (lambda (x) (* x x)) add1))
(add1-squared 4)  ; => 25  (5 的平方)
```

关键区别：Scheme 没有 `return` 关键字——每个函数的值就是最后一个表达式的值。Scheme 没有语句（statement），一切都是表达式（expression）。这种"一切皆表达式"的设计使得 Scheme 的每个位置都可以放任何表达式，不需要区分语句和表达式的上下文。

## 常见陷阱

### 1. 括号匹配

初学者最常犯的错误就是括号不匹配。Scheme 没有大括号和中括号作为语法（Racket 中方括号是圆括号的别名），所以所有结构都靠圆括号。

```scheme
;; 错误：括号不匹配
(+ 1 2     ; 缺少右括号

;; 建议：使用编辑器的括号匹配功能
;; 在 DrRacket 中，匹配的括号会自动高亮
```

### 2. 命名约定

Scheme 社区使用 kebab-case（连字符分隔）而非 camelCase 或 snake_case：

```scheme
;; Scheme 风格
(define my-variable-name 42)   ; 正确
(define myVariableName 42)     ; 不推荐（虽然合法）
(define my_variable_name 42)   ; 不推荐
```

### 3. `define` vs `set!`

`define` 创建新绑定，`set!` 修改已有绑定。混淆两者是常见错误：

```scheme
;; define 创建绑定
(define x 10)

;; set! 修改绑定
(set! x 20)

;; 错误：对未定义的变量使用 set!
(set! y 30)  ; 错误！y 未定义
```

### 4. 列表 vs 函数调用

```scheme
;; 这是一个函数调用，不是列表
(+ 1 2 3)  ; => 6

;; 这是一个列表数据
'(+ 1 2 3)  ; => (+ 1 2 3)

;; 初学者常犯的错误
(list + 1 2 3)  ; => (#<procedure:+> 1 2 3) —— 把 + 过程放进列表
'(+ 1 2 3)      ; => (+ 1 2 3) —— 把符号 + 放进列表
```

## 练习题

### 练习 1：Church 编码

使用纯 Lambda 演算（不使用 `define`、数字、布尔值等内建类型）实现以下功能：

1. 定义 Church 数 `four`
2. 定义加法函数 `church-add`
3. 定义乘法函数 `church-mul`

```scheme
;; 参考答案
(define four (lambda (f) (lambda (x) (f (f (f (f x)))))))

(define (church-add m n)
  (lambda (f) (lambda (x) ((m f) ((n f) x)))))

(define (church-mul m n)
  (lambda (f) (m (n f))))

;; 验证
(define (church->int n) ((n add1) 0))
(church->int four)                        ; => 4
(church->int (church-add two three))      ; => 5
(church->int (church-mul two three))      ; => 6
```

### 练习 2：实现一个 Scheme 解释器的 READ 部分

编写一个函数 `tokenize`，将 Scheme 表达式字符串分解为 token 列表：

```scheme
;; 参考答案
(define (tokenize str)
  (define tokens '())
  (define current "")
  (define (flush)
    (when (not (string=? current ""))
      (set! tokens (append tokens (list current)))
      (set! current "")))
  (string-for-each
   (lambda (ch)
     (cond
       [(char=? ch #\()  (flush) (set! tokens (append tokens (list "("))))]
       [(char=? ch #\))  (flush) (set! tokens (append tokens (list ")")))]
       [(char=? ch #\space) (flush)]
       [else (set! current (string-append current (string ch)))]))
   str)
  (flush)
  tokens)

(tokenize "(+ 1 (* 2 3))")
;; => ("(" "+" "1" "(" "*" "2" "3" ")" ")")
```

### 练习 3：Scheme 方言比较

分别用 R5RS 风格、R6RS 库风格和 R7RS 库风格实现一个简单的数学工具库，包含 `square`、`cube`、`factorial` 三个函数。比较三种写法的差异。

```scheme
;; R5RS 风格
(define (square x) (* x x))
(define (cube x) (* x x x))
(define (factorial n)
  (if (<= n 1) 1 (* n (factorial (- n 1)))))

;; R6RS 风格
(library (math-tools)
  (export square cube factorial)
  (import (rnrs))
  (define (square x) (* x x))
  (define (cube x) (* x x x))
  (define (factorial n)
    (if (<= n 1) 1 (* n (factorial (- n 1))))))

;; R7RS 风格
(define-library (math-tools)
  (export square cube factorial)
  (import (scheme base))
  (begin
    (define (square x) (* x x))
    (define (cube x) (* x x x))
    (define (factorial n)
      (if (<= n 1) 1 (* n (factorial (- n 1)))))))
```
