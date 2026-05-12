# SICP 核心概念

## 数据抽象与抽象屏障

SICP（《计算机程序的构造和解释》）的核心主题之一是**抽象屏障**（abstraction barrier）的思想。数据抽象将数据的**使用方式**与**实现方式**分离。

```scheme
;; 有理数的抽象数据类型
;; 使用层：只关心 make-rat, numer, denom
;; 实现层：如何用 cons/car/cdr 表示

;; 构造函数和选择函数
(define (make-rat n d)
  (let ([g (gcd n d)])
    (cons (/ n g) (/ d g))))

(define (numer rat) (car rat))
(define (denom rat) (cdr rat))

;; 使用有理数的运算（不关心内部表示）
(define (add-rat x y)
  (make-rat (+ (* (numer x) (denom y))
               (* (numer y) (denom x)))
            (* (denom x) (denom y))))

(define (mul-rat x y)
  (make-rat (* (numer x) (numer y))
            (* (denom x) (denom y))))

(define (print-rat rat)
  (printf "~a/~a~n" (numer rat) (denom rat)))

;; 使用
(define r1 (make-rat 1 3))
(define r2 (make-rat 1 6))
(print-rat (add-rat r1 r2))  ; => 1/2
(print-rat (mul-rat r1 r2))  ; => 1/18

;; 可以改变内部表示而不影响使用层
;; 例如，改为 (cons n d) 不约分，在 numer/denom 中约分
```

## 消息传递与面向对象

SICP 使用闭包实现**消息传递**风格的面向对象编程：

```scheme
;; 使用消息传递实现银行账户
(define (make-account balance)
  (define (withdraw amount)
    (if (>= balance amount)
        (begin (set! balance (- balance amount))
               balance)
        "余额不足"))
  (define (deposit amount)
    (set! balance (+ balance amount))
    balance)
  (define (dispatch msg)
    (cond
      [(eq? msg 'withdraw) withdraw]
      [(eq? msg 'deposit) deposit]
      [(eq? msg 'balance) balance]
      [else (error "未知请求" msg)]))
  dispatch)

(define acc (make-account 100))
((acc 'withdraw) 50)   ; => 50
((acc 'deposit) 30)    ; => 80
(acc 'balance)         ; => 80
```

## 流处理（Stream Processing）

流是 SICP 中处理大规模或无限序列的抽象。流使用**惰性求值**：只在需要时计算元素。

```scheme
;; 流的基本构造
(define-syntax-rule (cons-stream a b)
  (cons a (delay b)))

(define (stream-car s) (car s))
(define (stream-cdr s) (force (cdr s)))
(define stream-null? null?)
(define the-empty-stream '())

;; 流的高阶操作
(define (stream-map proc s)
  (if (stream-null? s)
      the-empty-stream
      (cons-stream (proc (stream-car s))
                   (stream-map proc (stream-cdr s)))))

(define (stream-filter pred s)
  (cond
    [(stream-null? s) the-empty-stream]
    [(pred (stream-car s))
     (cons-stream (stream-car s)
                  (stream-filter pred (stream-cdr s)))]
    [else (stream-filter pred (stream-cdr s))]))

(define (stream-ref s n)
  (if (= n 0)
      (stream-car s)
      (stream-ref (stream-cdr s) (- n 1))))

(define (stream-for-each proc s)
  (unless (stream-null? s)
    (proc (stream-car s))
    (stream-for-each proc (stream-cdr s))))

;; 无穷整数流
(define (integers-starting-from n)
  (cons-stream n (integers-starting-from (+ n 1))))

(define integers (integers-starting-from 1))

(stream-ref integers 0)    ; => 1
(stream-ref integers 99)   ; => 100

;; 使用流处理数据
(stream-car
  (stream-filter even?
    (stream-map (lambda (x) (* x x))
                integers)))
; => 4（第一个偶数平方）
```

## 元循环求值器（Metacircular Evaluator）

SICP 最著名的章节之一是用 Scheme 实现一个 Scheme 求值器（eval）。这展示了语言的自举能力。

```scheme
;; 元循环求值器的核心
(define (my-eval exp env)
  (cond
    ;; 自求值表达式
    [(number? exp) exp]
    [(string? exp) exp]

    ;; 变量查找
    [(symbol? exp) (lookup-variable exp env)]

    ;; 引用
    [(tagged-list? exp 'quote) (cadr exp)]

    ;; if 表达式
    [(tagged-list? exp 'if)
     (if (my-eval (cadr exp) env)
         (my-eval (caddr exp) env)
         (if (pair? (cdddr exp))
             (my-eval (cadddr exp) env)
             #f))]

    ;; lambda 表达式
    [(tagged-list? exp 'lambda)
     (make-procedure (cadr exp) (cddr exp) env)]

    ;; define 表达式
    [(tagged-list? exp 'define)
     (define-variable! (cadr exp)
                       (my-eval (caddr exp) env)
                       env)]

    ;; 应用（函数调用）
    [(pair? exp)
     (my-apply (my-eval (car exp) env)
               (map (lambda (e) (my-eval e env))
                    (cdr exp)))]

    [else (error "未知表达式类型" exp)]))

(define (my-apply procedure arguments)
  (cond
    [(primitive-procedure? procedure)
     (apply-primitive-procedure procedure arguments)]
    [(compound-procedure? procedure)
     (eval-sequence
       (procedure-body procedure)
       (extend-environment
         (procedure-parameters procedure)
         arguments
         (procedure-environment procedure)))]
    [else (error "未知过程类型" procedure)]))

;; 辅助函数
(define (tagged-list? exp tag)
  (and (pair? exp) (eq? (car exp) tag)))

(define (make-procedure params body env)
  (list 'procedure params body env))

(define (compound-procedure? proc)
  (tagged-list? proc 'procedure))

(define (procedure-parameters proc) (cadr proc))
(define (procedure-body proc) (caddr proc))
(define (procedure-environment proc) (cadddr proc))
```

## 环境模型

SICP 使用**环境模型**来描述求值过程。环境是帧的链表，每个帧包含变量绑定。

```scheme
;; 环境操作的实现
(define (extend-environment vars vals base-env)
  (if (= (length vars) (length vals))
      (cons (make-frame vars vals) base-env)
      (error "参数数量不匹配")))

(define (make-frame vars vals)
  (cons vars vals))

(define (frame-variables frame) (car frame))
(define (frame-values frame) (cdr frame))

(define (lookup-variable var env)
  (if (eq? env the-empty-environment)
      (error "未定义的变量" var)
      (let ([frame (first-frame env)])
        (let ([binding (assq var (frame-bindings frame))])
          (if binding
              (cdr binding)
              (lookup-variable var (enclosing-environment env)))))))

(define (define-variable! var val env)
  (let ([frame (first-frame env)])
    (let ([binding (assq var (frame-bindings frame))])
      (if binding
          (set-cdr! binding val)
          (add-binding-to-frame! var val frame)))))
```

## 赋值与变化的代价

SICP 用大量篇幅讨论可变状态带来的复杂性：

```scheme
;; 同一个输入，不同结果（赋值导致的问题）
(define (make-withdraw balance)
  (lambda (amount)
    (if (>= balance amount)
        (begin (set! balance (- balance amount))
               balance)
        "余额不足")))

(define W (make-withdraw 100))
(W 30)  ; => 70
(W 30)  ; => 40
;; 同样是 (W 30)，但结果不同！
;; 赋值打破了引用透明性

;; 引用透明的替代方案
(define (withdraw-stateless balance amount)
  (if (>= balance amount)
      (cons (- balance amount) (- balance amount))
      (cons balance "余额不足")))

(withdraw-stateless 100 30)  ; => (70 . 70)
(withdraw-stateless 70 30)   ; => (40 . 40)
;; 每次调用都需要传入当前状态，结果可预测
```

## SICP 的核心教义

1. **过程即数据**：过程不仅可以操作数据，过程本身也是数据
2. **抽象的力量**：通过层层抽象管理复杂性
3. **通用性与特殊性的平衡**：从特殊到一般，再从一般到特殊
4. **计算的本质**：什么是可计算的？什么是计算的过程？
5. **大系统组织**：模块化、数据抽象、元语言抽象
