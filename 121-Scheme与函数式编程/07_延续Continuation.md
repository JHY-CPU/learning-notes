# 延续（Continuation）

## 什么是延续

延续（continuation）是 Scheme 最独特和强大的特性之一。**延续代表了"程序在当前点之后还要做什么"**。每个表达式在求值时都有一个延续，即接收该表达式结果的"后续计算"。

```scheme
;; 思考延续的直觉方式
;; 在表达式 (+ 1 (* 2 3)) 中，当 (* 2 3) 求值得到 6 时，
;; 它的延续就是: "将结果加上 1"，即 (lambda (v) (+ 1 v))

;; 在表达式 (f (g x)) 中，(g x) 的延续是 (lambda (v) (f v))

;; 在 if 表达式中:
(if (> x 0)
    (sqrt x)         ; sqrt 的延续：将结果作为 if 的值返回
    (error "负数"))  ; error 的延续：不会被调用（抛出异常）

;; 在 begin 表达式中:
(begin
  (display "hello")  ; display 的延续：执行下一条语句
  (newline)          ; newline 的延续：执行下一条语句
  42)                ; 42 的延续：将结果返回给 begin 的外部
```

## call/cc（call-with-current-continuation）

`call-with-current-continuation`（简称 `call/cc`）是 Scheme 中捕获当前延续的机制。它接受一个单参数函数，并将当前延续传递给它。

```scheme
;; call/cc 的基本用法
(call/cc
  (lambda (k)     ; k 就是"当前延续"
    (display "before\n")
    (k 42)        ; 用 42 调用延续，立即返回 42
    (display "after\n")))  ; 这行不会执行
; 输出: before
; => 42

;; k 就像一个"逃生出口"——调用它会立即跳出当前计算
(+ 1 (call/cc
        (lambda (k)
          (+ 10 (k 3)))))  ; => (+ 1 3) => 4
;; 不是 (+ 1 (+ 10 3)) => 14
;; 因为 (k 3) 直接跳到 call/cc 的外部，将 3 作为结果

;; call/cc 的一个经典例子：非局部返回
(define (find-element lst target)
  (call/cc
    (lambda (return)
      (for-each
        (lambda (x)
          (when (equal? x target)
            (return x)))    ; 找到后立即返回
        lst)
      #f)))  ; 未找到

(find-element '(1 2 3 4 5) 3)   ; => 3
(find-element '(1 2 3 4 5) 9)   ; => #f
```

## 延续作为一等对象

`call/cc` 使得延续成为一等对象，可以存储、传递、多次调用：

```scheme
;; 保存延续以供后续使用
(define saved-k #f)

(define (save-and-return)
  (call/cc
    (lambda (k)
      (set! saved-k k)
      (k "第一次调用"))))

(save-and-return)  ; => "第一次调用"
(saved-k "再次调用")  ; => "再次调用"
(saved-k "又一次")    ; => "又一次"

;; 延续可以多次调用，每次都会"跳回"到 call/cc 的位置

;; 延续可以存储在数据结构中
(define continuations '())

(define (capture-continuation)
  (call/cc
    (lambda (k)
      (set! continuations (cons k continuations))
      "已捕获")))

(define result1 (capture-continuation))  ; => "已捕获"
((car continuations) "从延续返回")        ; => "从延续返回"
```

## 延续传递风格（Continuation-Passing Style, CPS）

CPS 是一种编程风格，其中每个函数都接受一个额外的延续参数，而不是直接返回结果。

```scheme
;; 直接风格
(define (factorial n)
  (if (<= n 1)
      1
      (* n (factorial (- n 1)))))

;; CPS 风格
(define (factorial-cps n k)
  (if (<= n 1)
      (k 1)                    ; 直接调用延续，传递结果
      (factorial-cps (- n 1)
                     (lambda (result)
                       (k (* n result))))))

;; 调用 CPS 版本
(factorial-cps 5 (lambda (x) x))  ; => 120

;; CPS 的调用过程展开:
;; (factorial-cps 5 k0)
;; (factorial-cps 4 (lambda (r1) (k0 (* 5 r1))))
;; (factorial-cps 3 (lambda (r2) ((lambda (r1) (k0 (* 5 r1))) (* 4 r2))))
;; ...
;; 最终 k0 收到 120

;; CPS 风格的 map
(define (map-cps f lst k)
  (if (null? lst)
      (k '())
      (f (car lst)
         (lambda (head)
           (map-cps f (cdr lst)
                    (lambda (tail)
                      (k (cons head tail))))))))

;; 使用 CPS map
(map-cps (lambda (x k) (k (* x x)))
         '(1 2 3 4 5)
         (lambda (result) result))
; => (1 4 9 16 25)
```

## 逃逸延续（Escape Continuations）

逃逸延续允许从深层嵌套中直接跳出，类似于其他语言中的异常机制。

```scheme
;; 使用 call/cc 实现类似 break 的行为
(define (search-with-break lst target)
  (call/cc
    (lambda (break)
      (for-each
        (lambda (x)
          (when (equal? x target)
            (break x)))  ; 找到就跳出循环
        lst)
      "未找到")))

;; 使用 call/cc 实现类似 continue 的行为
(define (process-with-skip lst)
  (call/cc
    (lambda (outer-break)
      (for-each
        (lambda (x)
          (call/cc
            (lambda (continue)
              (when (negative? x)
                (continue))  ; 跳过负数
              (printf "处理: ~a~n" x))))
        lst))))

(process-with-skip '(1 -2 3 -4 5))
;; 输出: 处理: 1 处理: 3 处理: 5

;; 使用 call/cc 实现简单的异常系统
(define (try-catch body handler)
  (call/cc
    (lambda (k)
      (let ([old-handler (current-exception-handler)])
        (parameterize ([current-exception-handler
                        (lambda (e)
                          (k (handler e)))])
          (body))))))

(define (throw e)
  ((current-exception-handler) e))
```

## call/cc 的高级应用

### 协程

```scheme
;; 使用 call/cc 实现简单的协程
(define coroutine-queue '())
(define (coroutine-yield)
  (call/cc
    (lambda (k)
      (let ([next (car coroutine-queue)])
        (set! coroutine-queue (cdr coroutine-queue))
        (set! coroutine-queue
              (append coroutine-queue (list k)))
        (next #f)))))

(define (coroutine-run . thunks)
  (set! coroutine-queue
        (map (lambda (thunk)
               (lambda (_) (thunk)))
             thunks))
  ((car coroutine-queue) #f))
```

### 回溯搜索

```scheme
;; 使用 call/cc 实现非确定性计算
(define fail-stack '())

(define (amb choices)
  (call/cc
    (lambda (k)
      (set! fail-stack
            (cons (lambda ()
                    (k (amb (cdr choices))))
                  fail-stack))
      (k (car choices)))))

(define (fail)
  (if (null? fail-stack)
      (error "所有选择已穷尽")
      ((car fail-stack))))

;; 使用 amb 解决简单约束问题
(define (solve-puzzle)
  (let ([a (amb '(1 2 3 4 5 6 7 8 9))]
        [b (amb '(1 2 3 4 5 6 7 8 9))]
        [c (amb '(1 2 3 4 5 6 7 8 9))])
    (unless (= (+ (* a 100) (* b 10) c)
               (* a b c))
      (fail))
    (list a b c)))

;; (solve-puzzle) => 找到满足条件的三位数
```

## 延续的注意事项

- `call/cc` 的性能开销较大，因为它需要保存当前调用栈的完整状态
- 过度使用延续会使代码难以理解和维护
- 在生产代码中，更推荐使用异常处理机制而非手动的延续操控
- CPS 变换是编译器实现尾调用优化的重要技术手段
