# Python递归函数


## 🔄 Python 递归函数


递归基础、基线条件、经典递归案例 (阶乘/斐波那契/汉诺塔)、递归限制、尾递归优化。


## 递归基础


```
// ========== 什么是递归 ==========
// 递归: 函数调用自身
// 两个关键要素:
// 1. 基线条件 (base case) — 停止递归的条件
// 2. 递归条件 (recursive case) — 函数调用自身

// ========== 阶乘 (n!) ==========
// n! = n × (n-1) × (n-2) × ... × 1
// 5! = 5 × 4 × 3 × 2 × 1 = 120

def factorial(n):
    # 基线条件
    if n <= 1:
        return 1
    # 递归条件
    return n * factorial(n - 1)

print(factorial(5))        # 120

// 执行过程:
// factorial(5) = 5 * factorial(4)
//             = 5 * 4 * factorial(3)
//             = 5 * 4 * 3 * factorial(2)
//             = 5 * 4 * 3 * 2 * factorial(1)
//             = 5 * 4 * 3 * 2 * 1
//             = 120

// ========== 递归 vs 循环 ==========
// 循环版本:
def factorial_loop(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

// 递归版本更接近数学定义
// 但循环版本效率更高
```


## 经典递归案例


```
// ========== 斐波那契数列 ==========
// F(0) = 0, F(1) = 1
// F(n) = F(n-1) + F(n-2)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))       # 55

// ⚠️ 指数级时间复杂度!
// fibonacci(30) 已经很慢了

// ========== 优化: 记忆化 ==========
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cache(n):
    if n <= 1:
        return n
    return fibonacci_cache(n - 1) + fibonacci_cache(n - 2)

print(fibonacci_cache(100))  # 354224848179261915075 (瞬间)

// ========== 汉诺塔 ==========
def hanoi(n, source, target, auxiliary):
    """将 n 个盘子从 source 移到 target"""
    if n == 1:
        print(f"移动盘子 1: {source} → {target}")
        return
    hanoi(n - 1, source, auxiliary, target)
    print(f"移动盘子 {n}: {source} → {target}")
    hanoi(n - 1, auxiliary, target, source)

hanoi(3, "A", "C", "B")
// A → C, A → B, C → B, A → C, B → A, B → C, A → C

// ========== 列表求和 ==========
def sum_list(lst):
    if not lst:
        return 0
    return lst[0] + sum_list(lst[1:])

print(sum_list([1, 2, 3, 4, 5]))  # 15

// ========== 展平嵌套列表 ==========
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

print(flatten([1, [2, [3, 4], 5], 6]))  # [1, 2, 3, 4, 5, 6]
```


## 递归限制与栈溢出


```
// ========== 递归深度限制 ==========
// Python 默认递归深度: 1000
// 超过会抛出 RecursionError

def infinite_recursion(n):
    return infinite_recursion(n + 1)

// infinite_recursion(0)
// RecursionError: maximum recursion depth exceeded

// ========== 查看/修改递归限制 ==========
import sys
print(sys.getrecursionlimit())   # 1000 (默认)

sys.setrecursionlimit(2000)      # 调高限制
print(sys.getrecursionlimit())   # 2000

// ⚠️ 调高限制可能导致栈溢出崩溃 (C 崩溃,不是 Python 异常)

// ========== 递归 vs 迭代 对比 ==========
// 递归:
// + 代码简洁,接近数学定义
// + 树/图遍历更自然
// - 有栈溢出风险
// - 性能较差 (函数调用开销)
// - 内存消耗大

// 迭代:
// + 无栈溢出风险
// + 性能好 (没有调用开销)
// + 内存效率高
// - 代码可能更复杂

// ========== 什么时候用递归 ==========
// ✅ 树的遍历 (目录结构、DOM)
// ✅ 分治算法 (快速排序、归并排序)
// ✅ 数学定义 (阶乘、斐波那契)
// ✅ 回溯算法 (八皇后、迷宫)

// ❌ 简单循环 (用 for/while)
// ❌ 深度不确定的大数据 (会栈溢出)
// ❌ 性能关键路径

// ========== 尾递归 ==========
// 尾递归: 递归调用是函数的最后一个操作
// 某些语言会优化尾递归 (重用栈帧)
// Python 不优化尾递归!

// 尾递归形式的阶乘:
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, acc * n)
// Python 中仍然会栈溢出,没有优化
```


> **Note:** 💡 递归要点: (1) 必须包含基线条件,否则无限递归; (2) Python 默认递归深度 1000; (3) 斐波那契用记忆化优化 (lru_cache); (4) 树形结构用递归最自然; (5) 简单循环用迭代,复杂分支用递归。


## 练习


<!-- Converted from: 38_Python递归函数.html -->
