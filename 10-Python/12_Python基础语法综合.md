# Python基础语法综合


## 📝 Python 基础语法综合


Python 程序结构、语句分隔、代码块与缩进、pass 语句、基础语法总结。


## 程序结构与缩进


```
// ========== Python 程序结构 ==========
// Python 程序由模块组成
// 每个 .py 文件就是一个模块

// 典型结构:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""模块文档字符串。"""

import os
import sys

# 常量
MAX_RETRIES = 3

# 函数
def main():
    pass

# 入口
if __name__ == "__main__":
    main()

// ========== 缩进规则 ==========
// Python 用缩进表示代码块
// 同一代码块必须有相同缩进
// 标准: 4 个空格 (不要用 Tab!)

if True:
    print("缩进 4 空格")
    if True:
        print("嵌套再缩进 4 空格")
print("回到外层")  # 无缩进

// 常见错误:
//   IndentationError: unexpected indent  (多余缩进)
//   IndentationError: expected an indented block (缺少缩进)

// ========== 语句分隔 ==========
// 一般: 一行一条语句
print("Hello")
print("World")

// 多条语句写在一行 (不推荐):
print("Hello"); print("World")  # ❌ 不 Pythonic

// 一行太长: 用 \ 或括号换行
total = (1 + 2 + 3 +
         4 + 5 + 6)        # ✅ 括号隐式续行

long_function_name(arg1, arg2,
                   arg3, arg4)  # ✅ 函数参数换行

result = 1 + 2 + 3 + \
         4 + 5 + 6         # ✅ \ 显式续行

// ========== pass 语句 ==========
// pass 是空操作,用作占位符

if True:
    pass  # 暂时不实现

def todo_function():
    pass  # 函数体不能为空,用 pass 占位

class MyClass:
    pass  # 类定义不能为空

for i in range(10):
    pass  # 循环暂不实现
```


## 语句与表达式


```
// ========== 语句 vs 表达式 ==========
// 表达式: 有值,可求值
// 语句:   执行操作,无值

// 表达式:
42                      # 值 42
2 + 3                   # 值 5
"hello".upper()         # "HELLO"
[1, 2][0]               # 1

// 语句:
x = 42                  # 赋值语句
print("hi")             # 调用语句
if x > 0: pass          # 条件语句
for i in range(5): pass # 循环语句

// Python 中赋值是语句,不是表达式!
// 所以不能: if x = 42:  (不像 C/JavaScript)

// ========== 空行 ==========
// 空行提高可读性,但被解释器忽略

// 函数间空 2 行
def func1():
    pass

def func2():
    pass

// 类间空 2 行
class ClassA:
    pass

class ClassB:
    pass

// 方法间空 1 行
class MyClass:
    def method1(self):
        pass

    def method2(self):
        pass
```


## Python 关键字


```
// ========== Python 全部关键字 ==========
// 35 个 (Python 3.12)
// 不能用作变量名、函数名、类名!

import keyword
print(keyword.kwlist)
// ['False', 'None', 'True', 'and', 'as', 'assert',
//  'async', 'await', 'break', 'class', 'continue',
//  'def', 'del', 'elif', 'else', 'except',
//  'finally', 'for', 'from', 'global', 'if',
//  'import', 'in', 'is', 'lambda', 'nonlocal',
//  'not', 'or', 'pass', 'raise', 'return',
//  'try', 'while', 'with', 'yield']

// ========== 关键字分类 ==========
// 值:        False, None, True
// 运算符:    and, or, not, in, is
// 控制流:    if, elif, else, for, while, break,
//            continue, pass, match, case
// 函数:      def, return, lambda, yield
// 类:        class
// 异常:      try, except, finally, raise, assert
// 导入:      import, from, as
// 上下文:    with, as
// 变量:      global, nonlocal, del
// 异步:      async, await

// ========== 常见误区 ==========
// 不要用关键字做变量名!
class = "math"           # ❌ SyntaxError!
list = [1, 2, 3]         # ❌ 覆盖内置函数 (不是关键字但也不推荐)
str = 42                 # ❌ 覆盖内置类型

// 好的做法:
course = "math"          # ✅
items = [1, 2, 3]        # ✅
text = "hello"           # ✅
```


## Pythonic 代码示例


```
// ========== Pythonic 风格示例 ==========
// Pythonic = 符合 Python 设计哲学的写法

// 1. 变量交换
// 其他语言: temp = a; a = b; b = temp
a, b = b, a              # ✅ Pythonic

// 2. 遍历列表
// ❌ C 风格:
for i in range(len(fruits)):
    print(fruits[i])

// ✅ Pythonic:
for fruit in fruits:
    print(fruit)

// 3. 带索引遍历
for i, fruit in enumerate(fruits):
    print(i, fruit)

// 4. 判断空
if len(items) == 0:      # ❌
    pass
if not items:            # ✅
    pass

// 5. 字符串连接
result = ""
for s in list_of_strings:
    result += s          # ❌ 低效
result = "".join(list_of_strings)  # ✅ 高效

// 6. 存在性检查
if "apple" in fruits:    # ✅
    pass

// 7. 链式比较
if 0 < x < 10:           # ✅ Python 特有
    pass

// 8. 解包
name, age = "Alice", 25  # ✅
first, *rest = [1, 2, 3, 4]  # first=1, rest=[2,3,4]
```


> **Note:** 💡 Python 基础语法要点: (1) 缩进是语法的一部分,统一用 4 空格; (2) 语句按逻辑用空行分组; (3) 一行一条语句,不需要分号; (4) 不要用关键字做变量名; (5) 写 Pythonic 代码: 简洁、可读、符合习惯。


## 练习


<!-- Converted from: 12_Python基础语法综合.html -->
