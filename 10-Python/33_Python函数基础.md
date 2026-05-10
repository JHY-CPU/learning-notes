# Python函数基础


## 📝 Python 函数基础


def 定义函数、return 返回值、参数传递、类型提示、函数是对象、文档字符串。


## 函数定义与调用


```
// ========== 基本语法 ==========
def function_name(parameters):
    """文档字符串"""
    # 函数体
    return value

// ========== 最简单的函数 ==========
def greet():
    print("Hello!")

greet()                    # Hello!

// ========== 带参数的函数 ==========
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")             # Hello, Alice!

// ========== 带返回值的函数 ==========
def add(a, b):
    return a + b

result = add(3, 5)         # 8
print(result)

// ========== 多个返回值 ==========
def min_max(items):
    return min(items), max(items)

low, high = min_max([3, 1, 4, 1, 5])
print(low, high)           # 1 5

// ========== 默认返回值 None ==========
def no_return():
    pass                   # 没有 return

result = no_return()
print(result)              # None
print(type(result))        #
```


## return 详解


```
// ========== return 的作用 ==========
// 1. 返回值给调用者
// 2. 立即退出函数

// ========== 提前退出 ==========
def check_age(age):
    if age < 0:
        return "无效年龄"   # 提前返回
    if age < 18:
        return "未成年"
    return "已成年"

// ========== 无 return = 返回 None ==========
def print_message(msg):
    print(msg)
    # 没有 return,默认返回 None

result = print_message("hi")
print(result)              # None

// ========== return None 的简写 ==========
def find_user(uid):
    if uid in database:
        return database[uid]
    return                 # 等价于 return None

// ========== 返回多个值 ==========
// 实际上返回的是元组
def get_user():
    name = "Alice"
    age = 25
    return name, age       # 返回元组 ("Alice", 25)

user = get_user()
print(user)                # ("Alice", 25)
print(type(user))          #

// 解包接收:
name, age = get_user()
print(name)                # Alice

// ========== 返回 None vs 返回空 ==========
def func1():
    return                 # 返回 None

def func2():
    return None            # 同上,更明确

def func3():
    pass                   # 也是返回 None

// 三个函数都返回 None
```


## 参数与实参


```
// ========== 位置参数 ==========
// 按位置顺序传递

def describe(name, age, city):
    print(f"{name} is {age}, from {city}")

describe("Alice", 25, "Beijing")
// name="Alice", age=25, city="Beijing"

// ========== 关键字参数 ==========
// 不依赖位置,通过参数名传递

describe(city="Shanghai", name="Bob", age=30)
// 顺序不重要!

// ========== 混合使用 ==========
// 位置参数必须在前,关键字参数在后
describe("Charlie", 35, city="Shenzhen")  # ✅
// describe(city="Shenzhen", "Charlie", 35)  # ❌ SyntaxError

// ========== 默认参数 ==========
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Alice")             # Hello, Alice!
greet("Bob", "Hi")         # Hi, Bob!
greet("Charlie", greeting="Hey")  # Hey, Charlie!

// ========== ⚠️ 默认参数的陷阱 ==========
// 默认参数在函数定义时求值,只求值一次!

// ❌ 不要用可变对象做默认值:
def add_item(item, items=[]):  # [] 只创建一次!
    items.append(item)
    return items

print(add_item("a"))       # ["a"]
print(add_item("b"))       # ["a", "b"] ← 共享同一个列表!

// ✅ 用 None + 内部创建:
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item("a"))       # ["a"]
print(add_item("b"))       # ["b"] (新的列表)
```


## 函数对象与类型提示


```
// ========== 函数是一等公民 ==========
// 函数也是对象,可以赋值给变量、存入列表、作为参数

def square(x):
    return x ** 2

// 赋值:
f = square
print(f(5))                # 25

// 作为参数:
def apply(func, value):
    return func(value)

print(apply(square, 5))    # 25

// 作为返回值:
def get_operation(op):
    if op == "+":
        return lambda a, b: a + b
    elif op == "-":
        return lambda a, b: a - b

add_fn = get_operation("+")
print(add_fn(3, 5))        # 8

// ========== 类型提示 ==========
// Python 3.5+,不影响运行,帮助编辑器检查

def greet(name: str) -> str:
    return f"Hello, {name}"

def add(a: int, b: int) -> int:
    return a + b

from typing import Union, Optional, List

def process(items: List[int]) -> Optional[str]:
    if items:
        return str(sum(items))
    return None

// ========== 函数文档 ==========
def calculate(price: float, tax_rate: float = 0.1) -> float:
    """计算含税价格。

    Args:
        price: 原始价格
        tax_rate: 税率,默认 0.1

    Returns:
        含税价格
    """
    return price * (1 + tax_rate)

help(calculate)            # 显示格式化的文档
```


> **Note:** 💡 函数基础要点: (1) 没有 return 的函数返回 None; (2) 默认参数用 None 而非可变对象; (3) 函数是一等公民,可以赋值和传递; (4) 类型提示让代码更清晰; (5) 为公共函数写 docstring。


## 练习


<!-- Converted from: 33_Python函数基础.html -->
