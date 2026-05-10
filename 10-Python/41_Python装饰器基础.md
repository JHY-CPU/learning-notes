# Python装饰器基础


## 🎨 Python 装饰器基础


装饰器语法 @decorator、函数装饰器、functools.wraps、常见装饰器（计时/日志/权限）、装饰器叠加顺序。


## 装饰器原理


```
// ========== 什么是装饰器 ==========
// 装饰器: 是一个函数,接收一个函数作为参数
// 返回一个新的函数,在不修改原函数代码的情况下增强功能

// 最简装饰器:
def my_decorator(func):
    def wrapper():
        print("调用前")
        func()               # 调用原函数
        print("调用后")
    return wrapper

def say_hello():
    print("Hello!")

# 手动调用装饰器:
say_hello = my_decorator(say_hello)
say_hello()

// 输出:
// 调用前
// Hello!
// 调用后

// ========== @ 语法糖 ==========
// Python 提供的简便写法:

def my_decorator(func):
    def wrapper():
        print("调用前")
        func()
        print("调用后")
    return wrapper

@my_decorator                # 等价于 say_hello = my_decorator(say_hello)
def say_hello():
    print("Hello!")

say_hello()                  # 调用的是 wrapper 函数

// @ 只是语法糖,本质就是函数调用
// @decorator 相当于 func = decorator(func)
```


## functools.wraps 保元数据


```
// ========== 问题: 元信息丢失 ==========
// 被装饰后,原函数的信息被 wrapper 覆盖了

def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def greet(name):
    """向某人打招呼"""
    return f"Hello, {name}"

print(greet.__name__)        # "wrapper"  (不是 "greet"!)
print(greet.__doc__)         # None       (文档丢失了!)

// ========== @wraps 修复 ==========
from functools import wraps

def decorator(func):
    @wraps(func)             # 从 func 复制 __name__, __doc__ 等属性
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def greet(name):
    """向某人打招呼"""
    return f"Hello, {name}"

print(greet.__name__)        # "greet" ✅
print(greet.__doc__)         # "向某人打招呼" ✅

// @wraps 还会复制:
// __module__, __qualname__, __annotations__
// __dict__ (wrapper 的), __wrapped__ (指向原函数)
// 始终使用 @wraps!
```


## 常用装饰器实战


```
// ========== 1. 计时器 ==========
import time
from functools import wraps

def timer(func):
    """打印函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} 耗时: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()  # slow_function 耗时: 1.00xx s

// ========== 2. 日志装饰器 ==========
def log_call(func):
    """记录函数调用信息"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"调用: {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"返回: {func.__name__} → {result!r}")
        return result
    return wrapper

@log_call
def add(a, b):
    return a + b

add(3, 5)
// 调用: add(3, 5)
// 返回: add → 8

// ========== 3. 权限检查 ==========
from functools import wraps

def require_admin(func):
    """检查用户是否为管理员"""
    @wraps(func)
    def wrapper(user, *args, **kwargs):
        if not user.get("is_admin"):
            raise PermissionError(f"用户 {user['name']} 不是管理员")
        return func(user, *args, **kwargs)
    return wrapper

@require_admin
def delete_user(admin, user_id):
    print(f"管理员 {admin['name']} 删除了用户 {user_id}")

// delete_user({"name": "Bob", "is_admin": False}, 42)
// PermissionError: 用户 Bob 不是管理员

delete_user({"name": "Alice", "is_admin": True}, 42)
// 管理员 Alice 删除了用户 42
```


## 多个装饰器叠加


```
// ========== 装饰器叠加顺序 ==========
// 叠加装饰器时,顺序很重要!
// @A 在最外层, @B 在中间, @C 最靠近函数

@decorator_a
@decorator_b
@decorator_c
def my_func():
    pass

// 等价于:
// my_func = decorator_a(decorator_b(decorator_c(my_func)))

// 执行顺序: 由外向内装饰,由内向外执行
// 装饰:  a(b(c(func)))   — a 最外层装饰
// 执行:  a → b → c → func → c → b → a

// ========== 实际例子 ==========
from functools import wraps

def bold(func):
    @wraps(func)
    def wrapper():
        return "<b>" + func() + "</b>"
    return wrapper

def italic(func):
    @wraps(func)
    def wrapper():
        return "<i>" + func() + "</i>"
    return wrapper

@bold           # 外层装饰器
@italic          # 内层装饰器
def hello():
    return "Hello"

print(hello())   # <b><i>Hello</i></b>
// bold(italic(hello))()
// italic(hello)() → "<i>Hello</i>"
// bold(...)       → "<b><i>Hello</i></b>"

// ========== 常见误区 ==========
// ❌ 错误的顺序:
@italic
@bold
def hello():
    return "Hello"

// 输出: <i><b>Hello</b></i> (顺序反了!)

// 记忆: 装饰器离函数越近,包装越"内层"
// 执行时,最内层的先执行
```


> **Note:** 💡 装饰器要点: (1) 装饰器是"函数加工厂",输入函数输出函数; (2) 始终使用 @wraps 保留原函数元信息; (3) 多个装饰器由下往上装饰,由上往下执行; (4) 适用场景: 日志/计时/权限/缓存/重试/事务。


## 练习


<!-- Converted from: 41_Python装饰器基础.html -->
