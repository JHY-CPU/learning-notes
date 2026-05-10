# Python装饰器进阶


## 🚀 Python 装饰器进阶


带参数的装饰器、类装饰器、@wraps 深入、装饰器常见模式（重试/缓存/单例/注册）、functools.partialmethod。


## 带参数的装饰器


```
// ========== 三层嵌套 ==========
// 带参数的装饰器需要三层函数:
// 1. 最外层接收参数
// 2. 中间层接收函数
// 3. 内层 wrapper 接收调用参数

def repeat(n):              # 第一层: 接收装饰器参数
    def decorator(func):    # 第二层: 接收被装饰函数
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):  # 第三层: 接收调用参数
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)                  # 先调用 repeat(3),再装饰
def greet(name):
    print(f"Hello, {name}")

greet("Alice")
// Hello, Alice
// Hello, Alice
// Hello, Alice

// ========== 带默认值的参数 ==========
def retry(max_attempts=3, delay=0):
    """带重试机制的装饰器"""
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise  # 最后一次,抛出异常
                    print(f"第 {attempt} 次失败: {e}")
                    if delay:
                        time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_api():
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络错误")
    return "成功!"

// ========== 可选参数的装饰器 ==========
// 让装饰器既支持 @decorator 也支持 @decorator(args)

def logged(func=None, *, level="INFO"):
    """支持 @logged 和 @logged(level="DEBUG") 两种用法"""
    from functools import wraps

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            print(f"[{level}] 调用 {f.__name__}")
            return f(*args, **kwargs)
        return wrapper

    if func is None:
        return decorator     # @logged(level="DEBUG") 用法
    return decorator(func)   # @logged 直接用法

@logged
def foo(): pass              # 无括号用法

@logged(level="DEBUG")
def bar(): pass              # 带参用法
```


## 类装饰器


```
// ========== 类作为装饰器 ==========
// 类实现 __call__ 方法,可以像函数一样调用

class Timer:
    """类装饰器: 计时"""
    def __init__(self, func):
        self.func = func
        from functools import wraps
        wraps(func)(self)     # 复制元信息到实例

    def __call__(self, *args, **kwargs):
        import time
        start = time.perf_counter()
        result = self.func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{self.func.__name__} 耗时: {elapsed:.4f}s")
        return result

@Timer
def slow():
    import time
    time.sleep(0.5)
    return "完成"

slow()  # slow 耗时: 0.50xxs

// ========== 带参数的类装饰器 ==========
class Retry:
    """类装饰器: 重试机制"""
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def __call__(self, func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, self.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_attempts:
                        raise
                    print(f"重试 {attempt}/{self.max_attempts}")
            return None
        return wrapper

@Retry(max_attempts=3)
def fetch_data():
    raise ConnectionError("超时")

// 类装饰器的优势:
// - 可以保存状态 (实例变量)
// - 可以添加额外方法
// - 适合需要维护状态的场景
```


## 常见装饰器模式


```
// ========== 1. 缓存装饰器 ==========
from functools import wraps

def cache(func):
    """简单的内存缓存"""
    memo = {}
    @wraps(func)
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper

@cache
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(100))  # 354224848179261915075 (瞬间)

// 也可以使用 functools.lru_cache / cache

// ========== 2. 单例模式 ==========
from functools import wraps

def singleton(cls):
    """确保类只有一个实例"""
    instances = {}
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("创建数据库连接")

db1 = Database()   # 创建数据库连接
db2 = Database()   # 不打印,复用实例
print(db1 is db2)  # True

// ========== 3. 注册表模式 ==========
# 自动注册函数到字典

plugins = {}

def register(name):
    """注册插件"""
    def decorator(func):
        plugins[name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@register("greet")
def say_hello(name):
    return f"Hello, {name}"

@register("farewell")
def say_goodbye(name):
    return f"Goodbye, {name}"

print(plugins)
# {"greet": <function say_hello>, "farewell": <function say_goodbye>}

// ========== 4. 属性验证 ==========
def validate_positive(func):
    """验证参数为正数"""
    @wraps(func)
    def wrapper(value):
        if value <= 0:
            raise ValueError(f"参数必须为正数, 得到 {value}")
        return func(value)
    return wrapper

@validate_positive
def sqrt(value):
    return value ** 0.5
```


## 装饰器技巧与陷阱


```
// ========== 技巧: functools.partialmethod ==========
// 用于类方法的偏函数
from functools import partialmethod

class Window:
    def resize(self, width, height):
        print(f"调整到 {width}x{height}")

    make_square = partialmethod(resize, height=100)  # 固定 height

w = Window()
w.make_square(200)  # 调整到 200x100

// ========== 陷阱: 装饰器在导入时执行 ==========
// 装饰器在模块导入时立即执行
// 不是在调用被装饰函数时才执行!

registry = {}

def register(func):
    print(f"注册: {func.__name__}")
    registry[func.__name__] = func
    return func

@register
def foo(): pass    # 导入时打印 "注册: foo"

// ========== 陷阱: 类方法上的装饰器 ==========
def method_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        print(f"调用 {func.__name__} 在 {self.__class__.__name__}")
        return func(self, *args, **kwargs)
    return wrapper

class MyClass:
    @method_decorator
    def method(self):
        pass

// 类装饰器要注意 self 参数
// wrapper 第一个参数必须是 self

// ========== wraps 源码近似 ==========
// @wraps 实际上做了:
WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__annotations__', '__doc__')
WRAPPER_UPDATES = ('__dict__',)

def my_wraps(wrapped):
    def decorator(wrapper):
        for attr in WRAPPER_ASSIGNMENTS:
            setattr(wrapper, attr, getattr(wrapped, attr, None))
        wrapper.__wrapped__ = wrapped  # 保留原函数引用
        return wrapper
    return decorator
```


> **Note:** 💡 装饰器进阶要点: (1) 带参装饰器需要三层嵌套; (2) 类装饰器通过 __call__ 实现; (3) @wraps 复制 __name__/__doc__ 等元信息; (4) 装饰器在导入时执行,谨慎使用副作用; (5) 缓存/重试/单例/注册是常见模式。


## 练习


<!-- Converted from: 42_Python装饰器进阶.html -->
