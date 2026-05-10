# Python魔术方法进阶


## 🚀 Python 魔术方法进阶


__call__ 可调用对象、__enter__ / __exit__ 上下文管理器、__iter__ / __next__ 迭代器、__getitem__ / __setitem__ 索引访问、__new__ 与元类基础、__slots__ 内存优化。


## __call__ 可调用对象


```
// ========== __call__ ==========
// 让对象像函数一样被调用

class Counter:
    """可调用的计数器"""
    def __init__(self, start=0):
        self.count = start

    def __call__(self):
        self.count += 1
        return self.count

    def reset(self):
        self.count = 0

counter = Counter(10)
print(counter())               # 11 (像函数一样调用!)
print(counter())               # 12
print(counter())               # 13
counter.reset()
print(counter())               # 1

// ========== __call__ 实际应用 ==========
// 1. 带状态的函数 (比闭包更清晰)
class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x + self.n

add5 = Adder(5)
print(add5(10))                # 15
print(add5(3))                 # 8

// 2. 可配置的装饰器
class Retry:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts

    def __call__(self, func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == self.max_attempts - 1:
                        raise
                    print(f"重试 {i+1}/{self.max_attempts}")
            return None
        return wrapper

@Retry(max_attempts=3)
def unstable():
    pass

// 3. 函数注册表
class Router:
    def __init__(self):
        self._routes = {}

    def route(self, path):
        def register(func):
            self._routes[path] = func
            return func
        return register

    def __call__(self, path):
        return self._routes.get(path)
```


## __enter__ / __exit__ 上下文管理器


```
// ========== 上下文管理器协议 ==========
// 实现 with 语句: __enter__ 进入, __exit__ 退出
// 无论是否异常,__exit__ 都会被调用

class ManagedFile:
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        """进入 with 块时调用"""
        print(f"打开文件: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file       # 绑定到 as 变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用 (总是执行)"""
        print(f"关闭文件: {self.filename}")
        self.file.close()
        # 返回 False 传播异常,返回 True 抑制异常
        if exc_type is ValueError:
            print(f"忽略 ValueError: {exc_val}")
            return True        # 抑制异常
        return False

with ManagedFile("test.txt", "w") as f:
    f.write("Hello!")
    # raise ValueError("测试")  # 会被 __exit__ 抑制

// ========== contextmanager 装饰器 ==========
// 用 yield 生成上下文管理器,更简洁!

from contextlib import contextmanager

@contextmanager
def managed_file(filename, mode="r"):
    """生成器版上下文管理器"""
    print(f"打开: {filename}")
    f = open(filename, mode)
    try:
        yield f                # 提供 as 变量
    finally:
        print(f"关闭: {filename}")
        f.close()

with managed_file("test.txt") as f:
    print(f.read())

// ========== 实际应用 ==========
// 1. 计时器
import time

@contextmanager
def timer(name="任务"):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} 耗时: {elapsed:.3f}s")

with timer("数据库查询"):
    time.sleep(0.5)

// 2. 临时重定向
import sys
from io import StringIO

@contextmanager
def capture_output():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout

with capture_output() as out:
    print("这段被捕获了")
output = out.getvalue()
```


## __getitem__ / __setitem__ 索引


```
// ========== __getitem__ / __setitem__ / __delitem__ ==========
// 实现 obj[key] 索引访问

class Dictionary:
    """支持键值访问的自定义字典"""
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):        # obj[key]
        print(f"读取: {key}")
        return self._data.get(key, None)

    def __setitem__(self, key, value): # obj[key] = value
        print(f"设置: {key} = {value}")
        self._data[key] = value

    def __delitem__(self, key):        # del obj[key]
        print(f"删除: {key}")
        del self._data[key]

    def __contains__(self, key):       # key in obj
        return key in self._data

    def __len__(self):
        return len(self._data)

d = Dictionary()
d["name"] = "Alice"           # 设置: name = Alice
d["age"] = 25                 # 设置: age = 25
print(d["name"])              # 读取: name → Alice
print("age" in d)             # True
print(len(d))                 # 2
del d["age"]                  # 删除: age

// ========== 切片支持 ==========
class ListLike:
    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        if isinstance(index, slice):   # 切片访问
            print(f"切片: {index}")
            return ListLike(self._items[index])
        print(f"索引: {index}")
        return self._items[index]

    def __repr__(self):
        return f"ListLike({self._items})"

l = ListLike([1, 2, 3, 4, 5])
print(l[0])                    # 索引: 0 → 1
print(l[1:3])                  # 切片: slice(1,3) → ListLike([2, 3])

// ========== __missing__ ==========
// 字典子类中,当 key 不存在时调用

class DefaultDict(dict):
    def __missing__(self, key):
        """当 dict[key] 找不到时调用"""
        return f"键 '{key}' 不存在"

d = DefaultDict({"a": 1})
print(d["a"])                  # 1
print(d["b"])                  # 键 'b' 不存在
// __missing__ 只在 __getitem__ 时调用
// d.get("b") 不会触发 __missing__
```


## __new__ 与元类基础


```
// ========== __new__ vs __init__ ==========
// __new__: 真正的构造器,创建实例 (静态方法)
// __init__: 初始化实例 (实例方法)
// __new__ 先调用,返回实例后 __init__ 才被调用

class Singleton:
    """单例模式: 始终返回同一个实例"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("创建唯一实例")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, value=None):
        if not hasattr(self, 'initialized'):
            self.value = value
            self.initialized = True

a = Singleton("first")
b = Singleton("second")
print(a is b)                  # True (同一个实例)
print(a.value)                 # first (只初始化一次)
print(b.value)                 # first (不是 second!)

// ========== __init_subclass__ ==========
// 当子类被创建时自动调用

class Base:
    _subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Base._subclasses.append(cls)
        print(f"新子类: {cls.__name__}")

class A(Base): pass             # 新子类: A
class B(Base): pass             # 新子类: B

print(Base._subclasses)         # [A, B]

// __init_subclass__ 常用于:
// - 自动注册子类 (插件系统)
// - 强制子类遵循某些规则
// - 配置继承行为

// ========== 元类简单介绍 ==========
// 元类是"类的类",控制类的创建行为
// type 是所有元类的基类

# 简单元类: 自动添加前缀
class PrefixMeta(type):
    def __new__(mcs, name, bases, namespace):
        # 给所有方法名加前缀
        new_namespace = {}
        for key, value in namespace.items():
            if callable(value) and not key.startswith("__"):
                new_namespace[f"my_{key}"] = value
            else:
                new_namespace[key] = value
        return super().__new__(mcs, name, bases, new_namespace)

class MyClass(metaclass=PrefixMeta):
    def hello(self):
        return "Hello"

obj = MyClass()
# obj.hello()                  # AttributeError!
print(obj.my_hello())           # Hello (自动加了 my_ 前缀)
```


> **Note:** 💡 进阶魔术方法要点: (1) __call__ 让对象可调用,适合带状态函数; (2) __enter__/__exit__ 实现 with 语句,可用 @contextmanager 简化; (3) __getitem__/__setitem__ 实现索引访问,支持切片; (4) __new__ 比 __init__ 先调用,用于单例/不可变对象; (5) 元类 type 控制类的创建,很少用但很强大。


## 练习


<!-- Converted from: 53_Python魔术方法进阶.html -->
