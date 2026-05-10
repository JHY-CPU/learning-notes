# Python抽象基类ABC


## 🔷 Python 抽象基类 ABC


abc 模块、ABCMeta、abstractmethod、抽象基类 vs 鸭子类型、register 虚拟子类、collections.abc 内置 ABC。


## 抽象基类基础


```
// ========== 什么是抽象基类 ==========
// 抽象基类 (ABC): 定义接口规范,不能直接实例化
// 子类必须实现所有抽象方法才能被实例化
// 用于"设计契约" — 约束子类的行为

from abc import ABC, abstractmethod

class Animal(ABC):               # 继承 ABC
    """抽象基类: 动物"""

    @abstractmethod
    def speak(self):             # 抽象方法,子类必须实现
        pass

    @abstractmethod
    def move(self):
        pass

    def eat(self):               # 具体方法,子类可以直接用
        return "吃东西"

// 不能实例化抽象类:
# a = Animal()                   # TypeError! Can't instantiate abstract class

class Dog(Animal):
    def speak(self):             # 必须实现
        return "汪汪!"

    def move(self):              # 必须实现
        return "跑"

dog = Dog()                      # ✅ 实现了所有抽象方法
print(dog.speak())               # 汪汪!
print(dog.eat())                 # 吃东西 (继承的具体方法)

// 如果没有实现所有抽象方法:
class Cat(Animal):
    def speak(self):
        return "喵~"
    # 没有实现 move()

# cat = Cat()                     # TypeError! Can't instantiate abstract class
```


## ABC 高级用法


```
// ========== @abstractproperty ==========
// Python 3.3+ 直接用 @abstractmethod + @property 组合

class Shape(ABC):
    @property
    @abstractmethod
    def area(self):
        """面积 (抽象属性)"""
        pass

    @property
    @abstractmethod
    def perimeter(self):
        """周长"""
        pass

class Circle(Shape):
    def __init__(self, radius):
        self._radius = radius

    @property
    def area(self):              # 必须实现抽象属性
        return 3.14 * self._radius ** 2

    @property
    def perimeter(self):
        return 2 * 3.14 * self._radius

c = Circle(5)
print(c.area)                    # 78.5

// ========== 抽象类方法/静态方法 ==========
class DataSource(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """从配置创建数据源"""
        pass

    @staticmethod
    @abstractmethod
    def validate(data):
        """验证数据"""
        pass

class FileSource(DataSource):
    @classmethod
    def from_config(cls, config):
        return cls(config["filepath"])

    @staticmethod
    def validate(data):
        return isinstance(data, str)

// ========== __init_subclass__ 钩子 ==========
class PluginBase(ABC):
    """插件基类: 自动注册子类"""
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @abstractmethod
    def run(self):
        pass

class EmailPlugin(PluginBase):
    def run(self):
        return "发送邮件"

class LogPlugin(PluginBase):
    def run(self):
        return "记录日志"

print(PluginBase._registry)
# {"EmailPlugin": <class EmailPlugin>, "LogPlugin": <class LogPlugin>}
```


## register 虚拟子类


```
// ========== 虚拟子类 ==========
// register() 可以将一个类注册为抽象基类的"虚拟子类"
// 虚拟子类不需要实现抽象方法!
// isinstance() 会返回 True,但实际没有继承关系

from abc import ABC

class Printable(ABC):
    """可打印接口"""
    @abstractmethod
    def print_me(self):
        pass

class MyClass:
    """普通类,没有继承 Printable"""
    def print_me(self):         # 有同样的方法,但没继承
        return "MyClass 打印"

Printable.register(MyClass)     # 注册为虚拟子类

obj = MyClass()
print(isinstance(obj, Printable))  # True! (虚拟子类)
print(issubclass(MyClass, Printable))  # True

// 但 MyClass 没有真的继承 Printable:
# print(obj.print_me())         # ✅ MyClass 自己的方法
# 但 Printable 的抽象方法检查不会发生

// ========== 虚子类的实际用途 ==========
// 让第三方类符合你的接口标准

# 假设第三方库的类:
class ThirdPartyList:
    def __len__(self):
        return 3
    def __getitem__(self, i):
        return i

# 注册为 Sequence 的虚拟子类:
from collections.abc import Sequence
Sequence.register(ThirdPartyList)

lst = ThirdPartyList()
print(isinstance(lst, Sequence))  # True
print(list(lst))                   # [0, 1, 2]

// ========== 通过 __subclasshook__ 自动注册 ==========
class DuckTyped(ABC):
    """鸭子类型检测: 有 quack 方法就是 DuckTyped"""
    @classmethod
    def __subclasshook__(cls, other):
        if cls is DuckTyped:
            if any("quack" in B.__dict__ for B in other.__mro__):
                return True
        return NotImplemented

class Duck:
    def quack(self):
        return "嘎!"

print(isinstance(Duck(), DuckTyped))  # True (自动检测!)
# 不需要 register,也不需要用继承!

// __subclasshook__ 让 isinstance() 自动检测协议
// 比 register 更灵活: 不需要显式注册
```


## collections.abc 内置 ABC


```
// ========== 常用内置 ABC ==========
// collections.abc 提供了丰富的抽象基类

from collections.abc import (
    Container, Iterable, Iterator,
    Sequence, MutableSequence,
    Set, MutableSet,
    Mapping, MutableMapping,
    Callable, Hashable,
    Sized,
)

// Sequence: 需要 __getitem__ + __len__
// MutableSequence: 还需要 __setitem__, __delitem__, insert
// Mapping: 需要 __getitem__, __len__, __iter__, __contains__
// Set: 需要 __contains__, __iter__, __len__

// ========== 检查内置类型 ==========
print(isinstance([], Sequence))     # True
print(isinstance({}, Mapping))      # True
print(isinstance("abc", Iterable))  # True
print(isinstance(42, Hashable))     # True
print(isinstance(print, Callable))  # True

// ========== 自定义序列 ==========
class MySequence(Sequence):
    """自定义序列 (必须实现 __getitem__ 和 __len__)"""
    def __init__(self, *items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    # Sequence 自动提供了:
    # __contains__, __iter__, __reversed__, count(), index()

s = MySequence(1, 2, 3, 4, 5)
print(len(s))                 # 5
print(3 in s)                 # True (自动获得)
print(s.index(4))             # 3 (自动获得)
for x in s:                   # 自动可迭代
    print(x, end=" ")         # 1 2 3 4 5

// ========== 自定义映射 ==========
class MyMapping(Mapping):
    def __init__(self, **kwargs):
        self._data = dict(kwargs)

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    # Mapping 自动提供: keys(), values(), items(), __contains__, get()
```


> **Note:** 💡 ABC 要点: (1) ABC 定义接口规范,子类必须实现 @abstractmethod; (2) 抽象类不能实例化; (3) register() 创建虚拟子类 (不继承但有 isinstance 关系); (4) __subclasshook__ 实现鸭子类型自动检测; (5) collections.abc 提供丰富内置 ABC。


## 练习


<!-- Converted from: 51_Python抽象基类ABC.html -->
