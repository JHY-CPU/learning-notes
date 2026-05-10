# Python 设计模式实现


## 🏗️ Python 设计模式实现


Pythonic 单例、工厂、策略、观察者、装饰器模式。利用 Python 特性 (装饰器/上下文管理器/元类) 简化实现。


## 单例模式


```
// ========== 单例 (Singleton) ==========
# 确保一个类只有一个实例

# 方式 1: 模块单例 (Python 最简方式)
# Python 模块天然是单例!
# mysingleton.py:
class Database:
    def __init__(self):
        self.connection = None

db = Database()  # 模块导出,全局唯一

# 使用: from mysingleton import db

# 方式 2: __new__ 方法
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.value = 42

a = Singleton()
b = Singleton()
print(a is b)  # True

# 方式 3: 装饰器
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Config:
    def __init__(self):
        self.data = {}

# 方式 4: 元类 (最正统)
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connection = "connected"
```


## 工厂模式


```
// ========== 工厂 (Factory) ==========
from abc import ABC, abstractmethod

# 简单工厂
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "汪!"

class Cat(Animal):
    def speak(self):
        return "喵!"

def animal_factory(animal_type: str) -> Animal:
    factories = {
        "dog": Dog,
        "cat": Cat,
    }
    if animal_type not in factories:
        raise ValueError(f"未知动物: {animal_type}")
    return factories[animal_type]()

# 使用:
animal = animal_factory("dog")
print(animal.speak())  # "汪!"

# ========== 抽象工厂 ==========
class DatabaseFactory(ABC):
    @abstractmethod
    def create_connection(self): pass

    @abstractmethod
    def create_query(self): pass

class MySQLFactory(DatabaseFactory):
    def create_connection(self):
        return MySQLConnection()

    def create_query(self):
        return MySQLQuery()

class PostgreSQLFactory(DatabaseFactory):
    def create_connection(self):
        return PostgreSQLConnection()

    def create_query(self):
        return PostgreSQLQuery()

# ========== Callable 工厂 ==========
# Python 中类本身就是工厂
class UserFactory:
    def __call__(self, name: str, age: int):
        return {"name": name, "age": age}

factory = UserFactory()
user = factory("Alice", 30)
```


## 策略模式


```
// ========== 策略 (Strategy) ==========
# 策略模式: 定义算法族,可互换
# Python 中函数是一等公民,可以更简洁

# 传统实现:
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount): pass

class Alipay(PaymentStrategy):
    def pay(self, amount):
        return f"支付宝支付: {amount}元"

class WeChatPay(PaymentStrategy):
    def pay(self, amount):
        return f"微信支付: {amount}元"

class CreditCard(PaymentStrategy):
    def pay(self, amount):
        return f"信用卡支付: {amount}元"

class Order:
    def __init__(self, strategy: PaymentStrategy):
        self.strategy = strategy

    def checkout(self, amount):
        return self.strategy.pay(amount)

# 使用:
order = Order(Alipay())
print(order.checkout(100))

# ========== Pythonic 策略 ==========
# 直接用函数 (更简洁)

def alipay(amount):
    return f"支付宝: {amount}"

def wechat_pay(amount):
    return f"微信: {amount}"

class Order:
    def __init__(self, payment_func):
        self.payment_func = payment_func

    def checkout(self, amount):
        return self.payment_func(amount)

order = Order(alipay)
print(order.checkout(100))

# ========== 策略字典 ==========
strategies = {
    "alipay": alipay,
    "wechat": wechat_pay,
    "credit": lambda a: f"信用卡: {a}",
}

def checkout(method: str, amount: int):
    if method not in strategies:
        raise ValueError(f"不支持的支付方式: {method}")
    return strategies[method](amount)
```


## 观察者模式


```
// ========== 观察者 (Observer) ==========
# 观察者模式: 一对多通知

from abc import ABC, abstractmethod

# 方式 1: 传统 OOP
class Observer(ABC):
    @abstractmethod
    def update(self, message): pass

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer: Observer):
        self._observers.append(observer)

    def detach(self, observer: Observer):
        self._observers.remove(observer)

    def notify(self, message):
        for observer in self._observers:
            observer.update(message)

class EmailNotifier(Observer):
    def update(self, message):
        print(f"[邮件] {message}")

class SMSNotifier(Observer):
    def update(self, message):
        print(f"[短信] {message}")

# 使用:
subject = Subject()
subject.attach(EmailNotifier())
subject.attach(SMSNotifier())
subject.notify("订单已发货")

# ========== Pythonic 观察者 ==========
# 直接用回调函数
class EventEmitter:
    def __init__(self):
        self._handlers = []

    def on(self, handler):
        self._handlers.append(handler)

    def off(self, handler):
        self._handlers.remove(handler)

    def emit(self, *args, **kwargs):
        for handler in self._handlers:
            handler(*args, **kwargs)

# 使用:
emitter = EventEmitter()
emitter.on(lambda msg: print(f"处理器1: {msg}"))
emitter.on(lambda msg: print(f"处理器2: {msg}"))
emitter.emit("事件触发!")

# ========== @property 观察者 ==========
class Observable:
    def __init__(self):
        self._value = None
        self._callbacks = []

    def bind(self, callback):
        self._callbacks.append(callback)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        old = self._value
        self._value = new_value
        for cb in self._callbacks:
            cb(new_value, old)
```


## 装饰器模式


```
// ========== 装饰器 (Decorator) ==========
# Python 装饰器本身就是装饰器模式的体现!

from functools import wraps
import time

# 基础装饰器: 日志
def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用: {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"完成: {elapsed:.3f}s")
        return result
    return wrapper

@log
def process_data(n):
    return sum(range(n))

# ========== 类装饰器 ==========
# 作为类的装饰器 (不修改类)
def add_repr(cls):
    """自动添加 __repr__ 方法"""
    def __repr__(self):
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        return f"{cls.__name__}({attrs})"
    cls.__repr__ = __repr__
    return cls

@add_repr
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # Point(x=3, y=4)

# ========== 装饰器类 ==========
class CountCalls:
    """统计调用次数的装饰器类"""

    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"第 {self.count} 次调用")
        return self.func(*args, **kwargs)

@CountCalls
def hello():
    print("Hello!")

hello()  # 第 1 次调用
hello()  # 第 2 次调用
```


> **Note:** 💡 Python 设计模式: 模块天然单例; 函数可做策略/回调; 装饰器 @wraps; 上下文管理器 __enter__/__exit__; @property 观察者; Python 优先用函数而非类。


## 练习


<!-- Converted from: 132_Python 设计模式实现.html -->
