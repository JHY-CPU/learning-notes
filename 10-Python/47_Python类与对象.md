# Python类与对象


## 🏛️ Python 类与对象


class 定义、__init__ 构造方法与 self、实例变量 vs 类变量、实例方法/类方法/静态方法、@classmethod/@staticmethod、dunder 属性。


## 类的基础


```
// ========== 定义类 ==========
// class 关键字定义类
// __init__ 是构造方法,创建对象时自动调用

class Student:
    """学生类"""

    # 类变量 (所有实例共享)
    school = "Python 大学"

    def __init__(self, name, age):
        """初始化方法,创建实例时自动调用"""
        self.name = name       # 实例变量
        self.age = age

    def introduce(self):       # 实例方法
        return f"我叫{self.name}, 今年{self.age}岁"

// 创建实例:
alice = Student("Alice", 20)
bob = Student("Bob", 22)

print(alice.name)              # Alice
print(alice.introduce())       # 我叫Alice, 今年20岁

// ========== self 是什么? ==========
// self 就是实例本身
// Python 调用方法时自动传入实例作为第一个参数

alice.introduce()
// 等价于:
Student.introduce(alice)       # 手动传 self

// self 只是约定名称,可以叫别的
// 但强烈建议用 self!

// ========== __init__ 详解 ==========
// __init__ 不是真正的构造器
// 真正的构造是 __new__ (稍后讲)
// __init__ 用于初始化实例属性

class Point:
    def __init__(self, x=0, y=0):  # 可以有默认参数
        self.x = x
        self.y = y
        # __init__ 不能 return 非 None 值

p = Point()                  # 使用默认值
print(p.x, p.y)              # 0 0

p = Point(3, 4)              # 传参
print(p.x, p.y)              # 3 4
```


## 实例变量 vs 类变量


```
// ========== 类变量 ==========
// 定义在类中,方法外的变量
// 所有实例共享

class Dog:
    species = "Canine"       # 类变量
    count = 0                # 用于计数

    def __init__(self, name):
        self.name = name     # 实例变量
        Dog.count += 1       # 每创建一个实例+1

print(Dog.species)           # Canine
print(Dog.count)             # 0

d1 = Dog("Buddy")
d2 = Dog("Max")
print(Dog.count)             # 2 (两个实例)

// ========== 变量查找顺序 ==========
// 先找实例变量,再找类变量

class Cat:
    species = "Feline"
    def __init__(self, name):
        self.name = name

c = Cat("Kitty")
print(c.name)                # Kitty (实例变量)
print(c.species)             # Feline (类变量)

c.species = "特殊猫"         # 创建了实例变量,遮蔽类变量!
print(c.species)             # 特殊猫
print(Cat.species)           # Feline (类变量不变)

// ========== 注意事项 ==========
// 类变量用可变对象要小心!

class BadExample:
    items = []               # ❌ 类变量,所有实例共享

a = BadExample()
b = BadExample()
a.items.append("x")
print(b.items)               # ["x"] 被影响!

// ✅ 应该用实例变量:
class GoodExample:
    def __init__(self):
        self.items = []      # 每个实例独立

a = GoodExample()
b = GoodExample()
a.items.append("x")
print(b.items)               # [] 不受影响
```


## 实例/类/静态方法


```
// ========== 三种方法对比 ==========
class MyClass:
    # 1. 实例方法
    def instance_method(self):
        """需要 self,可以访问实例和类"""
        return f"实例方法: {self}"

    # 2. 类方法
    @classmethod
    def class_method(cls):
        """需要 cls,只能访问类,不能访问实例"""
        return f"类方法: {cls}"

    # 3. 静态方法
    @staticmethod
    def static_method():
        """不需要 self/cls,就是一个普通函数"""
        return "静态方法"

obj = MyClass()
print(obj.instance_method())  # 实例方法: ...
print(obj.class_method())     # 类方法: <class 'MyClass'>
print(obj.static_method())    # 静态方法

// ========== @classmethod 用途 ==========
// 1. 工厂方法: 用不同方式创建实例
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_birth_year(cls, name, birth_year):
        """用出生年份创建 Person"""
        age = 2026 - birth_year
        return cls(name, age)   # 用 cls() 创建实例

    @classmethod
    def from_string(cls, data):
        """从 "name,age" 字符串创建"""
        name, age = data.split(",")
        return cls(name.strip(), int(age))

p1 = Person.from_birth_year("Alice", 2000)
p2 = Person.from_string("Bob, 25")

print(p1.age)                 # 26
print(p2.name)                # Bob

// 2. 访问类变量
class Config:
    settings = {"debug": True}

    @classmethod
    def is_debug(cls):
        return cls.settings.get("debug")

// ========== @staticmethod 用途 ==========
// 工具函数,与类相关但不需要访问类或实例
class MathUtils:
    @staticmethod
    def is_even(n):
        return n % 2 == 0

    @staticmethod
    def validate_age(age):
        return 0 <= age <= 150

print(MathUtils.is_even(4))   # True
print(MathUtils.validate_age(200))  # False
```


## dunder 属性与特殊方法


```
// ========== 类的内置属性 ==========
class Example:
    """示例类"""
    x = 10
    def __init__(self):
        self.y = 20

obj = Example()

print(Example.__name__)       # "Example"
print(Example.__doc__)        # "示例类"
print(Example.__module__)     # "__main__" (或模块名)
print(Example.__dict__)       # 类的属性字典
print(obj.__dict__)           # 实例的属性字典 {"y": 20}
print(obj.__class__)          # <class 'Example'>

// ========== __dict__ 详解 ==========
// 类和实例都有 __dict__,存储属性和方法

class User:
    role = "user"             # 存在 User.__dict__

    def __init__(self, name):
        self.name = name      # 存在实例.__dict__

u = User("Alice")

# 实例属性:
print(u.__dict__)             # {"name": "Alice"}

# 类属性:
print(User.__dict__)
# {"__module__": "__main__", "role": "user", "__init__": <func>, ...}

// ========== type() 和 isinstance() ==========
print(type(u))                # <class 'User'>
print(type(u) is User)        # True
print(isinstance(u, User))    # True

// type() 检查精确类型
// isinstance() 检查继承关系 (更常用)
```


> **Note:** 💡 类与对象要点: (1) class 定义类,__init__ 初始化实例; (2) self 是实例本身,Python 自动传入; (3) 类变量所有实例共享,实例变量各自独立; (4) @classmethod 传 cls 用于工厂方法,@staticmethod 是普通工具函数; (5) __dict__ 存储属性和方法,isinstance 比 type 更灵活。


## 练习


<!-- Converted from: 47_Python类与对象.html -->
