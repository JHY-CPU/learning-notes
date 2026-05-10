# Python魔术方法基础


## ✨ Python 魔术方法基础


__str__ / __repr__、__len__ / __bool__、__eq__ / __hash__、__lt__ / __gt__ / 比较运算符重载、__add__ / 算术运算符重载。


## 字符串表示: __str__ 和 __repr__


```
// ========== __str__ vs __repr__ ==========
// __str__: 面向用户的可读字符串 (str(), print())
// __repr__: 面向开发者的"官方"表示 (repr(), 调试)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # 面向用户:
    def __str__(self):
        return f"({self.x}, {self.y})"

    # 面向开发者 (最好能重现对象):
    def __repr__(self):
        return f"Point({self.x!r}, {self.y!r})"

p = Point(3, 4)
print(str(p))                  # (3, 4)       ← 用户友好
print(repr(p))                 # Point(3, 4)  ← 可重现
print(p)                       # (3, 4)       ← print() 调用 __str__

// ========== 最佳实践 ==========
// 至少实现 __repr__, 方便调试
// 如果没有 __str__, print() 会 fallback 到 __repr__

class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"User({self.name!r}, {self.age})"

u = User("Alice", 25)
print(u)                       # User("Alice", 25) (自动用 __repr__)

// 理想 __repr__ 格式: ClassName(args...) 且 eval() 能重建
// __str__ 可以是更友好的格式
```


## 容器方法: __len__ 和 __bool__


```
// ========== __len__ ==========
// len(obj) 时调用,返回容器长度

class Team:
    def __init__(self, members):
        self.members = members

    def __len__(self):
        return len(self.members)

    def __repr__(self):
        return f"Team({self.members!r})"

t = Team(["Alice", "Bob", "Charlie"])
print(len(t))                  # 3

// ========== __bool__ ==========
// bool(obj) 时调用
// 如果没有 __bool__, 则回退到 __len__ (非零为 True)

class MyList:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        # 自定义逻辑: 空列表 + 包含 None 视为 False
        return len(self) > 0 and all(x is not None for x in self.items)

print(bool(MyList([1, 2])))    # True
print(bool(MyList([])))        # False (len=0)
print(bool(MyList([None])))    # False (包含 None)

// ========== 真值判断优先级 ==========
// 1. 有 __bool__ → 调用 __bool__
// 2. 有 __len__  → __len__ != 0
// 3. 都没有      → 总是 True (自定义对象默认 True)
```


## 比较方法: __eq__, __hash__, __lt__


```
// ========== __eq__ 相等 ==========
// == 运算符的行为

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented  # 让 Python 尝试 other 的 __eq__
        return self.name == other.name and self.age == other.age

p1 = Person("Alice", 25)
p2 = Person("Alice", 25)
p3 = Person("Bob", 30)

print(p1 == p2)                # True (内容相等)
print(p1 == p3)                # False
print(p1 == "not a person")    # False (返回 NotImplemented)

// ========== __hash__ 哈希 ==========
// 实现 __eq__ 的类应该也实现 __hash__
// 否则对象不能放入集合或作为字典键

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        if not isinstance(other, Person):
            return NotImplemented
        return self.name == other.name and self.age == other.age

    def __hash__(self):
        return hash((self.name, self.age))  # 用不可变元组

    def __repr__(self):
        return f"Person({self.name!r}, {self.age})"

p1 = Person("Alice", 25)
p2 = Person("Alice", 25)

s = {p1, p2}                  # 集合 (需要 __hash__)
print(len(s))                 # 1 (p1 == p2, 所以去重)

d = {p1: "用户A"}
print(d[p2])                  # "用户A" (p2 的哈希值和 p1 相同)

// 原则: 相等对象必须有相同哈希值
// 可变对象不应实现 __hash__ (或者将 __hash__ 设为 None)

// ========== 完整排序 ==========
# __lt__  (<), __le__ (<=), __gt__ (>), __ge__ (>=)

class Score:
    def __init__(self, value):
        self.value = value

    def __lt__(self, other):        # <
        return self.value < other.value

    def __le__(self, other):        # <=
        return self.value <= other.value

    # 可以用 functools.total_ordering 自动补全!

from functools import total_ordering

@total_ordering
class Score:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):        # 只需定义 __eq__ + __lt__
        return self.value < other.value

    # total_ordering 自动提供: __gt__, __le__, __ge__

s1, s2 = Score(80), Score(90)
print(s1 < s2)                 # True
print(s1 <= s2)                # True (自动生成)
print(s1 > s2)                 # False (自动生成)
```


## 算术运算符重载


```
// ========== 算术魔术方法 ==========
// __add__(+), __sub__(-), __mul__(*), __truediv__(/), __floordiv__(//), __mod__(%)

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):        # self + other
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):        # self - other
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):       # self * scalar
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):      # scalar * self (反向)
        return self.__mul__(scalar)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)                 # Vector(4, 6)
print(v1 - v2)                 # Vector(-2, -2)
print(v1 * 3)                  # Vector(3, 6)
print(3 * v1)                  # Vector(3, 6) (需要 __rmul__)

// ========== 反向运算符 ==========
// 当左操作数不支持运算时,Python 尝试右操作数的反向方法
// a + b: 先 a.__add__(b), 如果 NotImplemented, 则 b.__radd__(a)

// ========== 就地运算符 ==========
// __iadd__(+=), __isub__(-=) 等

class Cart:
    def __init__(self, items=None):
        self.items = items or []

    def __iadd__(self, item):        # cart += item
        self.items.append(item)
        return self                  # 必须返回 self!

    def __repr__(self):
        return f"Cart({self.items})"

cart = Cart()
cart += "苹果"
cart += "香蕉"
print(cart)                    # Cart(["苹果", "香蕉"])
```


> **Note:** 💡 魔术方法要点: (1) __str__ 给用户看,__repr__ 给开发者看; (2) __eq__ 判断相等,配合 __hash__ 用于集合/字典键; (3) @total_ordering 只需定义 __eq__+__lt__; (4) __add__ 等实现自定义运算,__radd__ 处理反向运算; (5) NotImplemented 常量让 Python 尝试另一侧的运算。


## 练习


<!-- Converted from: 52_Python魔术方法基础.html -->
