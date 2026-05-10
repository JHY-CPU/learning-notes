# Python namedtuple与TypedDict


## 📋 Python namedtuple 与 TypedDict


namedtuple 定义与用法、namedtuple 方法详解、TypedDict 类型注解字典、SimpleNamespace、数据类选择指南 (dataclass vs namedtuple vs dict)。


## namedtuple 基础


```
// ========== namedtuple ==========
// 命名元组: 元组 + 字段名
// 可以像元组一样索引,也可以像类一样访问属性
// 不可变,比普通类更省内存

from collections import namedtuple

# 定义: namedtuple(类名, 字段名)
Point = namedtuple("Point", ["x", "y"])
# 或: namedtuple("Point", "x y")
# 或: namedtuple("Point", "x, y")

p = Point(3, 4)
print(p)                       # Point(x=3, y=4)
print(p.x)                     # 3 (名访问)
print(p[0])                    # 3 (索引访问)
print(p[1])                    # 4

x, y = p                       # 元组解包
print(x, y)                    # 3 4

// namedtuple = immutable + named fields
// 比普通类省内存,比元组好读

// ========== 实战: 坐标/记录 ==========
# CSV 数据行:
Employee = namedtuple("Employee", "name, age, department, salary")

e = Employee("Alice", 30, "Engineering", 120000)
print(e.name)                  # Alice
print(e.salary)                # 120000
print(e[2])                    # Engineering (索引)

// 遍历列表:
employees = [
    Employee("Bob", 25, "Sales", 80000),
    Employee("Charlie", 35, "Engineering", 150000),
]

for emp in employees:
    if emp.department == "Engineering":
        print(f"{emp.name}: {emp.salary}")
```


## namedtuple 方法详解


```
// ========== _asdict() ==========
// 转为 OrderedDict

p = Point(10, 20)
d = p._asdict()
print(d)                       # {'x': 10, 'y': 20}
print(d["x"])                  # 10

// ========== _replace() ==========
// 创建修改后的副本 (不可变对象)

p2 = p._replace(x=100)
print(p)                       # Point(x=10, y=20) (不变)
print(p2)                      # Point(x=100, y=20) (新对象)

// ========== _fields ==========
// 获取所有字段名

print(Point._fields)           # ('x', 'y')

Employee = namedtuple("Employee", "name, age, dept")
print(Employee._fields)        # ('name', 'age', 'dept')

// ========== _make() ==========
// 从可迭代对象创建

data = ["Alice", 30, "Eng"]
e = Employee._make(data)
print(e)                       # Employee(name='Alice', age=30, dept='Eng')

// 等价于:
e2 = Employee(*data)

// ========== ** 解包 ==========
// 从字典创建

d = {"name": "Bob", "age": 25, "dept": "Sales"}
e = Employee(**d)
print(e)                       # Employee(name='Bob', age=25, dept='Sales')

// ========== 继承 namedtuple ==========
// 添加方法

class Point(namedtuple("Point", ["x", "y"])):
    """带方法的命名元组"""
    __slots__ = ()             # 防止创建 __dict__

    @property
    def distance(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

p1 = Point(3, 4)
p2 = Point(1, 2)
print(p1.distance)             # 5.0
print(p1 + p2)                 # Point(x=4, y=6)

// ========== 默认值 ==========
// 设置字段默认值 (通过 __new__.__defaults__)

Person = namedtuple("Person", "name age gender")
Person.__new__.__defaults__ = ("unknown",)  # gender 的默认值

p = Person("Alice", 25)
print(p)                       # Person(name='Alice', age=25, gender='unknown')

// rename=True: 自动重名冲突字段
T = namedtuple("T", "class, name, def", rename=True)
print(T._fields)               # ('_0', 'name', '_2') (保留字被改名)
```


## TypedDict 与 SimpleNamespace


```
// ========== TypedDict (Python 3.8+) ==========
// 给字典加上类型注解
// 不影响运行时,只用于类型检查

from typing import TypedDict

class PersonDict(TypedDict):
    name: str
    age: int
    email: str

// 用法: 和普通字典一样
p: PersonDict = {"name": "Alice", "age": 25, "email": "alice@test.com"}
print(p["name"])               # Alice

// 可选字段:
class Config(TypedDict, total=False):
    host: str
    port: int
    debug: bool

c: Config = {"debug": True}    # 仅部分字段

// 混合:
class Employee(TypedDict, total=True):
    name: str
    age: int
    email: str = ""             # TypedDict 不支持默认值!

// ========== SimpleNamespace ==========
// 最简单的"属性容器"

from types import SimpleNamespace

obj = SimpleNamespace(x=10, y=20, name="test")
print(obj.x)                    # 10
print(obj.name)                 # test

# 可以随意添加/修改属性
obj.z = 30
obj.new_attr = "hello"

# 转字典:
d = vars(obj)
print(d)                        # {'x': 10, 'y': 20, 'name': 'test', 'z': 30, 'new_attr': 'hello'}

# 比较:
print(obj == SimpleNamespace(x=10, y=20, name="test", z=30, new_attr="hello"))  # True

// SimpleNamespace 适合:
// - 快速原型
// - 测试 mock
// - 不需要方法的简单数据
// 但比 dataclass/namedtuple 慢,类型提示差
```


## 数据类选择指南


```
// ========== 完整对比 ==========
// 特性        dict  namedtuple  dataclass  SimpleNamespace
// 可变性      ✅    ❌         ✅可选      ✅
// 属性访问    ❌    ✅         ✅         ✅
// 索引访问    ❌    ✅         ❌         ❌
// 可哈希      ❌    ✅         frozen时   ❌
// 省内存      ✅    ✅         ❌(默认)   ✅
// 类型提示    可选  ❌         ✅         ❌
// 方法        ❌   ✅ 继承     ✅         ❌
// 序列化JSON  ✅    _asdict()  asdict()   vars()
// 解包        ❌   ✅         ❌         ❌

// ========== 选择建议 ==========
// 用 dict:
//   - JSON 数据,临时数据
//   - 键为字符串的动态数据
//   - 不需要固定结构

// 用 namedtuple:
//   - 不可变数据 (坐标/记录)
//   - 需要解包
//   - 内存效率高
//   - 简单数据行

// 用 dataclass:
//   - 需要行为/方法的复杂数据
//   - 需要类型检查
//   - 需要可变性
//   - 需要验证 (__post_init__)

// 用 SimpleNamespace:
//   - 快速原型
//   - 测试 mock
//   - 临时数据对象

// ========== 性能对比 ==========
import sys
from dataclasses import dataclass

# dict
d = {"x": 1, "y": 2}
print("dict:", sys.getsizeof(d))     # ~232

# namedtuple
PointNT = namedtuple("PointNT", "x y")
nt = PointNT(1, 2)
print("namedtuple:", sys.getsizeof(nt))  # ~56 (和元组一样)

# dataclass
@dataclass
class PointDC:
    x: int
    y: int
dc = PointDC(1, 2)
print("dataclass:", sys.getsizeof(dc))   # ~48 (无 __dict__)

# 结论: namedtuple 和 dataclass __slots__ 最省内存
# dict 最灵活但最耗内存
```


> **Note:** 💡 namedtuple 要点: (1) namedtuple 是元组+字段名,不可变; (2) _asdict() 转字典,_replace() 创建副本,_fields 查看字段; (3) TypedDict 给字典加类型提示; (4) SimpleNamespace 是最简单的属性容器; (5) 选择: dict 灵活 > namedtuple 不可变 > dataclass 功能全 > SimpleNamespace 原型。


## 练习


<!-- Converted from: 57_Python namedtuple与TypedDict.html -->
