# Python dataclass基础


## 📦 Python dataclass 基础


@dataclass 装饰器、自动生成 __init__/__repr__/__eq__ 等、类型注解、默认值、frozen 不可变、order 排序、__post_init__ 后处理。


## dataclass 入门


```
// ========== 什么是 dataclass ==========
// dataclass: 自动生成 __init__, __repr__, __eq__ 等的类
// 专门用于"数据容器" — 存储数据的类
// Python 3.7+ 引入,减少样板代码

from dataclasses import dataclass

// 普通类:
class PersonOld:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

    def __repr__(self):
        return f"PersonOld({self.name!r}, {self.age!r}, {self.email!r})"

    def __eq__(self, other):
        if not isinstance(other, PersonOld):
            return NotImplemented
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)

// dataclass: 自动生成上述所有!
@dataclass
class Person:
    name: str
    age: int
    email: str

p1 = Person("Alice", 25, "alice@example.com")
p2 = Person("Bob", 30, "bob@example.com")

print(p1)                     # Person(name='Alice', age=25, email='alice@example.com')
print(p1 == p2)               # False (自动生成 __eq__)
print(p1.name)                # Alice (可以直接访问)

// dataclass 自动生成:
// __init__    — 所有字段作为参数
// __repr__    — 类名+字段 格式
// __eq__      — 按字段比较
// __hash__    — 仅 frozen=True 时生成 (否则为 None)
// __delattr__ — 仅 frozen=False 时
```


## 字段默认值与类型


```
// ========== 默认值 ==========
// 支持类型注解和默认值

@dataclass
class Config:
    host: str = "localhost"      # 有默认值
    port: int = 8080
    debug: bool = False
    name: str = "default"        # 字符串默认值

c1 = Config()                    # 全部用默认值
c2 = Config(port=5432)           # 只覆盖 port

print(c1)                        # Config(host='localhost', port=8080, debug=False, name='default')
print(c2)                        # Config(host='localhost', port=5432, debug=False, name='default')

// ========== 可变默认值陷阱 ==========
// 和函数默认参数一样,不要用可变对象当默认值!

// ❌ 错误:
@dataclass
class Bad:
    items: list = []             # Warning: mutable default!

// ✅ 正确: 用 field(default_factory=...)
from dataclasses import field

@dataclass
class Good:
    items: list = field(default_factory=list)
    tags: set = field(default_factory=set)
    config: dict = field(default_factory=dict)

a = Good([1, 2])
b = Good([3])
print(a.items)                   # [1, 2]
print(b.items)                   # [3] (互不影响!)

// ========== 类型注解 ==========
// dataclass 利用类型注解定义字段
// 注解类型不强制,但推荐使用 (mypy 检查)
from typing import Optional, List, Dict

@dataclass
class User:
    id: int
    name: str
    email: Optional[str] = None      # 可选字段
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)

u = User(id=1, name="Alice")
print(u)                             # User(id=1, name='Alice', email=None, tags=[], meta={})
```


## frozen 与 order 参数


```
// ========== frozen=True (不可变) ==========
// 类似 namedtuple,创建后字段不能修改
// 可以当字典键 (生成 __hash__)

@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(3.0, 4.0)
print(p.x)                       # 3.0
# p.x = 5.0                     # FrozenInstanceError! 不能修改

// frozen 的好处:
// - 线程安全
// - 可哈希 (能做字典键)
// - 行为可预测

points = {Point(1,2): "A", Point(3,4): "B"}
print(points[Point(1,2)])        # A

// ========== order=True (排序) ==========
// 自动生成 __lt__, __le__, __gt__, __ge__
// 按字段定义顺序比较

from dataclasses import dataclass

@dataclass(order=True)
class Score:
    value: int
    name: str                     # name 也参与比较

scores = [
    Score(85, "Bob"),
    Score(92, "Alice"),
    Score(78, "Charlie"),
]

for s in sorted(scores):
    print(s)                      # 按 value 排序

print(Score(80, "A") < Score(90, "B"))  # True (先比 value)

// ========== 指定排序字段 ==========
@dataclass(order=True)
class Student:
    # sort_index 作为排序字段,不参与 __init__
    sort_index: int = field(init=False, repr=False)
    name: str
    grade: int
    age: int

    def __post_init__(self):
        self.sort_index = self.grade  # 按 grade 排序

s1 = Student("Alice", 85, 22)
s2 = Student("Bob", 92, 20)
print(s1 < s2)                   # True (85 < 92)
```


## __post_init__ 后处理


```
// ========== __post_init__ ==========
// __init__ 完成后自动调用
// 用于: 验证,派生字段,默认值转换

from dataclasses import dataclass
from typing import Optional

@dataclass
class Person:
    name: str
    age: int
    adult: bool = field(init=False)  # 不来自 __init__

    def __post_init__(self):
        # 派生字段
        self.adult = self.age >= 18
        # 验证
        if self.age < 0:
            raise ValueError("年龄不能为负")
        if not self.name.strip():
            raise ValueError("名字不能为空")

p = Person("Alice", 25)
print(p.adult)                   # True (自动计算)

// ========== 验证 + 标准化 ==========
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0

    def __post_init__(self):
        # 标准化
        self.name = self.name.strip().title()
        # 验证
        if self.price < 0:
            raise ValueError("价格不能为负")

p = Product("  python book  ", 39.9, 10)
print(p.name)                    # Python Book (自动标准化)

// ========== field(init=False) ==========
// 声明不来自 __init__ 的字段
// 通常在 __post_init__ 中计算

@dataclass
class Order:
    items: List[str] = field(default_factory=list)
    total: float = field(init=False, default=0.0)

    def __post_init__(self):
        # 可以从外部设置
        pass

// __post_init__ 常见用途:
// 1. 字段验证
// 2. 派生/计算字段
// 3. 数据标准化
// 4. 可变默认值的深拷贝
```


> **Note:** 💡 dataclass 要点: (1) @dataclass 自动生成 __init__/__repr__/__eq__; (2) field(default_factory=...) 替代可变默认值; (3) frozen=True 不可变 (可哈希); (4) order=True 自动排序; (5) __post_init__ 做验证和派生字段。


## 练习


<!-- Converted from: 55_Python dataclass基础.html -->
