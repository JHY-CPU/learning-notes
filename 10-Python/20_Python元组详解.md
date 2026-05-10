# Python元组详解


## 📦 Python 元组详解


元组创建、不可变性、打包与解包、具名元组 namedtuple、元组 vs 列表。


## 元组基础


```
// ========== 创建元组 ==========
// 元组: 不可变的序列,用 () 表示

empty = ()                     # 空元组
single = (1,)                  # 单元素元组 (逗号不能少!)
single_wrong = (1)             # 1 (不是元组,是整数!)
pairs = (1, 2, 3)              # 三个元素的元组
mixed = (1, "hello", 3.14)     # 混合类型

// 省略括号:
nums = 1, 2, 3                 # (1, 2, 3) (逗号创建元组)
a = 1,                         # (1,) (单元素元组)

// tuple() 构造函数:
tuple([1, 2, 3])               # (1, 2, 3) (列表转元组)
tuple("hello")                 # ("h", "e", "l", "l", "o")
tuple(range(5))                # (0, 1, 2, 3, 4)

// ========== 访问元素 ==========
t = (10, 20, 30, 40, 50)
t[0]                          # 10
t[-1]                         # 50
t[1:3]                        # (20, 30) (切片返回新元组)
t[::-1]                       # (50, 40, 30, 20, 10) (反转)

// ========== 不可变性 ==========
t = (1, 2, 3)
// t[0] = 100                  # TypeError! ❌ 元组不可变
// t.append(4)                 # AttributeError! 没有 append 方法

// 但如果元组包含可变对象,该对象可以变:
t = (1, [2, 3], 4)
t[1].append(4)                # ✅ 列表是可变对象
print(t)                      # (1, [2, 3, 4], 4)
// ⚠️ 元组不可变,但元组中的列表可变!
```


## 打包与解包


```
// ========== 元组打包 ==========
// 多个值用逗号组合 = 元组打包
point = 3, 4                    # (3, 4) (打包成元组)
info = "Alice", 25, "Engineer"  # ("Alice", 25, "Engineer")

// ========== 元组解包 ==========
// 将元组拆分为多个变量
x, y = point                    # x=3, y=4
name, age, job = info           # name="Alice", age=25, job="Engineer"

// 交换变量:
a, b = 1, 2
a, b = b, a                    # a=2, b=1 (无需临时变量!)

// 星号解包 (Python 3):
first, *rest = (1, 2, 3, 4)     # first=1, rest=[2, 3, 4]
first, *middle, last = (1, 2, 3, 4)  # first=1, middle=[2,3], last=4

// ========== 函数多返回值 ==========
def min_max(items):
    """返回最小值和最大值"""
    return min(items), max(items)

result = min_max([3, 1, 4, 1, 5])  # result = (1, 5)
low, high = min_max([3, 1, 4, 1, 5])  # low=1, high=5

// ========== 使用 _ 忽略值 ==========
_, age, _ = ("Alice", 25, "Engineer")  # 只取 age
print(age)                              # 25
```


## 元组方法


```
// ========== 元组只有两个方法 ==========
// 元组方法比列表少 (因为不可变)
// count() 和 index() 是仅有的两个方法

t = (1, 2, 3, 2, 4, 2, 5)

t.count(2)                 # 3 (2 出现了 3 次)
t.index(3)                 # 2 (3 的索引)
t.index(2)                 # 1 (第一个 2 的索引)
t.index(2, 2)              # 3 (从索引 2 开始找 2)

// ========== 通用操作 ==========
t1 = (1, 2, 3)
t2 = (4, 5, 6)

t1 + t2                    # (1, 2, 3, 4, 5, 6) (连接)
t1 * 3                     # (1, 2, 3, 1, 2, 3, 1, 2, 3) (重复)

len(t1)                    # 3
3 in t1                    # True
10 in t1                   # False

sorted(t1)                 # 返回列表 (sorted 返回列表)
tuple(sorted(t1))          # 排完转回元组

// ========== 遍历元组 ==========
t = (10, 20, 30)

for value in t:
    print(value)

for i, value in enumerate(t):
    print(i, value)

// ========== 元组与列表转换 ==========
list((1, 2, 3))            # [1, 2, 3] (元组→列表)
tuple([1, 2, 3])           # (1, 2, 3) (列表→元组)
```


## namedtuple 具名元组


```
// ========== namedtuple ==========
// 带字段名的元组,兼具元组的轻量和类的可读性
// 来自 collections 标准库

from collections import namedtuple

// 定义:
Point = namedtuple("Point", ["x", "y"])
// 或: Point = namedtuple("Point", "x y")
// 或: Point = namedtuple("Point", "x, y")

// 创建实例:
p = Point(3, 4)
p2 = Point(x=10, y=20)

// 访问:
print(p.x)                 # 3 (通过字段名)
print(p[0])                # 3 (通过索引)
print(p.y)                 # 4
print(p[1])                # 4

// 解包:
x, y = p                   # 元组解包也支持
print(x, y)                # 3 4

// ========== 实际应用 ==========
Person = namedtuple("Person", ["name", "age", "email"])
alice = Person("Alice", 25, "alice@example.com")
print(f"{alice.name} is {alice.age}")  # "Alice is 25"

// ========== 方法 ==========
alice._asdict()            # {"name": "Alice", "age": 25, "email": "..."}
alice._replace(age=26)     # 返回新实例 (age 改为 26)
alice._fields              # ("name", "age", "email")

// ========== 元组 vs 列表 ==========
// 元组: 不可变,哈希,轻量,表示"固定结构"
// 列表: 可变,不可哈希,更多方法,表示"动态序列"

// 适用元组:
// - 函数多个返回值
// - 字典的键
// - 数据记录 (配合 namedtuple)
// - 不需要修改的数据集合

// 适用列表:
// - 需要增删改
// - 同类型数据集合
// - 排序/反转等操作
```


> **Note:** 💡 元组要点: (1) 单元素元组必须加逗号 (1,); (2) 逗号创建元组,括号可选; (3) 元组不可变,但如果包含可变对象(列表),该对象可以变; (4) 元组可作字典键,列表不行; (5) namedtuple 结合元组的轻量和类的可读性。


## 练习


<!-- Converted from: 20_Python元组详解.html -->
