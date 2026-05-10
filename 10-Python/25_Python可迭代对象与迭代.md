# Python可迭代对象与迭代


## 🔄 Python 可迭代对象与迭代


可迭代对象、迭代器协议、解包运算符 * 和 **、iter()/next()、自定义可迭代类。


## 可迭代对象


```
// ========== 可迭代对象 ==========
// 可迭代对象: 可以用 for 循环遍历的对象
// 常见可迭代对象: 列表、元组、字典、集合、字符串、range、文件

// 可以 for 遍历的都是可迭代对象:
for x in [1, 2, 3]: pass      # 列表
for x in (1, 2, 3): pass      # 元组
for x in {1, 2, 3}: pass      # 集合
for x in {"a": 1}: pass       # 字典 (遍历键)
for x in "hello": pass        # 字符串
for x in range(10): pass      # range

// 检查是否可迭代:
def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

is_iterable([1, 2])           # True
is_iterable(42)               # False

// ========== 迭代器协议 ==========
// 可迭代对象: 实现了 __iter__() 方法
// 迭代器:     实现了 __iter__() 和 __next__() 方法

// for 循环的工作原理:
// 1. 调用 iter(obj) 获取迭代器
// 2. 反复调用 next(iterator) 获取元素
// 3. 遇到 StopIteration 异常时停止

my_list = [1, 2, 3]
iterator = iter(my_list)       # 获取迭代器
print(next(iterator))          # 1
print(next(iterator))          # 2
print(next(iterator))          # 3
// print(next(iterator))       # StopIteration! (没有元素了)
```


## 解包运算符 * 和 **


```
// ========== * 解包 (可迭代对象) ==========
// * 将可迭代对象"展开"为元素

// 列表解包:
first, *rest = [1, 2, 3, 4]     # first=1, rest=[2, 3, 4]
*beginning, last = [1, 2, 3, 4] # beginning=[1, 2, 3], last=4
first, *middle, last = [1, 2, 3, 4, 5]  # first=1, middle=[2,3,4], last=5

// 函数调用时解包:
def add(a, b, c):
    return a + b + c

nums = [1, 2, 3]
add(*nums)                     # 6 (等价于 add(1, 2, 3))

// 合并可迭代对象:
combined = [1, 2, *[3, 4], 5]   # [1, 2, 3, 4, 5]
combined = [*range(3), *"ab"]   # [0, 1, 2, "a", "b"]

// ========== ** 解包 (字典) ==========
// ** 将字典展开为键值对

def greet(name, age):
    return f"{name} is {age}"

user = {"name": "Alice", "age": 25}
greet(**user)                   # "Alice is 25" (等价于 greet(name="Alice", age=25))

// 合并字典:
d1 = {"a": 1, "b": 2}
d2 = {"c": 3}
merged = {**d1, **d2}           # {"a": 1, "b": 2, "c": 3}

// 覆盖值:
defaults = {"host": "localhost", "port": 80, "debug": False}
config = {**defaults, **{"port": 8080, "debug": True}}
// {"host": "localhost", "port": 8080, "debug": True}
```


## 实用迭代工具


```
// ========== enumerate — 带索引遍历 ==========
items = ["a", "b", "c"]

for i, item in enumerate(items):
    print(i, item)           # 0 a, 1 b, 2 c

// 指定起始索引:
for i, item in enumerate(items, start=1):
    print(i, item)           # 1 a, 2 b, 3 c

// ========== zip — 并行遍历 ==========
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["Beijing", "Shanghai", "Shenzhen"]

for name, age, city in zip(names, ages, cities):
    print(f"{name} ({age}) - {city}")

// zip 到列表/字典:
list(zip(names, ages))              # [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
dict(zip(names, ages))              # {"Alice": 25, "Bob": 30, "Charlie": 35}

// 不等长时,zip 以最短为准:
list(zip([1, 2, 3, 4], ["a", "b"]))  # [(1, "a"), (2, "b")]

// zip_longest 以最长为准:
from itertools import zip_longest
list(zip_longest([1, 2, 3], ["a", "b"], fillvalue="?"))
// [(1, "a"), (2, "b"), (3, "?")]

// ========== reversed — 反转 ==========
for x in reversed([1, 2, 3]):
    print(x)                 # 3, 2, 1

for x in reversed("hello"):
    print(x)                 # o, l, l, e, h

// ========== sorted — 排序 ==========
for x in sorted([3, 1, 4, 1, 5]):
    print(x)                 # 1, 1, 3, 4, 5

for x in sorted([3, 1, 4, 1, 5], reverse=True):
    print(x)                 # 5, 4, 3, 1, 1
```


## itertools 迭代工具


```
// ========== itertools 标准库 ==========
import itertools

// count: 无限计数
for i in itertools.count(10):  # 10, 11, 12, ...
    if i > 15: break

// cycle: 无限循环
for item in itertools.cycle(["A", "B"]):  # A, B, A, B, ...
    ...

// repeat: 重复
list(itertools.repeat(42, 3))     # [42, 42, 42]

// chain: 连接多个可迭代对象
list(itertools.chain([1, 2], [3, 4]))  # [1, 2, 3, 4]

// product: 笛卡尔积
list(itertools.product([1, 2], ["a", "b"]))
// [(1,"a"), (1,"b"), (2,"a"), (2,"b")]

// permutations: 排列
list(itertools.permutations([1, 2, 3], 2))
// [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]

// combinations: 组合
list(itertools.combinations([1, 2, 3], 2))
// [(1,2), (1,3), (2,3)]

// groupby: 分组
data = [("A", 1), ("A", 2), ("B", 3), ("B", 4)]
for key, group in itertools.groupby(data, key=lambda x: x[0]):
    print(key, list(group))
// A [(A,1), (A,2)]
// B [(B,3), (B,4)]

// islice: 切片迭代
list(itertools.islice(range(10), 2, 8, 2))  # [2, 4, 6]
```


> **Note:** 💡 迭代要点: (1) 能用 for 遍历的就是可迭代对象; (2) * 解包列表/元组,** 解包字典; (3) enumerate()+zip() 是最常用的迭代助手; (4) itertools 提供了强大的迭代工具; (5) 迭代器只能前进,不能回退。


## 练习


<!-- Converted from: 25_Python可迭代对象与迭代.html -->
