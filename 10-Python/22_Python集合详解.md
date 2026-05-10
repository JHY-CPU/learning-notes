# Python集合详解


## ⛓️ Python 集合详解


集合创建、元素唯一性、增删操作、集合运算 (并/交/差/对称差)、frozenset 不可变集合。


## 集合基础


```
// ========== 创建集合 ==========
// 集合: 无序、不重复的元素序列

// 花括号:
fruits = {"apple", "banana", "orange", "apple"}  # {"apple", "banana", "orange"} (自动去重)

// set() 构造函数:
set([1, 2, 2, 3, 3, 3])         # {1, 2, 3} (列表去重)
set("hello")                     # {"h", "e", "l", "o"} (字符串拆成字符)
set(range(5))                    # {0, 1, 2, 3, 4}

// 空集合:
empty_set = set()                # {} (空集合)
// empty = {}                     # ❌ 这是空字典,不是空集合!

// ========== 集合特性 ==========
// 1. 无序: 不能通过索引访问
// 2. 不重复: 自动去重
// 3. 元素必须可哈希 (不可变)

s = {1, 2, 3}
// s[0]                          # TypeError! 集合不支持索引

// 元素类型:
{1, "hello", (1, 2)}            # ✅ 可哈希元素
// {[1, 2], {"a": 1}}            # TypeError! 列表/字典不可哈希

// ========== 去重常用技巧 ==========
names = ["Alice", "Bob", "Alice", "Charlie", "Bob"]
unique_names = list(set(names))        # 去重 (但顺序不确定)

// 保持顺序去重 (Python 3.7+):
unique_ordered = list(dict.fromkeys(names))  # ["Alice", "Bob", "Charlie"]
```


## 增删改查


```
// ========== 增加 ==========
s = {1, 2, 3}

s.add(4)                     # {1, 2, 3, 4}
s.add(2)                     # {1, 2, 3, 4} (已存在,不变)

s.update([5, 6, 7])          # {1, 2, 3, 4, 5, 6, 7} (批量添加)
s.update({8, 9}, [10])       # 可以多个参数

// ========== 删除 ==========
s = {1, 2, 3, 4, 5}

s.remove(3)                  # {1, 2, 4, 5} (不存在则 KeyError)
s.discard(10)                # 无变化 (不存在不报错)
s.discard(5)                 # {1, 2, 4}

s.pop()                      # 删除并返回任意一个元素 (空集合 KeyError)
s.clear()                    # 清空集合

// ========== 查找 ==========
s = {1, 2, 3, 4, 5}

2 in s                       # True (O(1) 哈希查找!)
10 in s                      # False

len(s)                       # 5

// ========== 子集与超集 ==========
a = {1, 2, 3}
b = {1, 2, 3, 4, 5}
c = {1, 2, 3}

a.issubset(b)                # True (a 是 b 的子集)
b.issuperset(a)              # True (b 是 a 的超集)
a.isdisjoint({4, 5})         # True (无交集)
a == c                       # True (相等)
```


## 集合运算


```
// ========== 集合运算 ==========
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

// 并集 (Union):
a | b                        # {1, 2, 3, 4, 5, 6}
a.union(b)                   # 同上

// 交集 (Intersection):
a & b                        # {3, 4}
a.intersection(b)            # 同上

// 差集 (Difference):
a - b                        # {1, 2} (在 a 不在 b)
b - a                        # {5, 6} (在 b 不在 a)
a.difference(b)              # 同上

// 对称差集 (Symmetric Difference):
a ^ b                        # {1, 2, 5, 6} (在任一集合但不同时在两者)
a.symmetric_difference(b)    # 同上

// ========== 修改集合 ==========
a = {1, 2, 3, 4}

a |= {4, 5, 6}               # 并集赋值: {1, 2, 3, 4, 5, 6}
a &= {1, 3, 5}               # 交集赋值: {1, 3, 5}
a -= {3}                     # 差集赋值: {1, 5}
a ^= {1, 7}                  # 对称差赋值: {5, 7}

// ========== 集合推导式 ==========
{x ** 2 for x in range(10)}              # {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}
{x for x in "hello world" if x != " "}   # {"h", "e", "l", "o", "w", "r", "d"}

// ========== 实用场景 ==========
// 1. 列表去重:
list(set([1, 2, 2, 3, 3, 3]))           # [1, 2, 3]

// 2. 共同元素:
friends_a = {"Alice", "Bob", "Charlie"}
friends_b = {"Bob", "David", "Eve"}
mutual = friends_a & friends_b            # {"Bob"}

// 3. 差异比较:
old = {"apple", "banana", "orange"}
new = {"apple", "banana", "grape"}
added = new - old                         # {"grape"}
removed = old - new                       # {"orange"}
```


## frozenset 不可变集合


```
// ========== frozenset ==========
// frozenset 是不可变版本
// 可哈希,可作为字典键或集合元素

fs = frozenset([1, 2, 3, 3, 4])  # frozenset({1, 2, 3, 4})

// frozenset 不可修改:
// fs.add(5)                  # AttributeError!
// fs.remove(1)               # AttributeError!

// 但支持集合运算:
fs1 = frozenset([1, 2, 3])
fs2 = frozenset([2, 3, 4])
fs1 | fs2                    # frozenset({1, 2, 3, 4})
fs1 & fs2                    # frozenset({2, 3})

// ========== frozenset 适用场景 ==========
// 1. 作为字典的键:
groups = {
    frozenset(["Alice", "Bob"]): "Team A",
    frozenset(["Charlie", "David"]): "Team B",
}

// 2. 作为集合的元素:
set_of_sets = {frozenset([1, 2]), frozenset([3, 4])}

// 3. 需要哈希但想要集合操作时

// ========== 集合选型 ==========
// set:        需要增删改,不需要哈希
// frozenset:  需要哈希(作字典键),不需要修改
// list:       需要顺序,允许重复
// tuple:      固定数据,需要哈希

// ========== 性能对比 ==========
// 成员检查 in:
//   列表: O(n)  (线性扫描)
//   集合: O(1)  (哈希表)

import time
data = list(range(1000000))
data_set = set(data)

// 500000 in data      # 慢 (O(n))
// 500000 in data_set  # 快 (O(1))

// 大量查找时优先用集合!
search_items = range(1000, 2000)
found = sum(1 for x in search_items if x in data_set)  # 快!
```


> **Note:** 💡 集合要点: (1) 自动去重是集合最实用的特性; (2) 集合运算 (| & - ^) 非常简洁; (3) in 检查 O(1),大量查找用集合; (4) 集合元素必须可哈希; (5) frozenset 可做字典键,set 不行。


## 练习


<!-- Converted from: 22_Python集合详解.html -->
