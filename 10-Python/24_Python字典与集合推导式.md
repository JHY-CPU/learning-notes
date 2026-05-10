# Python字典与集合推导式


## 📐 Python 字典与集合推导式


字典推导式语法、集合推导式、条件过滤、键值对转换、实用场景。


## 字典推导式


```
// ========== 字典推导式语法 ==========
// {键表达式: 值表达式 for 变量 in 可迭代对象 if 条件}

// 基本:
squares = {x: x ** 2 for x in range(6)}
// {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

// 相当于:
squares = {}
for x in range(6):
    squares[x] = x ** 2

// ========== 条件过滤 ==========
even_squares = {x: x ** 2 for x in range(10) if x % 2 == 0}
// {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

// ========== 从列表创建字典 ==========
keys = ["name", "age", "city"]
defaults = {k: "" for k in keys}
// {"name": "", "age": "", "city": ""}

users = ["Alice", "Bob", "Charlie"]
user_dict = {u: len(u) for u in users}
// {"Alice": 5, "Bob": 3, "Charlie": 8}
```


## 字典转换与处理


```
// ========== 键值反转 ==========
original = {"a": 1, "b": 2, "c": 3}
reversed_dict = {v: k for k, v in original.items()}
// {1: "a", 2: "b", 3: "c"}

// ⚠️ 值不唯一时会覆盖!
original = {"a": 1, "b": 1, "c": 2}
reversed_dict = {v: k for k, v in original.items()}
// {1: "b", 2: "c"} (1 被覆盖了!)

// ========== 大小写转换 ==========
user = {"Name": "Alice", "Age": 25, "City": "Beijing"}
lower = {k.lower(): v for k, v in user.items()}
// {"name": "Alice", "age": 25, "city": "Beijing"}

// ========== 条件过滤 ==========
scores = {"Alice": 95, "Bob": 72, "Charlie": 88, "David": 55}

// 筛选及格:
passed = {name: score for name, score in scores.items() if score >= 60}
// {"Alice": 95, "Bob": 72, "Charlie": 88}

// 分段标记:
graded = {name: "优秀" if s >= 90 else "良好" if s >= 80 else "及格" if s >= 60 else "不及格"
          for name, s in scores.items()}
// {"Alice": "优秀", "Bob": "及格", "Charlie": "良好", "David": "不及格"}

// ========== 值变换 ==========
prices = {"apple": 5, "banana": 3, "orange": 4}
with_tax = {item: price * 1.1 for item, price in prices.items()}
// {"apple": 5.5, "banana": 3.3, "orange": 4.4}

// ========== 从两个列表创建 ==========
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
user_map = {name: age for name, age in zip(names, ages)}
// {"Alice": 25, "Bob": 30, "Charlie": 35}
```


## 嵌套与复杂字典推导


```
// ========== 嵌套字典推导 ==========
// 生成乘法表:
mul_table = {i: {j: i * j for j in range(1, 6)} for i in range(1, 6)}
// {1: {1: 1, 2: 2, ...}, 2: {1: 2, 2: 4, ...}, ...}

// 相当于:
mul_table = {}
for i in range(1, 6):
    row = {}
    for j in range(1, 6):
        row[j] = i * j
    mul_table[i] = row

// ========== 分组 ==========
items = ["apple", "banana", "avocado", "blueberry", "cherry"]

// 按首字母分组:
from collections import defaultdict
grouped = defaultdict(list)
for item in items:
    grouped[item[0]].append(item)
// {"a": ["apple", "avocado"], "b": ["banana", "blueberry"], "c": ["cherry"]}

// 用字典推导式 + sorted 分组:
groups = {ch: [w for w in items if w.startswith(ch)]
          for ch in set(w[0] for w in items)}
// {"a": ["apple", "avocado"], "b": ["banana", "blueberry"], "c": ["cherry"]}

// ========== 过滤 None ==========
data = {"a": 1, "b": None, "c": 3, "d": None}
clean = {k: v for k, v in data.items() if v is not None}
// {"a": 1, "c": 3}

// ========== 枚举索引 ==========
items = ["apple", "banana", "cherry"]
indexed = {i: item for i, item in enumerate(items)}
// {0: "apple", 1: "banana", 2: "cherry"}
```


## 集合推导式


```
// ========== 集合推导式 ==========
// {表达式 for 变量 in 可迭代对象 if 条件}

// 基本:
squares = {x ** 2 for x in range(10)}
// {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}

// 等同于列表推导式 + set():
set([x ** 2 for x in range(10)])

// ========== 去重 ==========
names = ["Alice", "Bob", "Alice", "Charlie", "Bob"]
unique_lengths = {len(name) for name in names}
// {3, 5} (Bob=3, Alice/Charlie=5)

// ========== 过滤 ==========
numbers = range(-5, 6)
positive = {x for x in numbers if x > 0}
// {1, 2, 3, 4, 5}

// ========== 字符串处理 ==========
text = "hello world hello python"
unique_chars = {c for c in text if c != " "}
// {"h", "e", "l", "o", "w", "r", "d", "p", "y", "t", "n"}

// 统计不重复字符:
len({c for c in text if c.isalpha()})  # 不重复字母数

// ========== 三种推导式对比 ==========
data = [1, 2, 2, 3, 3, 3]

[x * 2 for x in data]        # 列表: [2, 4, 4, 6, 6, 6]
{x * 2 for x in data}        # 集合: {2, 4, 6} (去重)
{x: x * 2 for x in data}     # 字典: {1: 2, 2: 4, 3: 6}

// ========== 生成器表达式 (不是推导式) ==========
// 用 () 而不是 []
gen = (x ** 2 for x in range(10))  # 生成器对象 (惰性求值)
sum(gen)                           # 285 (只求值一次)

// 列表 vs 生成器:
list_comp = [x ** 2 for x in range(1000000)]  # 立即创建 100 万个元素
gen_expr = (x ** 2 for x in range(1000000))   # 几乎不占内存

// 大集合用生成器,小集合用列表推导式
```


> **Note:** 💡 推导式要点: (1) 字典推导式用 {} 但包含 : 键值对; (2) 集合推导式用 {} 但不含 :; (3) 三种推导式都支持 if 过滤; (4) 不要过度嵌套,保持可读性; (5) 大量数据用生成器表达式节省内存。


## 练习


<!-- Converted from: 24_Python字典与集合推导式.html -->
