# Python map-filter-reduce


## 🔄 Python map-filter-reduce


map() 映射、filter() 过滤、functools.reduce() 归约、与列表推导式对比、惰性求值。


## map() 函数


```
// ========== map() 基础 ==========
// map(func, iterable) — 将函数应用到每个元素上
// 返回 map 迭代器 (惰性)

// 基本:
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x ** 2, numbers)
print(list(squared))          # [1, 4, 9, 16, 25]

// ========== 多个可迭代参数 ==========
// map 可以接受多个可迭代对象,函数必须接收对应数量的参数

a = [1, 2, 3]
b = [10, 20, 30]
sums = map(lambda x, y: x + y, a, b)
print(list(sums))             # [11, 22, 33]

// ========== 字符串处理 ==========
names = ["  alice  ", "  BOB  ", "  CHARLIE  "]
cleaned = map(lambda s: s.strip().title(), names)
print(list(cleaned))          # ["Alice", "Bob", "Charlie"]

// ========== 类型转换 ==========
str_numbers = ["1", "2", "3", "4", "5"]
int_numbers = list(map(int, str_numbers))
print(int_numbers)            # [1, 2, 3, 4, 5]

// ========== map 是惰性的 ==========
// map 返回迭代器,不会立即计算所有元素
m = map(lambda x: x ** 2, range(1000000))
print(m)                      #
// 还没有计算! 只有 list() 时才计算
print(list(m)[:5])            # [0, 1, 4, 9, 16]
```


## filter() 函数


```
// ========== filter() 基础 ==========
// filter(func, iterable) — 过滤,保留使函数为 True 的元素
// func 返回 True 的元素被保留

numbers = [1, 2, 3, 4, 5, 6]
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))            # [2, 4, 6]

// ========== 过滤 None/空值 ==========
data = [0, "hello", "", None, [], [1, 2], False, "world"]
valid = filter(None, data)    # None 作为函数时,过滤掉 Falsy 值
print(list(valid))            # ["hello", [1, 2], "world"]

// 注意: 0, "", None, [], False 都被过滤了

// ========== 复杂过滤 ==========
students = [
    {"name": "Alice", "score": 95},
    {"name": "Bob", "score": 52},
    {"name": "Charlie", "score": 78},
    {"name": "David", "score": 63},
]

passed = filter(lambda s: s["score"] >= 60, students)
print(list(passed))
// [{"name": "Alice", ...}, {"name": "Charlie", ...}, {"name": "David", ...}]

// ========== filter 也是惰性的 ==========
f = filter(lambda x: x > 0, range(-5, 5))
print(list(f))                # [1, 2, 3, 4]

// ========== map + filter 组合 ==========
numbers = [1, 2, 3, 4, 5, 6]

// 先过滤偶数,再平方:
result = map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers))
print(list(result))           # [4, 16, 36]

// 等价列表推导式:
result = [x ** 2 for x in numbers if x % 2 == 0]
```


## reduce() 函数


```
// ========== reduce() 基础 ==========
// reduce(func, iterable) — 累积运算
// 在 functools 模块中

from functools import reduce

# 累加:
total = reduce(lambda a, b: a + b, [1, 2, 3, 4, 5])
print(total)                  # 15

# 执行过程:
# 1: a=1, b=2 → 3
# 2: a=3, b=3 → 6
# 3: a=6, b=4 → 10
# 4: a=10, b=5 → 15

// ========== 初始值 ==========
// reduce(func, iterable, initial)
// 指定初始值

total = reduce(lambda a, b: a + b, [1, 2, 3, 4, 5], 100)
print(total)                  # 115

// 初始值作为第一个参数传给 a
// 1: a=100, b=1 → 101
// 2: a=101, b=2 → 103
// ...

// ========== 实用案例 ==========
// 求阶乘:
factorial = reduce(lambda a, b: a * b, range(1, 6))
print(factorial)              # 120

// 找最大值:
max_value = reduce(lambda a, b: a if a > b else b, [3, 1, 4, 1, 5])
print(max_value)              # 5

// 拼接字符串:
words = ["Hello", " ", "World", "!"]
sentence = reduce(lambda a, b: a + b, words)
print(sentence)               # "Hello World!"

// ========== 常见替代 ==========
// Python 内置替代 reduce:
sum([1, 2, 3, 4, 5])         # 累加 (代替 reduce 加法)
max([3, 1, 4, 1, 5])         # 最大值
"".join(["Hello", " World"])  # 字符串拼接

// 大部分 reduce 场景都有更 Pythonic 的替代!
// reduce 在 Python 中已经不那么常见
```


## map/filter/reduce vs 推导式


```
// ========== 对比 ==========
numbers = [1, 2, 3, 4, 5, 6]

// map:
list(map(lambda x: x * 2, numbers))
[x * 2 for x in numbers]            # ✅ 更清晰

// filter:
list(filter(lambda x: x > 3, numbers))
[x for x in numbers if x > 3]       # ✅ 更清晰

// map + filter:
list(map(lambda x: x*2, filter(lambda x: x > 3, numbers)))
[x * 2 for x in numbers if x > 3]   # ✅ 更清晰

// ========== 什么时候用 map/filter ==========
// 1. 已有命名函数时,map 更简洁
list(map(str.upper, ["a", "b", "c"]))     # ✅ map
[s.upper() for s in ["a", "b", "c"]]      # 也可以

list(map(int, ["1", "2", "3"]))           # ✅ map
[int(s) for s in ["1", "2", "3"]]         # 也可以

// 2. 处理多个可迭代对象时,map 更方便
list(map(lambda x, y: x + y, [1, 2], [3, 4]))
// 推导式需要 zip:
[x + y for x, y in zip([1, 2], [3, 4])]

// 3. 惰性求值时
// map 和 filter 返回迭代器,适合大数据
large = map(expensive_func, huge_dataset)
result = filter(condition, large)
// 不会立即全部求值

// ========== 推荐选择 ==========
// ✅ 列表推导式: 清晰直观,首选
// ✅ map + 命名函数: 简洁
// ✅ filter(None, data): 过滤空值
// ❌ map + lambda: 推导式通常更好
// ❌ reduce: 大部分有内置替代
// ❌ 复杂的 map/filter/reduce 链: 不可读

// 一句话: 能用推导式就用推导式!
```


> **Note:** 💡 map/filter/reduce 要点: (1) map 和 filter 返回惰性迭代器; (2) 列表推导式通常比 map+lambda 更清晰; (3) map(函数, 多个可迭代对象) 可用于并行处理; (4) filter(None, data) 过滤 Falsy 值; (5) reduce 已降级为 functools 模块,大部分场景有替代。


## 练习


<!-- Converted from: 37_Python map-filter-reduce.html -->
