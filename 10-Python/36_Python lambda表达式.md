# Python lambda表达式


## λ Python lambda 表达式


lambda 语法、匿名函数、sort/key、与 def 对比、闭包、使用场景与限制。


## lambda 基础


```
// ========== lambda 语法 ==========
// lambda 参数: 表达式
// 创建小型匿名函数,只能包含一个表达式

// 基本:
square = lambda x: x ** 2
print(square(5))           # 25

// 多参数:
add = lambda a, b: a + b
print(add(3, 5))           # 8

// 无参数:
hi = lambda: "Hello!"
print(hi())                # Hello!

// ========== 等价 def ==========
// lambda:
square = lambda x: x ** 2

// def:
def square(x):
    return x ** 2

// lambda 更简洁,但功能有限
// lambda 只能写表达式,不能写语句

// ========== 表达式 vs 语句 ==========
// ✅ lambda 可以:
lambda x: x ** 2
lambda a, b: a if a > b else b
lambda s: s.strip().lower()

// ❌ lambda 不可以:
// lambda x: return x     # return 是语句
// lambda x: if x > 0: x  # if 是语句
// lambda x: print(x)     # print 是语句 (但 Python 3 中 print() 是函数,实际可以)
// 不过不推荐在 lambda 中用 print
```


## lambda 实用场景


```
// ========== 场景 1: sorted() 的 key ==========
items = ["apple", "Banana", "cherry", "Date"]

// 忽略大小写排序:
sorted(items, key=lambda x: x.lower())
// ["apple", "Banana", "cherry", "Date"]

// 按长度排序:
sorted(items, key=lambda x: len(x))
// ["Date", "apple", "cherry", "Banana"]

// 按最后一个字符:
sorted(items, key=lambda x: x[-1])

// ========== 场景 2: max()/min() 的 key ==========
students = [
    {"name": "Alice", "score": 95},
    {"name": "Bob", "score": 82},
    {"name": "Charlie", "score": 91},
]

best = max(students, key=lambda s: s["score"])
print(best["name"])        # Alice

worst = min(students, key=lambda s: s["score"])
print(worst["name"])       # Bob

// ========== 场景 3: 列表排序 ==========
pairs = [(1, "one"), (3, "three"), (2, "two")]

// 按第二个元素排序:
pairs.sort(key=lambda x: x[1])
print(pairs)               # [(1, "one"), (3, "three"), (2, "two")]

// 按第二个元素长度排序:
pairs.sort(key=lambda x: len(x[1]))

// ========== 场景 4: map/filter ==========
list(map(lambda x: x * 2, [1, 2, 3]))       # [2, 4, 6]
list(filter(lambda x: x > 2, [1, 2, 3, 4])) # [3, 4]

// 列表推导式通常更清晰:
[x * 2 for x in [1, 2, 3]]
[x for x in [1, 2, 3, 4] if x > 2]
```


## lambda 闭包与陷阱


```
// ========== lambda 闭包 ==========
// lambda 可以捕获外层变量

def make_multiplier(n):
    return lambda x: x * n

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))           # 10
print(triple(5))           # 15

// ========== 延迟绑定陷阱 ==========
// 常见陷阱: lambda 在循环中捕获变量

// ❌ 问题:
funcs = []
for i in range(3):
    funcs.append(lambda: i)

for f in funcs:
    print(f())             # 2, 2, 2 (都是 i=2)

// 原因: lambda 捕获的是 i 的引用,调用时才取值
// 循环结束后 i = 2

// ✅ 修复 1: 默认参数 (定义时绑定)
funcs = []
for i in range(3):
    funcs.append(lambda i=i: i)   # 默认参数在定义时求值

for f in funcs:
    print(f())             # 0, 1, 2

// ✅ 修复 2: 用函数工厂
def make_func(i):
    return lambda: i

funcs = [make_func(i) for i in range(3)]
for f in funcs:
    print(f())             # 0, 1, 2

// ========== lambda 和 def 的选择 ==========
// 用 lambda:
// - 简单转换,一次性使用
// - 作为 key 函数
// - 函数体只有一行表达式

// 用 def:
// - 需要多条语句
// - 需要文档字符串
// - 逻辑复杂
// - 需要重复使用
```


## lambda 综合示例


```
// ========== 实战: 数据处理 ==========
data = [
    {"name": "Alice", "age": 25, "score": 95},
    {"name": "Bob", "age": 30, "score": 82},
    {"name": "Charlie", "age": 22, "score": 91},
    {"name": "David", "age": 28, "score": 78},
]

// 按分数降序:
sorted(data, key=lambda x: x["score"], reverse=True)

// 筛选年龄 >= 25 的:
list(filter(lambda x: x["age"] >= 25, data))

// 提取名字:
list(map(lambda x: x["name"], data))

// 但列表推导式通常更清晰:
[x["name"] for x in data]
[x for x in data if x["age"] >= 25]
sorted(data, key=lambda x: x["score"], reverse=True)

// ========== 实战: 字符串处理 ==========
texts = ["  hello  ", "WORLD", "  Python  "]

// 清理文本:
cleaned = list(map(lambda t: t.strip().lower(), texts))
// ["hello", "world", "python"]

// 等价于:
cleaned = [t.strip().lower() for t in texts]

// ========== 实战: 条件选择 ==========
numbers = [1, 2, 3, 4, 5, 6]

// 分类:
even = list(filter(lambda x: x % 2 == 0, numbers))    # [2, 4, 6]
odd = list(filter(lambda x: x % 2 != 0, numbers))     # [1, 3, 5]

// 转换:
squared = list(map(lambda x: x ** 2, numbers))         # [1, 4, 9, 16, 25, 36]

// ========== 何时避免 lambda ==========
// ❌ 太复杂的 lambda (没人能看懂):
f = lambda x: (lambda y: (lambda z: x + y + z)(3))(2)

// ✅ 拆成 def:
def add_three(x):
    def add_y(y):
        def add_z(z):
            return x + y + z
        return add_z
    return add_y
// 其实也复杂,但可以命名和加文档

// 对于复杂逻辑,用普通函数!
```


> **Note:** 💡 lambda 要点: (1) lambda 是单表达式的匿名函数; (2) 主要用做 sorted/max/min 的 key; (3) 列表推导式通常比 map/filter+lambda 更清晰; (4) 注意循环中 lambda 的延迟绑定陷阱; (5) 复杂逻辑用 def,简单转换用 lambda。


## 练习


<!-- Converted from: 36_Python lambda表达式.html -->
