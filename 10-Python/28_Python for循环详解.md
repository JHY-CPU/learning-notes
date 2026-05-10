# Python for循环详解


## 🔄 Python for 循环详解


for 遍历各种类型、range() 详解、enumerate()、嵌套循环、循环中的 else。


## for 循环基础


```
// ========== 基本语法 ==========
// for 变量 in 可迭代对象:
//     循环体

// ========== 遍历列表 ==========
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)
// apple, banana, orange

// ========== 遍历字符串 ==========
for char in "hello":
    print(char)
// h, e, l, l, o

// ========== 遍历元组 ==========
for value in (1, 2, 3):
    print(value)
// 1, 2, 3

// ========== 遍历字典 ==========
user = {"name": "Alice", "age": 25}

for key in user:                    # 遍历键
    print(key)                      # name, age

for value in user.values():         # 遍历值
    print(value)                    # Alice, 25

for key, value in user.items():     # 遍历键值对
    print(f"{key}: {value}")        # name: Alice, age: 25

// ========== 遍历集合 ==========
for item in {"a", "b", "c"}:
    print(item)
// 顺序不确定 (集合无序)
```


## range() 函数


```
// ========== range() 三种形式 ==========
// range(stop)           — 0 到 stop-1
// range(start, stop)    — start 到 stop-1
// range(start, stop, step) — start 到 stop-1,步长 step

// ========== range(stop) ==========
for i in range(5):
    print(i)             # 0, 1, 2, 3, 4

// ========== range(start, stop) ==========
for i in range(2, 6):
    print(i)             # 2, 3, 4, 5

// ========== range(start, stop, step) ==========
for i in range(0, 10, 2):
    print(i)             # 0, 2, 4, 6, 8

// 倒序:
for i in range(5, 0, -1):
    print(i)             # 5, 4, 3, 2, 1

// ========== range 特性 ==========
// range 是惰性的 (不立即生成所有数字)
r = range(1000000)
len(r)                   # 1000000
r[0]                     # 0
r[-1]                    # 999999

// 检查:
10 in range(100)         # True

// 转列表:
list(range(5))           # [0, 1, 2, 3, 4]

// ========== 什么时候用 range? ==========
// ❌ 遍历列表:
for i in range(len(fruits)):  # C 风格,不推荐
    print(fruits[i])

// ✅ Pythonic:
for fruit in fruits:          # 直接遍历元素
    print(fruit)

// ✅ 需要索引时才用 range:
for i in range(len(fruits)):
    if i % 2 == 0:
        print(fruits[i])
```


## enumerate() 带索引遍历


```
// ========== enumerate() ==========
// 同时获取索引和值

fruits = ["apple", "banana", "orange"]

// ❌ C 风格:
for i in range(len(fruits)):
    print(f"{i}: {fruits[i]}")

// ✅ Pythonic:
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
// 0: apple, 1: banana, 2: orange

// 指定起始索引:
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}: {fruit}")
// 1: apple, 2: banana, 3: orange

// ========== 实际应用 ==========
// 找元素:
def find_index(items, target):
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

find_index(["a", "b", "c"], "b")  # 1

// 标记行号:
with open("file.txt") as f:
    for i, line in enumerate(f, 1):
        if "error" in line.lower():
            print(f"第{i}行: {line.strip()}")

// ========== 嵌套循环 ==========
for i in range(3):
    for j in range(3):
        print(f"({i}, {j})", end=" ")
    print()

// 输出:
// (0, 0) (0, 1) (0, 2)
// (1, 0) (1, 1) (1, 2)
// (2, 0) (2, 1) (2, 2)

// 乘法表:
for i in range(1, 10):
    for j in range(1, 10):
        print(f"{i}×{j}={i*j:2}", end="  ")
    print()
```


## 循环最佳实践


```
// ========== 直接遍历 vs range ==========
// ✅ 直接遍历元素:
for item in items:
    process(item)

// ✅ 需要索引用 enumerate:
for i, item in enumerate(items):
    print(i, item)

// ✅ 需要多个列表用 zip:
for name, age in zip(names, ages):
    print(f"{name} is {age}")

// ❌ 避免 C 风格:
for i in range(len(items)):
    process(items[i])

// ========== 遍历同时修改 ==========
// ❌ 遍历列表时删除元素:
items = [1, 2, 3, 4, 5]
for item in items:
    if item % 2 == 0:
        items.remove(item)   # 会跳过元素!
# 结果: [1, 3, 5]? 但实际可能 [1, 3, 4, 5]?

// ✅ 复制一份再遍历:
for item in items[:]:
    if item % 2 == 0:
        items.remove(item)

// ✅ 或用列表推导式:
items = [1, 2, 3, 4, 5]
items = [x for x in items if x % 2 != 0]

// ========== 循环中的解包 ==========
pairs = [(1, "a"), (2, "b"), (3, "c")]
for number, letter in pairs:
    print(f"{number} → {letter}")

// 星号解包:
records = [(1, "Alice", 25), (2, "Bob", 30)]
for id, *rest in records:
    print(f"ID {id}: {rest}")  # ID 1: ["Alice", 25]
```


> **Note:** 💡 for 循环要点: (1) for 直接遍历元素,需要索引用 enumerate; (2) range() 惰性求值,适合数字序列; (3) 并行遍历用 zip; (4) 不要在遍历列表时修改它; (5) 嵌套循环中 break 只退出最内层。


## 练习


<!-- Converted from: 28_Python for循环详解.html -->
