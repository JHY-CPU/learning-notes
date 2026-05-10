# Python break-continue-pass


## ⏹️ Python break-continue-pass


break 退出循环、continue 跳过本次、pass 占位符、嵌套循环的 break、哨兵模式。


## break — 退出循环


```
// ========== break 基础 ==========
// break 立即退出当前循环
// 只退出最内层循环

for i in range(10):
    if i == 5:
        break
    print(i, end=" ")
// 0 1 2 3 4

// ========== 查找第一个匹配 ==========
numbers = [3, 7, 2, 8, 1, 9, 4]
first_even = None

for n in numbers:
    if n % 2 == 0:
        first_even = n
        break          # 找到第一个偶数就退出

print(first_even)      # 2

// ========== 提前退出优化 ==========
// 在大型列表中查找时,break 可以节省大量时间
def has_even(numbers):
    for n in numbers:
        if n % 2 == 0:
            return True
    return False

// 也可以直接用 any():
any(n % 2 == 0 for n in numbers)

// ========== 输入哨兵模式 ==========
// 直到输入特定值退出
total = 0
while True:
    value = input("输入数字 (输入 0 结束): ")
    try:
        num = float(value)
        if num == 0:
            break          # 哨兵值,退出循环
        total += num
    except ValueError:
        print("无效输入")

print(f"总和: {total}")
```


## continue — 跳过本次循环


```
// ========== continue 基础 ==========
// continue 跳过本次循环剩余部分,进入下一次

for i in range(10):
    if i % 2 == 0:
        continue          # 跳过偶数
    print(i, end=" ")
// 1 3 5 7 9

// ========== 过滤数据 ==========
data = ["apple", "", "banana", None, "cherry", "", "date"]
valid = []

for item in data:
    if not item:          # 跳过 None 和空字符串
        continue
    valid.append(item)

print(valid)
// ["apple", "banana", "cherry", "date"]

// ========== continue vs 嵌套 if ==========
// 两种写法效果一样:

// continue 方式:
for item in items:
    if not is_valid(item):
        continue
    if not has_permission(item):
        continue
    process(item)

// 嵌套 if 方式:
for item in items:
    if is_valid(item) and has_permission(item):
        process(item)

// 多层过滤时 continue 减少嵌套

// ========== while 中的 continue 注意 ==========
// while 中使用 continue 要小心!
// 确保循环变量在 continue 之前更新

// ❌ 错误:
i = 0
while i < 10:
    if i % 2 == 0:
        continue          # 跳过了 i += 1,无限循环!
    i += 1

// ✅ 正确:
i = 0
while i < 10:
    i += 1                # 先更新!
    if i % 2 == 0:
        continue
    print(i)
```


## pass 占位符


```
// ========== pass 基础 ==========
// pass 是空操作,什么都不做
// 用于语法上需要语句但逻辑上不需要的地方

// 占位函数:
def todo_function():
    pass                  # 稍后实现

// 占位类:
class TodoClass:
    pass                  # 稍后实现

// 条件分支占位:
if condition:
    pass                  # 暂时不处理

// ========== pass vs continue ==========
// pass: 占位,什么都不做
// continue: 跳过本次循环

for i in range(3):
    if i == 1:
        pass              # 继续执行下面的代码
    print(i, end=" ")
// 0 1 2

for i in range(3):
    if i == 1:
        continue          # 跳过本次
    print(i, end=" ")
// 0 2

// ========== pass 实用场景 ==========
// 1. 异常处理中忽略某些错误:
try:
    do_something()
except ValueError:
    pass                  # 忽略值错误
except Exception as e:
    print(f"其他错误: {e}")

// 2. 抽象基类的方法体:
class BaseHandler:
    def handle(self, data):
        pass              # 子类实现

// 3. 可以后续扩展的条件:
if something:
    handle_something()
elif something_else:
    pass                  # 暂不处理,先占位
else:
    handle_default()

// 4. 循环等待条件 (空循环):
while not condition():
    pass                  # 忙等待 (不推荐,用 time.sleep)
```


## 嵌套循环的 break


```
// ========== break 只退出最内层 ==========
for i in range(3):
    for j in range(3):
        if j == 1:
            break         # 只退出内层 j 循环
        print(f"({i},{j})", end=" ")
    print()
// (0,0) (1,0) (2,0)

// ========== 退出所有循环 ==========
// 方法 1: 使用标志变量
found = False
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            found = True
            break
        print(f"({i},{j})", end=" ")
    if found:
        break
// (0,0) (0,1) (0,2) (1,0)

// 方法 2: 使用 for-else (检查是否正常结束)
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            break
        print(f"({i},{j})", end=" ")
    else:
        continue          # 内层没有 break,继续外层
    break                 # 内层 break 了,外层也 break
// (0,0) (0,1) (0,2) (1,0)

// 方法 3: 用函数 + return (最清晰)
def find_element(matrix, target):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value == target:
                return (i, j)   # 直接返回,退出所有循环
    return None

matrix = [[1, 2], [3, 4]]
find_element(matrix, 3)   # (1, 0)

// ========== else 子句 ==========
// for-else: 循环正常结束时执行 else
// 如果 break 退出,则不执行 else

for n in range(2, 10):
    for i in range(2, n):
        if n % i == 0:
            print(f"{n} = {i} × {n//i}")
            break
    else:
        print(f"{n} 是质数")  # 只有内层没 break 才执行
```


> **Note:** 💡 break/continue/pass 要点: (1) break 退出整个循环,continue 只跳过本次; (2) break 只退出最内层循环; (3) 退出多重循环用标志变量或封装成函数; (4) pass 是占位符,continue 跳过本次循环; (5) while 中 continue 要确保循环变量在之前更新。


## 练习


<!-- Converted from: 30_Python break-continue-pass.html -->
