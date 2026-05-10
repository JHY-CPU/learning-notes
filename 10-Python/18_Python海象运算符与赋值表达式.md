# Python海象运算符与赋值表达式


## 🦭 Python 海象运算符与赋值表达式


:= 语法 (Python 3.8+)、赋值与表达式结合、实用场景、常见争议。


## 海象运算符基础


```
// ========== := 海象运算符 ==========
// Python 3.8+ (PEP 572)
// 在表达式中同时完成赋值操作
// 名称来自 := 看起来像海象的眼睛和长牙

// 传统写法:
n = len(items)
if n > 10:
    print(f"太多项目了: {n}")

// 海象运算符写法:
if (n := len(items)) > 10:
    print(f"太多项目了: {n}")

// ========== 基本语法 ==========
// (variable := expression)
// 赋值并返回值,可以在表达式中使用

// 简单例子:
print(x := 5)            # 5 (先赋值,再打印)
print(x)                 # 5 (x 已被赋值)

// ========== while 循环中 ==========
// 传统:
data = input("输入: ")
while data != "quit":
    print(f"你输入了: {data}")
    data = input("输入: ")

// 海象运算符:
while (data := input("输入: ")) != "quit":
    print(f"你输入了: {data}")

// ========== 必须加括号的情景 ==========
// 海象运算符优先级最低 (仅高于 lambda)
// 在子表达式中使用时通常需要括号

// ❌ 错误:
if n := len(items) > 10:  # 等价于 n := (len(items) > 10), n 是布尔值!
    pass

// ✅ 正确:
if (n := len(items)) > 10:
    pass
```


## 实用场景


```
// ========== 场景 1: 正则表达式匹配 ==========
import re

text = "我的邮箱是 alice@example.com"

// 传统:
match = re.search(r"\w+@\w+\.\w+", text)
if match:
    email = match.group()
    print(f"找到邮箱: {email}")

// 海象运算符:
if (match := re.search(r"\w+@\w+\.\w+", text)):
    print(f"找到邮箱: {match.group()}")

// ========== 场景 2: 列表推导式中 ==========
// 传统:
results = []
for x in data:
    result = expensive_func(x)
    if result:
        results.append(result)

// 海象运算符:
results = [result for x in data if (result := expensive_func(x))]

// ========== 场景 3: 避免重复计算 ==========
// 传统:
total = sum(items)
discount = calculate_discount(total)
final = total - discount

// 海象运算符也可以,但不一定更好:
final = (total := sum(items)) - calculate_discount(total)

// 更好的做法是直接用传统赋值

// ========== 场景 4: 分块读取文件 ==========
// 传统:
while True:
    chunk = file.read(1024)
    if not chunk:
        break
    process(chunk)

// 海象运算符:
while (chunk := file.read(1024)):
    process(chunk)
```


## 更复杂的例子


```
// ========== 列表推导式中的过滤 ==========
// 计算开销大的过滤条件
def parse_value(x):
    """解析并返回处理后的值,无效则返回 None"""
    try:
        return int(x)
    except ValueError:
        return None

data = ["42", "abc", "7", "xyz", "99"]

// 传统:
parsed = []
for x in data:
    v = parse_value(x)
    if v:
        parsed.append(v)

// 海象:
parsed = [v for x in data if (v := parse_value(x))]

// ========== 多条件判断 ==========
// 传统:
age_input = input("输入年龄: ")
age = int(age_input) if age_input.isdigit() else 0
if age >= 18:
    print(f"年龄 {age}, 已成年")

// 海象:
if (age := int(age_input) if (age_input := input("输入年龄: ")).isdigit() else 0) >= 18:
    print(f"年龄 {age}, 已成年")
// ⚠️ 这太复杂了,不建议这样用!

// ========== 适度原则 ==========
// ✅ 好用法: 清晰,减少重复
if (match := re.search(pattern, text)):
    process(match.group())

// ✅ 好用法: while 循环简化
while (line := file.readline()):
    process(line)

// ❌ 滥用: 可读性下降
result = (x := a) + (y := b) + (x + y)  # 谁写的?!

// ❌ 滥用: 在简单表达式中
if (x := y > 0) and (z := process(x)):  # 没必要用海象
    pass
```


## 讨论与最佳实践


```
// ========== 支持与反对 ==========
// 支持:
// - 减少重复计算
// - 在 while 和推导式中非常有用
// - 让代码更紧凑

// 反对:
// - 降低可读性
// - 容易写出难以理解的代码
// - 不是必需的,只是语法糖

// ========== 最佳实践 ==========
// 1. 用海象运算符简化 while 循环
while (line := file.readline()):
    process(line)

// 2. 用海象运算符避免重复计算
if (result := expensive_function(data)):
    print(f"结果: {result}")

// 3. 推导式中过滤 + 转换
results = [v for x in items if (v := transform(x))]

// 4. 模式匹配 (正则)
if (match := re.search(pattern, text)):
    print(match.group())

// 5. 不要嵌套过度
// ❌
if (a := (b := c + d) > 0):
    pass

// ✅ 拆解
b = c + d
if (a := b) > 0:
    pass

// ========== 总结 ==========
// 海象运算符让 Python 更灵活
// 但可读性永远是第一位的
// 如果一行代码需要 3 秒才能看懂,那就拆开写
```


> **Note:** 💡 海象运算符要点: (1) Python 3.8+ 才支持; (2) 优先级很低,常在子表达式中需要括号; (3) while 循环和推导式是最佳使用场景; (4) 不要在简单赋值中使用,会降低可读性; (5) 如果一行代码变复杂了,拆开写。


## 练习


<!-- Converted from: 18_Python海象运算符与赋值表达式.html -->
