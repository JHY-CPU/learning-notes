# Python输入与输出


## 📥 Python 输入与输出


print() 函数详解、input() 获取用户输入、格式化输出、重定向输出。


## print() 函数详解


```
// ========== print() 基本用法 ==========
print("Hello, World!")    # 最基本的输出

// print() 可以接受多个参数
print("Hello", "World")   # "Hello World" (默认空格分隔)

// ========== sep 参数 — 分隔符 ==========
print("a", "b", "c")               # "a b c" (默认空格)
print("a", "b", "c", sep="-")      # "a-b-c"
print("a", "b", "c", sep="")       # "abc" (无分隔)
print("2024", "01", "15", sep="/") # "2024/01/15"

// ========== end 参数 — 结尾符 ==========
print("Hello")              # "Hello\n" (默认换行)
print("Hello", end="")      # "Hello" (不换行)
print("Hello", end="!!\n")  # "Hello!!" (自定义结尾)

// 常用: 逐行打印不换行
for i in range(5):
    print(i, end=" ")       # "0 1 2 3 4 "

// ========== file 参数 — 输出到文件 ==========
import sys
print("错误信息", file=sys.stderr)  # 输出到标准错误

with open("log.txt", "w") as f:
    print("写入文件", file=f)        # 输出到文件

// ========== flush 参数 — 立即刷新 ==========
print("请稍候...", flush=True)      # 立即显示，不缓存
// 常用于进度指示、长时间运行的任务
```


## input() 获取输入


```
// ========== input() 基本用法 ==========
name = input("请输入你的名字: ")
print(f"你好, {name}!")

// input() 总是返回字符串!
age = input("请输入年龄: ")
print(type(age))   #

// ========== 类型转换 ==========
// 必须手动转换输入为所需类型

age = int(input("请输入年龄: "))       # 转整数
price = float(input("请输入价格: "))   # 转浮点数
is_ok = input("是否继续? (y/n): ") == "y"  # 转布尔

// ========== 输入验证 ==========
// 基本验证:
while True:
    try:
        age = int(input("请输入年龄: "))
        if age < 0 or age > 150:
            print("年龄范围 0-150")
            continue
        break
    except ValueError:
        print("请输入有效数字!")

// ========== 多值输入 ==========
// 用 split() 分割
data = input("输入三个数 (空格分隔): ")
a, b, c = map(int, data.split())
print(f"和: {a + b + c}")

// ========== 密码输入 ==========
import getpass
password = getpass.getpass("输入密码: ")  # 不显示输入内容
print("密码已接收")
```


## 格式化输出


```
// ========== f-string (Python 3.6+) ==========
name, age, score = "Alice", 25, 92.5
print(f"Name: {name}, Age: {age}, Score: {score}")

// 表达式:
print(f"总和: {2 + 3}")           # "总和: 5"
print(f"大写: {name.upper()}")    # "大写: ALICE"

// 格式化数字:
pi = 3.1415926535
print(f"PI = {pi:.2f}")     # "PI = 3.14" (2位小数)
print(f"{pi:.0f}")          # "3" (整数)
print(f"{pi:010.2f}")       # "0000003.14" (前补零)

// 百分比:
rate = 0.856
print(f"进度: {rate:.1%}")  # "进度: 85.6%"

// 对齐:
s = "hello"
print(f"|{s:>10}|")   # "|     hello|" (右对齐)
print(f"|{s:<10}|")   # "|hello     |" (左对齐)
print(f"|{s:^10}|")   # "|  hello   |" (居中)

// ========== % 格式化 (旧式) ==========
"Name: %s, Age: %d" % ("Alice", 25)
"Price: %.2f" % 12.345           # "Price: 12.35"
"Hex: %x" % 255                  # "Hex: ff"

// ========== str.format() ==========
"Name: {}, Age: {}".format("Alice", 25)
"Name: {1}, Age: {0}".format(25, "Alice")  # 索引
"PI: {:.2f}".format(3.14159)

// ========== repr() vs str() ==========
print("hello")           # hello
print(repr("hello"))     # 'hello' (带引号)
print(repr(42))          # 42

// 自定义打印:
data = [1, 2, 3]
print(data)              # [1, 2, 3]
print(*data)             # 1 2 3 (解包)
print(*data, sep=",")    # 1,2,3
```


> **Note:** 💡 输入输出要点: (1) input() 返回字符串,需要手动类型转换; (2) print(sep='', end='', file=, flush=True) 参数很实用; (3) f-string 是格式化的首选,简洁高效; (4) 大量文本拼接用 join() 而非 +=; (5) 调试用 print(f"{var=}") 是 Python 3.8+ 的便捷语法。


## 重定向与高级输出


```
// ========== 输出重定向 ==========
// 命令行: python script.py > output.txt
// print() 默认输出到 stdout

import sys

// 临时重定向:
with open("out.txt", "w") as f:
    print("内容到文件", file=f)

// ========== sys.stdout 操作 ==========
sys.stdout.write("直接写入\n")   # 等价于 print
sys.stdout.flush()               # 强制刷新缓冲区

// 保存和恢复:
old_stdout = sys.stdout
sys.stdout = open("log.txt", "w")
print("这个会写入文件")
sys.stdout.close()
sys.stdout = old_stdout  # 恢复

// ========== 进度条效果 ==========
import time
for i in range(101):
    print(f"\r进度: {'#' * (i // 5)} {i}%", end="", flush=True)
    time.sleep(0.05)
print()  # 换行

// ========== 表格化输出 ==========
headers = ["Name", "Age", "City"]
rows = [
    ["Alice", 25, "Beijing"],
    ["Bob", 30, "Shanghai"],
    ["Charlie", 35, "Shenzhen"],
]

// 简单对齐:
for row in [headers] + rows:
    print(f"{row[0]:<10} {row[1]:<5} {row[2]:<10}")

// ========== Pretty Print ==========
import pprint
data = {"users": [{"name": "Alice", "scores": [95, 87, 92]}]}
pprint.pprint(data)  # 格式化打印复杂结构
```


## 练习


<!-- Converted from: 6_Python输入与输出.html -->
