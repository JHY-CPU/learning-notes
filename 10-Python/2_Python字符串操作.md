# Python字符串操作


## 🔤 Python 字符串操作


字符串方法 (split/join/replace/strip/find/index/count)、切片、格式化、正则入门。


## 字符串方法


```
// ========== 大小写转换 ==========
text = "Hello World"
text.upper()             # "HELLO WORLD"
text.lower()             # "hello world"
text.title()             # "Hello World"
text.capitalize()        # "Hello world"
text.swapcase()          # "hELLO wORLD"

// ========== 查找与统计 ==========
text = "hello world hello"
text.find("world")       # 6 (索引,找不到返回 -1)
text.index("world")      # 6 (找不到抛出 ValueError)
text.rfind("hello")      # 12 (从右找)
text.count("hello")      # 2
text.startswith("hello") # True
text.endswith("world")   # False

// ========== 判断方法 ==========
"hello".isalpha()        # True (全是字母)
"123".isdigit()          # True (全是数字)
"abc123".isalnum()       # True (字母数字)
"   ".isspace()          # True (全是空白)
"Hello".istitle()        # True (标题格式)
"hello".islower()        # True (全小写)
"HELLO".isupper()        # True (全大写)

// ========== 修整与填充 ==========
text = "  hello world  "
text.strip()             # "hello world" (去首尾空白)
text.lstrip()            # "hello world  " (去左)
text.rstrip()            #  "  hello world" (去右)

// 指定字符
"---hello---".strip("-") # "hello"

// 填充
"42".zfill(5)            # "00042" (前补零)
"hello".center(11)       # "   hello   "
"hello".ljust(10)        # "hello     "
"hello".rjust(10)        # "     hello"
```


## 分割与连接


```
// ========== split — 分割 ==========
text = "apple,banana,orange"
text.split(",")           # ["apple", "banana", "orange"]

"hello world".split()     # ["hello", "world"] (默认按空白)

"a,b,c,d".split(",", 2)  # ["a", "b", "c,d"] (最多 2 次)

// ========== join — 连接 ==========
items = ["apple", "banana", "orange"]
", ".join(items)          # "apple, banana, orange"

" ".join(items)           # "apple banana orange"
"".join(items)            # "applebananaorange"

// 常用于: 列表 → 字符串
lines = ["line1", "line2", "line3"]
"\n".join(lines)          # "line1\nline2\nline3"

// ========== partition — 分区 ==========
text = "hello:world"
text.partition(":")       # ("hello", ":", "world")
text.rpartition(":")      # 从右分区

// ========== splitlines — 分行 ==========
text = "line1\nline2\r\nline3"
text.splitlines()         # ["line1", "line2", "line3"]

// ========== 移除前缀/后缀 (Python 3.9+) ==========
"filename.txt".removeprefix("file")  # "name.txt"
"filename.txt".removesuffix(".txt")  # "filename"
```


## 字符串切片


```
// ========== 切片语法 ==========
// str[开始:结束:步长]
// 左闭右开: [开始, 结束)

text = "hello world"

text[0]                  # "h" (第一个字符)
text[-1]                 # "d" (最后一个字符)
text[0:5]                # "hello" (索引 0-4)
text[6:]                 # "world" (6 到末尾)
text[:5]                 # "hello" (开头到 4)
text[:]                  # "hello world" (全部,副本)
text[::2]                # "hlowrd" (步长 2)
text[::-1]               # "dlrow olleh" (反转)

// ========== 切片实战 ==========
url = "https://example.com"
url[8:]                  # "example.com" (去掉协议)
url.split("://")[1]      # "example.com"

email = "alice@example.com"
email[:email.index("@")]  # "alice" (用户名)
email[email.index("@")+1:] # "example.com" (域名)

filename = "data.csv"
filename[-4:]             # ".csv"
filename[:-4]             # "data"

// ========== 不可变性 ==========
// 字符串不可变,所有方法都返回新字符串
s = "hello"
s[0] = "H"               # TypeError! 不能修改
s = "H" + s[1:]          # "Hello" (创建新字符串)

// 大量拼接用列表 + join (效率高)
parts = []
for i in range(100):
    parts.append(str(i))
result = "".join(parts)   # 比 += 快得多
```


## 字符串格式化


```
// ========== % 格式化 (旧式) ==========
name = "Alice"
age = 25
"Name: %s, Age: %d" % (name, age)   # "Name: Alice, Age: 25"
"PI = %.2f" % 3.14159                # "PI = 3.14"

// ========== str.format() ==========
"Name: {}, Age: {}".format(name, age)
"Name: {1}, Age: {0}".format(age, name)  # 索引
"Name: {n}, Age: {a}".format(n=name, a=age)  # 命名
"PI = {:.2f}".format(3.14159)

// 对齐:
"{:<10}".format("left")    # 左对齐
"{:>10}".format("right")   # 右对齐
"{:^10}".format("center")  # 居中对齐

// ========== f-string (推荐) ==========
// Python 3.6+, 最简洁的方式
name = "Alice"
age = 25
print(f"Name: {name}, Age: {age}")
print(f"PI ≈ {3.14159:.2f}")
print(f"Hex: {255:x}")       # ff
print(f"Percent: {0.85:.0%}") # 85%

// 表达式:
print(f"Sum: {2 + 3}")
print(f"Upper: {name.upper()}")

// 多行 f-string:
query = f"""
SELECT * FROM users
WHERE name = '{name}'
AND age = {age}
"""

// ========== 转义字符 ==========
print("line1\nline2")      # 换行
print("tab\there")         # 制表符
print("backslash: \\")     # 反斜杠
print('single "quote"')   # 引号
print("it's fine")         # 引号

// 原始字符串 (路径/正则):
path = r"C:\Users\name"
regex = r"\d+\.\d+"
```


> **Note:** 💡 字符串处理原则: (1) f-string 是格式化的首选,简洁清晰; (2) 大量拼接用列表+join; (3) 字符串切片 [::-1] 是最简单的反转; (4) 用 str[开始:结束] 而不是 for 循环截取子串。


## 练习


<!-- Converted from: 2_Python字符串操作.html -->
