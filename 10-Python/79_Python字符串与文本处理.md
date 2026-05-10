# Python字符串与文本处理


## 📝 Python 字符串与文本处理


string 模块常量与模板、str 编码解码 encode/decode、textwrap 文本包装/缩进、difflib 文本比较、多行字符串与格式化技巧。


## string 模块常量


```
// ========== string 常量 ==========
import string

# 字符集常量:
print(string.ascii_lowercase)  # 'abcdefghijklmnopqrstuvwxyz'
print(string.ascii_uppercase)  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print(string.ascii_letters)    # 'abc...XYZ' (大小写合并)
print(string.digits)           # '0123456789'
print(string.hexdigits)        # '0123456789abcdefABCDEF'
print(string.octdigits)        # '01234567'
print(string.punctuation)      # '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
print(string.whitespace)       # ' \t\n\r\x0b\x0c'
print(string.printable)        # 所有可打印字符

# 实用: 密码字符集
import random
password_chars = string.ascii_letters + string.digits + "!@#$%"
password = ''.join(random.choices(password_chars, k=12))
print(password)                # 随机 12 位密码
```


## string 模板格式化


```
// ========== string.Template ==========
from string import Template

# 简单的字符串替换 (比 % 和 f-string 更安全)
t = Template("$name 今年 $age 岁")
msg = t.substitute(name="小明", age=18)
print(msg)                     # 小明今年 18 岁

# safe_substitute: 缺失变量不报错
t2 = Template("Hello $name, $title")
msg2 = t2.safe_substitute(name="World", title="")
print(msg2)                    # Hello World,

# $ 转义:
t3 = Template("$$ ${name}的余额")
print(t3.substitute(name="小明"))  # $ 小明的余额

# 适合: 用户提供的模板 (避免注入)
# 比 f-string 更安全
```


## 编码与解码


```
// ========== str.encode / bytes.decode ==========
# 字符串 → 字节:
text = "你好 Python"
utf8_bytes = text.encode("utf-8")
print(utf8_bytes)               # b'\xe4\xbd\xa0\xe5\xa5\xbd Python'
print(type(utf8_bytes))         #

# 不同编码:
gbk_bytes = text.encode("gbk")
print(gbk_bytes)                # b'\xc4\xe3\xba\xc3 Python'

# 字节 → 字符串:
decoded = utf8_bytes.decode("utf-8")
print(decoded)                  # '你好 Python'

# ========== 编码错误处理 ==========
# errors: 'strict' (默认) / 'ignore' / 'replace' / 'backslashreplace'
text_en = "Hello 你好"

# ignore: 忽略无法编码的字符
print(text_en.encode("ascii", errors="ignore"))
# b'Hello '

# replace: 用 ? 替换
print(text_en.encode("ascii", errors="replace"))
# b'Hello ??'

# backslashreplace: 用 \x 转义
print(text_en.encode("ascii", errors="backslashreplace"))
# b'Hello \\u4f60\\u597d'

# 解码错误处理:
broken = b"Hello \xff\xfe"
print(broken.decode("utf-8", errors="replace"))
# 'Hello ��'

# ========== bytes 与 bytearray ==========
b = b"hello"                     # 不可变
ba = bytearray(b"hello")         # 可变
ba[0] = 72                       # 修改
print(ba.decode())               # 'Hello'
```


## textwrap 文本包装


```
// ========== textwrap 模块 ==========
import textwrap

text = "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。由 Guido van Rossum 于 1989 年底发明。"

# ========== wrap: 按宽度换行 ==========
lines = textwrap.wrap(text, width=20)
for line in lines:
    print(line)
# Python 是一种解释型、面
# 向对象、动态数据类型的
# 高级程序设计语言。由
# Guido van Rossum 于
# 1989 年底发明。

# ========== fill: 直接返回字符串 ==========
wrapped = textwrap.fill(text, width=30)
print(wrapped)
# Python 是一种解释型、面向对
# 象、动态数据类型的高级程序设
# 计语言。由 Guido van Rossum...

# ========== shorten: 截断文本 ==========
short = textwrap.shorten(text, width=30, placeholder="...")
print(short)
# Python 是一种解释型、面向对象、...

# ========== dedent / indent ==========
# 移除公共缩进:
code = """
    def hello():
        print("world")
"""
clean = textwrap.dedent(code)
print(clean)
# def hello():
#     print("world")

# 添加缩进:
indented = textwrap.indent("line1\nline2\n", ">> ")
print(indented)
# >> line1
# >> line2
```


## difflib 文本比较


```
// ========== difflib 模块 ==========
import difflib

# 字符串相似度:
a = "Python 编程"
b = "Python 程序"

ratio = difflib.SequenceMatcher(None, a, b).ratio()
print(f"相似度: {ratio:.2%}")    # 相似度: 77.78%

# 找出差异:
diff = difflib.unified_diff(
    ["line1\n", "line2\n", "line3\n"],
    ["line1\n", "modified\n", "line3\n"],
    fromfile="old.py", tofile="new.py"
)
print(''.join(diff))
# --- old.py
# +++ new.py
# @@ -1,3 +1,3 @@
#  line1
# -line2
# +modified
#  line3

# 最佳匹配:
words = ["Python", "Java", "JavaScript", "Go"]
print(difflib.get_close_matches("Jva", words))
# ['Java', 'JavaScript']
```


## 文本格式化技巧


```
// ========== 多行字符串 ==========
# 括号隐式连接:
text = ("这是第一行 "
        "这是第二行 "
        "这是第三行")
print(text)   # 一行字符串

# 反斜杠续行:
text = "第一行 \
第二行 \
第三行"

# 三引号 (保持格式):
text = """第一行
    第二行 (有缩进)
第三行"""

// ========== 对齐与填充 ==========
print("左对齐".ljust(20, "-"))    # '左对齐-----------------'
print("右对齐".rjust(20, "-"))    # '-----------------右对齐'
print("居中".center(20, "-"))     # '--------居中--------'
print("42".zfill(5))              # '00042' (前导零填充)

// ========== 前缀/后缀检查 ==========
url = "https://example.com"
print(url.startswith("https"))    # True
print(url.endswith(".com"))       # True

// ========== 分割与连接 ==========
csv = "苹果,香蕉,橘子"
items = csv.split(",")            # ['苹果', '香蕉', '橘子']
print("|".join(items))            # '苹果|香蕉|橘子'

// ========== strip ==========
text = "  \n\t Hello World \n  "
print(text.strip())               # 'Hello World'
print(text.lstrip())              # 去掉左边空白
print(text.rstrip())              # 去掉右边空白
# 也可指定字符: text.strip("*")
```


> **Note:** 💡 字符串处理要点: (1) string.ascii_letters/digits/punctuation 字符集常量; (2) str.encode() / bytes.decode() 编码转换; (3) textwrap.fill/wrap/dedent/indent 文本排版; (4) difflib.SequenceMatcher 比较相似度; (5) Template 安全替换,适合用户模板。


## 练习


<!-- Converted from: 79_Python字符串与文本处理.html -->
