# Python转义字符与原始字符串


## 🔧 Python 转义字符与原始字符串


转义序列、原始字符串 r''、字节串 bytes、Unicode 与编码、常见陷阱。


## 转义字符大全


```
// ========== 常见转义序列 ==========
// 反斜杠 \ + 字符 = 特殊含义

text = "Hello\nWorld"
// Hello
// World  (换行)

text = "Hello\tWorld"
// Hello   World  (制表符)

// ========== 完整转义表 ==========
\\      # 反斜杠本身           print("C:\\Users\\name")
\'      # 单引号               print('it\'s fine')
\"      # 双引号               print("say \"hello\"")
\n      # 换行 (LF)            print("line1\nline2")
\r      # 回车 (CR)            print("line1\rline2")
\t      # 水平制表符           print("col1\tcol2")
\b      # 退格                 print("hello\b world")  # "hell world"
\f      # 换页                 print("page1\fpage2")
\v      # 垂直制表符           print("a\vb")

// ========== 八进制与十六进制 ==========
\123    # 八进制字符           "\123" → "S"
\x41    # 十六进制字符          "\x41" → "A"
\x42    #                       "\x42" → "B"

// ========== Unicode 转义 ==========
A  # 16-bit Unicode       "A" → "A"
\U0001F600  # 32-bit Unicode   "\U0001F600" → "😀"
\N{name}    # Unicode 名称     "\N{LATIN SMALL LETTER A}" → "a"

print("中文")           # "中文"
print("\U0001F436")              # "🐶" (小狗 emoji)
print("\N{GRINNING FACE}")       # "😀"
```


## 原始字符串 r''


```
// ========== 原始字符串 ==========
// r"..." 或 r'...' 让反斜杠不再转义
// 常用于: 路径、正则表达式、Windows 路径

// 普通字符串:
path = "C:\\Users\\name"   # "C:\Users\name" (需要双反斜杠)

// 原始字符串 (推荐):
path = r"C:\Users\name"    # "C:\\Users\\name" (一个反斜杠保留)

// ========== 正则表达式 ==========
import re

// 普通字符串匹配数字:
pattern = "\\d+\\.\\d+"    # 🤯 双反斜杠

// 原始字符串匹配数字:
pattern = r"\d+\.\d+"      # 😊 清晰!

// 实际使用:
re.findall(r"\d+", "abc123def456")  # ["123", "456"]
re.search(r"hello\sworld", "hello world")  # 匹配

// ========== 原始字符串的边界 ==========
// 原始字符串不能以反斜杠结尾!
// r"abc\"   # ❌ 语法错误 (反斜杠会转义引号)

// 正确方式:
r"abc" "\\"               # "abc\\"
"abc\\"                   # "abc\\"

// ========== 多行原始字符串 ==========
sql = r"""
SELECT *
FROM users
WHERE name LIKE '%\%%'  # 反斜杠不被转义
"""
```


## 字节串 bytes 与 bytearray


```
// ========== bytes (不可变) ==========
// b"..." 表示字节串
// 每个元素是 0-255 的整数

b = b"hello"
type(b)                  #
b[0]                     # 104 (ASCII 码 'h')

// 创建 bytes:
bytes([104, 101, 108, 108, 111])  # b"hello"
bytes(5)                           # b"\x00\x00\x00\x00\x00"
"hello".encode("utf-8")            # b"hello"

// ========== bytearray (可变) ==========
ba = bytearray(b"hello")
ba[0] = 72               # 修改第一个字节 (72 = 'H')
ba                       # bytearray(b"Hello")
ba.append(33)            # bytearray(b"Hello!")

// ========== 编码与解码 ==========
text = "中文"
encoded = text.encode("utf-8")    # b'\xe4\xb8\xad\xe6\x96\x87'
decoded = encoded.decode("utf-8") # "中文"

// 常见编码:
text.encode("utf-8")     # UTF-8 (默认,推荐)
text.encode("gbk")       # GBK (中文系统)
text.encode("ascii")     # ASCII (不含中文会报错)

// ========== 编码错误处理 ==========
text = "中文"
text.encode("ascii")                          # UnicodeEncodeError!
text.encode("ascii", errors="ignore")         # b'' (忽略无法编码的)
text.encode("ascii", errors="replace")        # b'??' (替换为 ?)
text.encode("ascii", errors="backslashreplace") # b'\\u4e2d\\u6587'

b = b'\xe4\xb8\xad'
b.decode("utf-8")        # "中"
b.decode("ascii", errors="replace")  # "���"
```


## Unicode 与字符串比较


```
// ========== Python 3 字符串 ==========
// Python 3 的 str 默认是 Unicode!
// 每个字符都是 Unicode 码点

s = "A"
len(s)                   # 1
s = "中"
len(s)                   # 1 (一个字符,不是多个字节!)
s = "😀"
len(s)                   # 1 (emoji 也是一个字符!)

// ========== 字符串比较与排序 ==========
// 字符串比较基于 Unicode 码点
ord("A")                 # 65
ord("中")                # 20013
ord("😀")                # 128512
chr(65)                  # "A"
chr(20013)               # "中"

// 排序问题:
sorted(["apple", "Banana", "cherry"])
// ["Banana", "apple", "cherry"] (大写 B 的码点 66 < 小写 a 97)

// 忽略大小写排序:
sorted(["apple", "Banana", "cherry"], key=str.lower)
// ["apple", "Banana", "cherry"]

// ========== 常见陷阱 ==========
// 陷阱 1: 多重编码
s1 = "café"
s2 = "café"        # e + 重音组合
s1 == s2                 # False! (不同编码)

import unicodedata
unicodedata.normalize("NFC", s1) == unicodedata.normalize("NFC", s2)
# True (标准化后比较)

// 陷阱 2: len 对 emoji
len("😀")                # 1 (Python 3)
// 但组合 emoji:
len("👨‍👩‍👧‍👦")          # 7 (多个码点组合!)
```


> **Note:** 💡 转义与编码要点: (1) Windows 路径和正则表达式用原始字符串 r"..."; (2) Python 3 的 str 是 Unicode,不需要纠结编码; (3) 文件 I/O 时才需要 encode/decode; (4) 原始字符串不能以反斜杠结尾; (5) 字符串比较用 unicodedata.normalize 处理重音字符。


## 练习


<!-- Converted from: 15_Python转义字符与原始字符串.html -->
