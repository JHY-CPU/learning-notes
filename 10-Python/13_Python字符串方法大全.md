# Python字符串方法大全


## 📖 Python 字符串方法大全


完整字符串方法参考，按功能分类：大小写、查找、判断、修整、填充、分割、连接、格式化。


## 大小写转换


```
// ========== 大小写方法 ==========
text = "hello World"

text.upper()              # "HELLO WORLD"   (全大写)
text.lower()              # "hello world"   (全小写)
text.title()              # "Hello World"   (标题格式,每个单词首字母大写)
text.capitalize()         # "Hello world"   (首字母大写,其余小写)
text.swapcase()           # "HELLO wORLD"   (大小写互换)
text.casefold()           # "hello world"   (更激进的 lower, 用于忽略大小写匹配)

// casefold vs lower:
"Straße".lower()          # "straße"
"Straße".casefold()       # "strasse" (特殊字符也能处理)
"straße".casefold()       # "strasse"

// 常用于: 大小写不敏感比较
if user_input.lower() == "yes":   # 匹配 Yes/YeS/yes
    pass

if user_input.casefold() == text.casefold():  # 国际化比较
    pass
```


## 查找与统计


```
// ========== 查找 ==========
text = "hello world hello"

text.find("world")        # 6 (找到返回索引)
text.find("xyz")          # -1 (找不到返回 -1)
text.index("world")       # 6 (找到返回索引)
text.index("xyz")         # ValueError! (找不到抛出异常)

text.rfind("hello")       # 12 (从右查找)
text.rindex("hello")      # 12 (从右查找,找不到抛异常)

// 指定范围查找:
text.find("o", 5)         # 从索引 5 开始找 "o"
text.find("o", 5, 10)     # 在索引 5-10 范围内找

// ========== 统计与判断 ==========
text.count("hello")       # 2
text.count("l")           # 5 (字符也可以)
text.count("l", 3, 8)     # 范围统计

text.startswith("hello")  # True
text.startswith("he")     # True
text.startswith("world", 6)  # True (从索引 6 开始)
text.endswith("hello")    # True
text.endswith(("world", "hello"))  # True (多个选项)

// ========== 包含检查 ==========
"world" in text           # True
"xyz" in text             # False
"xyz" not in text         # True
```


## 判断方法


```
// ========== 字符类型判断 ==========
"hello".isalpha()         # True (全是字母)
"hello123".isalpha()      # False (含数字)
"123".isdigit()           # True (全是数字)
"１２３".isdigit()        # True (全角数字)
"123".isdecimal()         # True (十进制数字)
"123".isnumeric()         # True (任何数字)
"½".isnumeric()           # True (Unicode 分数)
"½".isdecimal()           # False
"abc123".isalnum()        # True (字母或数字)
"   ".isspace()           # True (全是空白字符)
"".isspace()              # False!

// ========== 格式判断 ==========
"Hello World".istitle()   # True (标题格式)
"hello world".islower()   # True (全小写)
"HELLO".isupper()         # True (全大写)
"Hello".isprintable()     # True (可打印)
"Hello\n".isprintable()   # False (含换行符)

// ========== 标识符判断 ==========
"valid_name".isidentifier()  # True (合法标识符)
"2name".isidentifier()       # False (数字开头)
"class".isidentifier()       # True (但 class 是关键字)

// 检查是否 Python 关键字:
import keyword
keyword.iskeyword("class")   # True
keyword.iskeyword("hello")   # False
```


## 修整与填充


```
// ========== 修整 (strip) ==========
text = "  hello world  "
text.strip()              # "hello world"   (去首尾空白)
text.lstrip()             # "hello world  " (去左侧)
text.rstrip()             # "  hello world" (去右侧)

// 指定字符:
"---hello---".strip("-")      # "hello"
"www.example.com".strip("w")  # ".example.com" (注意: 去首尾的 w)
"   hello   ".strip()         # "hello" (默认空白: 空格/制表符/换行)

// 常用:
def clean_input(text):
    return text.strip().lower()

// ========== 填充 ==========
"42".zfill(5)             # "00042" (前补零到 5 位)
"-42".zfill(5)            # "-0042" (负数也有效)

"hello".center(11)        # "   hello   "
"hello".center(11, "*")   # "***hello***"

"hello".ljust(10)         # "hello     "
"hello".rjust(10)         # "     hello"
"hello".ljust(10, ".")    # "hello....."

// ========== 扩展与压缩 ==========
"hello\tworld".expandtabs(4)  # "hello   world" (Tab 转空格)

text = "  hello   world  "
" ".join(text.split())         # "hello world" (压缩空白)
```


## 分割、连接与替换


```
// ========== 分割 ==========
text = "apple,banana,orange"
text.split(",")           # ["apple", "banana", "orange"]

"hello world".split()     # ["hello", "world"] (默认按空白分割)
"hello   world".split()   # ["hello", "world"] (多个空白当做一个)

"a,b,c,d".split(",", 2)  # ["a", "b", "c,d"] (最多 2 次分割)
"a,b,c,d".rsplit(",", 2) # ["a,b", "c", "d"] (从右分割)

// 按行分割:
"line1\nline2\nline3".splitlines()      # ["line1", "line2", "line3"]
"line1\nline2\nline3".splitlines(True)  # ["line1\n", "line2\n", "line3"]

// 分区:
"hello:world".partition(":")   # ("hello", ":", "world")
"hello:world:foo".partition(":")   # ("hello", ":", "world:foo")
"hello:world:foo".rpartition(":")  # ("hello:world", ":", "foo")
"no-separator".partition(":")      # ("no-separator", "", "")

// ========== 连接 ==========
items = ["apple", "banana", "orange"]
", ".join(items)          # "apple, banana, orange"
"".join(items)            # "applebananaorange"
"\n".join(items)          # "apple\nbanana\norange"

// ⚠️ join 是字符串方法,不是列表方法!
// 正确: ",".join(list)
// 错误: list.join(",")

// ========== 替换 ==========
text = "hello world hello"
text.replace("hello", "hi")        # "hi world hi"
text.replace("hello", "hi", 1)     # "hi world hello" (只替换 1 次)
text.replace("l", "L")             # "heLLo worLd heLLo"

// 移除前缀/后缀 (Python 3.9+):
"filename.txt".removeprefix("file")  # "name.txt"
"filename.txt".removesuffix(".txt")  # "filename"
"filename.txt".removeprefix("FILE")  # "filename.txt" (区分大小写)

// translate + maketrans:
trans = str.maketrans("aeiou", "12345")
"hello world".translate(trans)  # "h2ll4 w4rld"
```


> **Note:** 💡 字符串方法要点: (1) 所有方法返回新字符串,不修改原串; (2) find 找不到返回 -1, index 找不到抛异常; (3) split 默认按空白分割,可指定 maxsplit; (4) join 是字符串方法; (5) 判断方法 isalpha/isdigit/isalnum 常用于输入验证。


## 练习


<!-- Converted from: 13_Python字符串方法大全.html -->
