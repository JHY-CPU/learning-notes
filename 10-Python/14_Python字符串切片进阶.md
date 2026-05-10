# Python字符串切片进阶


## ✂️ Python 字符串切片进阶


切片完整语法、步长技巧、负索引、切片边界、实用切片模式、切片赋值。


## 切片基础回顾


```
// ========== 切片语法 ==========
// str[开始:结束:步长]
// 左闭右开: [开始, 结束)

text = "hello world"

text[0]                  # "h"  (索引)
text[-1]                 # "d"  (倒数第一个)
text[0:5]                # "hello" (索引 0~4)
text[6:]                 # "world" (6 到末尾)
text[:5]                 # "hello" (开头到 4)
text[:]                  # "hello world" (完整副本)

// ========== 省略规则 ==========
// 省略开始 = 从开头开始
// 省略结束 = 直到末尾
// 省略步长 = 步长 1

text[:]                  # 完整字符串 (副本)
text[::]                 # 同上
text[0:len(text)]        # 同上 (显式写法)

// ========== 负索引切片 ==========
text[-5:]                # "world" (后 5 个字符)
text[:-5]                # "hello " (去掉后 5 个)
text[-5:-1]              # "worl" (倒数第 5 到倒数第 2)
text[-5:0]               # "" (空的! 步长为正时不会往回走)

// ========== 越界安全 ==========
// 切片越界不会报错!
text[0:100]              # "hello world" (超过末尾,截断到末尾)
text[100:200]            # "" (空字符串,不报错)
text[5:0]                # "" (步长为正,开始 > 结束,返回空)
```


## 步长与反转


```
// ========== 步长 (step) ==========
text = "hello world"

text[::2]                # "hlowrd" (每 2 个取 1 个)
text[1::2]               # "el ol" (从索引 1 开始,每 2 个取 1 个)
text[::3]                # "hlwl" (每 3 个取 1 个)
text[1:8:2]              # "el o" (索引 1~7, 步长 2)

// ========== 反转字符串 ==========
text[::-1]               # "dlrow olleh" (最简反转)

// 反转但保留部分:
text[-1:-10:-1]          # "dlrow ol" (从右往左取)
text[5::-1]              # " olleh" (从索引 5 向左)
text[:5:-1]              # "dlrow" (从末尾到索引 5,向左)

// ========== 步长为负数 ==========
// 负步长: 从右向左取
// 此时开始 > 结束才有结果

text = "abcdef"
text[5:0:-1]             # "fedcb" (从索引 5 到 1)
text[5::-1]              # "fedcba" (全部反转)
text[:0:-1]              # "fedcb" (反转去掉第一个)
text[::-1]               # "fedcba" (完全反转)

// 记住: 负步长时,开始默认为末尾,结束默认为开头
```


## 实用切片模式


```
// ========== 提取子串 ==========
url = "https://example.com/path/to/page"
url[8:]                  # "example.com/path/to/page" (去掉协议)

email = "alice@example.com"
username = email[:email.index("@")]      # "alice"
domain = email[email.index("@") + 1:]    # "example.com"

// ========== 文件扩展名 ==========
filename = "data.csv"
filename[-4:]            # ".csv"
filename[:-4]            # "data"
filename[-3:]            # "csv" (无点)

// 更通用的方式:
def get_extension(filename):
    return filename[filename.rfind(".") + 1:] if "." in filename else ""

get_extension("data.csv")      # "csv"
get_extension("archive.tar.gz") # "gz"

// ========== 每隔 N 个取 ==========
"abcdefghij"[::3]        # "adgj" (每 3 个取第一个)
"abcdefghij"[1::3]       # "beh" (从索引 1 开始每 3 个)
"abcdefghij"[2::3]       # "cfi" (从索引 2 开始每 3 个)

// ========== 去掉首尾字符 ==========
text = "[[hello]]"
text[2:-2]               # "hello" (去掉头尾各 2 个)

"[hello]".strip("[]")    # "hello" (也可以用 strip)
"[[hello]]".strip("[]")  # "hello" (strip 去掉所有首尾匹配)
"[[hello]]"[2:-2]        # "hello" (精确去掉头尾各 2 个)

// ========== 回文检查 ==========
def is_palindrome(s):
    clean = s.lower().replace(" ", "")
    return clean == clean[::-1]

is_palindrome("racecar")         # True
is_palindrome("A man a plan canal Panama".lower().replace(" ", ""))  # True
```


## 切片对象与赋值


```
// ========== slice() 对象 ==========
// 可以用 slice() 创建可复用的切片

slice_5_10 = slice(5, 10)
text = "hello world python"
text[slice_5_10]           # " worl"

slice_even = slice(0, None, 2)  # 从开头到结束,步长 2
text[slice_even]           # "hlowrdpthn"

// ========== 字符串不支持切片赋值! ==========
// 字符串是不可变类型
s = "hello"
// s[0] = "H"             # TypeError!

// 必须创建新字符串:
s = "H" + s[1:]           # "Hello"

// ========== 列表支持切片赋值 ==========
// 列表是可变类型,支持切片赋值
items = [0, 1, 2, 3, 4, 5]
items[1:4] = [10, 20]     # [0, 10, 20, 4, 5] (替换)
items[1:3] = []           # [0, 4, 5] (删除)
items[0:0] = [99]         # [99, 0, 4, 5] (插入)

// ========== 切片性能 ==========
// 切片创建新字符串 (O(n) 时间, O(n) 空间)
// 频繁切片大字符串用索引 + 循环更省内存

// 查看第一个字符:
s[0]                      # O(1) ✅
s[:1]                     # O(n) ❌ (创建新字符串)

// 处理大字符串:
big = "x" * 1_000_000
first = big[0]            # ✅ 快
first_slice = big[:1]     # ❌ 慢 (复制 100 万字符)
```


> **Note:** 💡 切片要点: (1) [::-1] 是最简洁的反转; (2) 切片越界返回空或截断,不报错; (3) 负步长从右向左; (4) 字符串不可变,不能切片赋值; (5) 取单个字符用索引而非切片 (性能更好)。


## 练习


<!-- Converted from: 14_Python字符串切片进阶.html -->
