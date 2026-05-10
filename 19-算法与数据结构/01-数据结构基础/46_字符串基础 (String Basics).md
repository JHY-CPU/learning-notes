# 字符串基础 (String Basics)

## 一、字符串的本质

字符串是由字符组成的**有序序列**。在大多数编程语言中，字符串是**不可变**的。

### 1.1 Python 字符串操作

```python
s = "Hello, World!"

# 基本操作
len(s)           # 长度 13
s[0]             # 'H' — 索引访问
s[7:12]          # 'World' — 切片
s[-1]            # '!' — 负数索引
'H' in s         # True — 包含判断
s.index('World') # 7 — 查找位置
s.count('l')     # 3 — 计数

# 变换
s.upper()        # 全大写
s.lower()        # 全小写
s.strip()        # 去除首尾空白
s.split(', ')    # ['Hello', 'World!']
''.join(['a','b']) # 'ab' — 拼接
s.replace('World', 'Python')

# 判断
s.startswith('Hello')
s.endswith('!')
s.isdigit()
s.isalpha()
```

### 1.2 C++ 字符串操作

```cpp
#include <string>
using namespace std;

string s = "Hello, World!";
int n = s.size();            // 长度
char c = s[0];               // 'H'
string sub = s.substr(7, 5); // "World"
size_t pos = s.find("World"); // 7
s += "!";                    // 拼接
s.erase(5);                  // 删除从位置5开始的内容
```

### 1.3 字符串与字符数组

```cpp
// C风格字符串
char str[] = "Hello";
int len = strlen(str);
char* p = strchr(str, 'e');  // 查找字符
int cmp = strcmp(str1, str2); // 比较
```

---

## 二、常见操作与技巧

### 2.1 字符串反转

```python
# Python
s = "hello"
rev = s[::-1]        # "olleh"
rev = ''.join(reversed(s))

# C++
void reverse_string(string& s) {
    int left = 0, right = s.size() - 1;
    while (left < right) {
        swap(s[left++], s[right--]);
    }
}
```

### 2.2 回文判断

```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1; right -= 1
    return True

# 忽略大小写和非字母数字
def is_palindrome_ignore(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
```

### 2.3 字符串转数字

```python
# Python
num = int("123")
num = int("-456")
num = float("3.14")

# 注意溢出（Python无此问题）
# C++ 需要处理溢出
int myAtoi(string s) {
    long result = 0;
    int sign = 1, i = 0;
    while (s[i] == ' ') i++;
    if (s[i] == '+' || s[i] == '-') sign = s[i++] == '+' ? 1 : -1;
    while (isdigit(s[i])) {
        result = result * 10 + (s[i++] - '0');
        if (result * sign > INT_MAX) return INT_MAX;
        if (result * sign < INT_MIN) return INT_MIN;
    }
    return (int)(result * sign);
}
```

### 2.4 字符计数

```python
from collections import Counter

s = "abracadabra"
freq = Counter(s)
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# 手动实现
freq = {}
for c in s:
    freq[c] = freq.get(c, 0) + 1
```

---

## 三、字符串匹配

### 3.1 暴力匹配 — O(nm)

```python
def brute_search(text, pattern):
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            return i
    return -1
```

### 3.2 Python 内置方法

```python
s = "Hello, World!"
s.find("World")     # 7，找不到返回 -1
s.index("World")    # 7，找不到抛异常
s.rfind("l")        # 10，从右找
"World" in s        # True
```

---

## 四、不可变性的代价

### 4.1 字符串拼接优化

```python
# 慢 — O(n^2)，每次拼接创建新字符串
result = ""
for s in list_of_strings:
    result += s

# 快 — O(n)
result = "".join(list_of_strings)
```

### 4.2 C++ 中的优化

```cpp
// 使用 reserve 预分配
string result;
result.reserve(total_length);
for (auto& s : strings) {
    result += s;
}
```

---

## 五、编码基础

### 5.1 ASCII 与 Unicode

- **ASCII：** 128个字符，值 0-127
- **Unicode：** 通用字符集
- **UTF-8：** 变长编码，1-4字节

```python
ord('A')   # 65
chr(65)    # 'A'
ord('中')  # 20013
```

### 5.2 面试中的字符集假设

通常假设：
- 小写字母（26个）
- 大小写字母（52个）
- ASCII可打印字符（95个）
- Unicode（需要哈希表）

---

## 六、面试要点

1. **不可变性** — 理解为什么字符串拼接慢
2. **边界处理** — 空串、单字符
3. **编码知识** — ord/chr、字符集大小
4. **内置方法** — 熟练使用但不能只依赖
