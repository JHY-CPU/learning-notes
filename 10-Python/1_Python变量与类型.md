# Python变量与类型


## 📦 Python 变量与类型


变量与动态类型、基本数据类型 (int/float/str/bool/None)、type()、id()、类型转换。


## 变量与动态类型


```
// ========== Python 变量 ==========
// Python 是动态类型语言
// 变量不需要事先声明类型
// 变量名 = 值

name = "Alice"           # 字符串
age = 25                 # 整数
height = 1.68            # 浮点数
is_student = True        # 布尔值

// 变量可以重新赋值为不同类型 (动态类型)
data = 42                # data 是 int
data = "hello"           # data 现在是 str (同一变量)

// ========== 变量命名规则 ==========
// 1. 字母/数字/下划线,不能数字开头
// 2. 区分大小写: name != Name
// 3. 不能使用保留字 (关键字)
// 4. 推荐 snake_case 命名

// 合法:
user_name = "Alice"
user1 = "Bob"
_private = "hidden"
myVar = "camelCase"      # 但不推荐

// 不合法:
// 1user = "no"     # SyntaxError
// user-name = "no"  # SyntaxError
// class = "no"     # class 是保留字

// Python 关键字:
import keyword
print(keyword.kwlist)
# ['False', 'None', 'True', 'and', 'as', 'assert', 'async',
#  'await', 'break', 'class', 'continue', 'def', 'del',
#  'elif', 'else', 'except', 'finally', 'for', 'from',
#  'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
#  'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
#  'with', 'yield']

// ========== 多变量赋值 ==========
a = b = c = 0            # 链式赋值
x, y, z = 1, 2, 3        # 多重赋值
x, y = y, x              # 交换变量 (无需临时变量!)
```


## 基本数据类型


```
// ========== 整数 (int) ==========
age = 25
count = -10
big = 1_000_000_000     # 千位分隔 (Python 3.6+)
hex_num = 0xFF           # 255 (十六进制)
bin_num = 0b1010         # 10 (二进制)
oct_num = 0o77           # 63 (八进制)

// Python 整数无限大 (不受 32/64 位限制)
huge = 10 ** 100         # 大整数

// ========== 浮点数 (float) ==========
pi = 3.14159
height = 1.75
scientific = 1.5e-10     # 科学计数法
inf = float('inf')       # 无穷大
nan = float('nan')       # 非数字

// 浮点精度问题:
0.1 + 0.2                # 0.30000000000000004 (不是 0.3!)
// 精确小数用 Decimal:
from decimal import Decimal
Decimal('0.1') + Decimal('0.2')  # Decimal('0.3')

// ========== 字符串 (str) ==========
name = "Alice"
greeting = 'Hello'
multi = """多行
字符串
内容"""
raw = r"C:\Users\name"   # 原始字符串 (不转义)

// f-string (Python 3.6+):
name = "Alice"
age = 25
print(f"姓名: {name}, 年龄: {age}")
print(f"PI ≈ {3.14159:.2f}")  # 格式化: PI ≈ 3.14

// ========== 布尔值 (bool) ==========
is_active = True
is_deleted = False

// 布尔值实际上是整数 (True=1, False=0)
True + True              # 2 (但不推荐这样做!)

// 假值 (Falsy):
// False, None, 0, 0.0, "" (空串), [] (空列表),
// {} (空字典), () (空元组), set()

// ========== None 空值 ==========
result = None            # 表示"没有值"
# 常见用途:
# - 函数返回值默认 None
# - 变量先定义后赋值
# - 表示缺失/可选值
```


## 类型检查与转换


```
// ========== type() — 查看类型 ==========
type(42)                 #
type("hello")            #
type(3.14)               #
type(True)               #
type(None)               #
type([1, 2])             #

// 检查类型:
isinstance(42, int)      # True
isinstance("hello", str) # True
isinstance(42, (int, float))  # True (多种类型用元组)

// ========== 类型转换 ==========
// 显式转换:

int("42")                # 42 (字符串→整数)
float("3.14")            # 3.14
str(42)                  # "42"
bool(1)                  # True
bool(0)                  # False
bool("")                 # False (空字符串)

// 注意事项:
int("hello")             # ValueError! (无法转换)
int(3.9)                 # 3 (截断,不是四舍五入)
round(3.9)               # 4 (四舍五入)

// ========== id() — 内存地址 ==========
a = 42
b = 42
id(a)                    # 内存地址 (整数)
id(a) == id(b)           # True (小整数被缓存)

// 可变 vs 不可变:
// 不可变: int, float, str, bool, tuple, frozenset
// 可变:   list, dict, set, 自定义对象

// 不可变: 修改会创建新对象
s = "hello"
id(s)                    # 地址 A
s += " world"            # 创建新字符串
id(s)                    # 地址 B (不同!)
```


> **Note:** 💡 Python 是强类型语言: "42" + 1 会报 TypeError,不会自动转换。这和 JavaScript 的 "42" + 1 = "421" 不同。用 isinstance() 而不是 type() 检查类型更安全 (支持继承)。不可变类型在多线程中安全,因为不会变。


## 练习


<!-- Converted from: 1_Python变量与类型.html -->
