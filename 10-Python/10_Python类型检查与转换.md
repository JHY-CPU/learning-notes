# Python类型检查与转换


## 🏷️ Python 类型检查与转换


type() vs isinstance()、鸭子类型、显式类型转换、类型提示入门、强类型特性。


## type() vs isinstance()


```
// ========== type() 获取类型 ==========
type(42)                 #
type("hello")            #
type([1, 2])             #
type(None)               #

// type() 比较:
type(42) == int          # True
type("hello") == str     # True
type(42) == str          # False

// ========== isinstance() 推荐! ==========
isinstance(42, int)                   # True
isinstance("hello", str)              # True
isinstance(42, str)                   # False

// 支持继承检查 (type() 不支持):
class Animal: pass
class Dog(Animal): pass

d = Dog()
type(d) == Animal        # False! ❌ (type 不认继承)
isinstance(d, Animal)    # True ✅ (isinstance 认继承)

// 多种类型:
isinstance(42, (int, float, str))    # True (元组中的任一类型)
isinstance("hello", (int, float))    # False

// ========== 何时用哪个 ==========
// isinstance():  95% 的情况,支持继承,更灵活
// type() ==:     需要精确匹配类型时 (不接受子类)

// ========== issubclass() ==========
issubclass(Dog, Animal)              # True
issubclass(bool, int)                # True (bool 是 int 的子类!)
issubclass(Dog, (Animal, object))    # True
```


## 显式类型转换


```
// ========== 整数转换 int() ==========
int("42")                # 42
int(3.9)                 # 3 (向下截断,非四舍五入)
int("1010", 2)           # 10 (二进制字符串→整数)
int("ff", 16)            # 255 (十六进制)
int(True)                # 1

// 常见错误:
int("hello")             # ValueError!

// ========== 浮点转换 float() ==========
float("3.14")            # 3.14
float(42)                # 42.0
float("1.5e-10")         # 1.5e-10
float("inf")             # inf

// ========== 字符串转换 str() ==========
str(42)                  # "42"
str(3.14)                # "3.14"
str(True)                # "True"
str([1, 2, 3])           # "[1, 2, 3]"

// ========== 布尔转换 bool() ==========
bool(1)                  # True
bool(0)                  # False
bool("")                 # False
bool("hello")            # True
bool([])                 # False
bool([1, 2])             # True
bool(None)               # False

// ========== 列表/元组/集合转换 ==========
list("hello")            # ["h", "e", "l", "l", "o"]
tuple([1, 2, 3])         # (1, 2, 3)
set([1, 2, 2, 3])        # {1, 2, 3}
list((1, 2, 3))          # [1, 2, 3]

// ========== 安全转换模式 ==========
def safe_int(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

safe_int("42")           # 42
safe_int("abc")          # None
safe_int("abc", 0)       # 0
```


## 鸭子类型与类型提示


```
// ========== 鸭子类型 ==========
// "如果它走路像鸭子,叫声像鸭子,那它就是鸭子"
// Python 不关心对象的类型,只关心它有什么方法

def process(data):
    result = data.upper()  # 不检查类型,直接调用方法
    return result

process("hello")          # "HELLO" (str 有 upper())
process(42)               # AttributeError! (int 没有)

// 鸭子类型风格 (检查行为而非类型):
def duplicate(obj):
    if hasattr(obj, '__mul__'):  # 检查是否有 __mul__ 方法
        return obj * 2
    raise TypeError("不支持此类型")

duplicate(5)              # 10
duplicate("ab")           # "abab"
duplicate([1, 2])         # [1, 2, 1, 2]

// ========== 类型提示 (Python 3.5+) ==========
// 类型提示不影响运行,但让代码更清晰
// 配合编辑器获得自动补全和检查

name: str = "Alice"
age: int = 25

def greet(name: str) -> str:
    return f"Hello, {name}"

from typing import List, Dict, Optional, Union

def process(items: List[int]) -> Dict[str, int]:
    return {"sum": sum(items), "count": len(items)}

def find_user(id: int) -> Optional[dict]:
    # 可能返回 None
    if id in database:
        return database[id]
    return None

def handle(value: Union[int, str]) -> None:
    # 接受 int 或 str
    print(value)
```


## Python 强类型特性


```
// ========== Python 是强类型语言 ==========
// 不会隐式转换不兼容的类型
// 和 JavaScript/ PHP 不同

// Python:
"42" + 1                 # TypeError! ❌
// 必须显式转换:
int("42") + 1            # 43 ✅
"42" + str(1)            # "421" ✅

// JavaScript (对比):
// "42" + 1  → "421" (隐式转换)
// "42" - 1  → 41

// ========== 常见类型错误 ==========
None + 1                 # TypeError
"hello" * "world"        # TypeError
[1, 2] + {3, 4}          # TypeError (列表+集合)
{1: "a"} + {2: "b"}      # TypeError (字典不支持 +)

// ========== 类型安全实践 ==========
// 1. 函数入口做类型检查
def divide(a: float, b: float) -> float:
    if not isinstance(b, (int, float)):
        raise TypeError("b 必须是数字")
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

// 2. 使用类型提示 + mypy 静态检查
// 命令行: pip install mypy && mypy script.py

// 3. 边界情况检查
def get_first(items: list, default=None):
    if not items:        # 处理空列表
        return default
    return items[0]
```


> **Note:** 💡 类型最佳实践: (1) 用 isinstance() 而非 type() 检查类型; (2) Python 是强类型,不要试图和 JS 比; (3) 类型提示让代码自文档化,大型项目强烈推荐; (4) 鸭子类型更 Pythonic,但必要时做防御性检查; (5) 配合 mypy 静态检查,可以提前发现类型错误。


## 练习


<!-- Converted from: 10_Python类型检查与转换.html -->
