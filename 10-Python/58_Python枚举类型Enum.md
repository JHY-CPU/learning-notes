# Python枚举类型Enum


## 🏷️ Python 枚举类型 Enum


Enum 定义与成员、name/value 属性、遍历枚举、@unique 验证、IntEnum/StrEnum (3.11+)、Flag 位标志、auto 自动赋值、枚举应用场景。


## Enum 基础


```
// ========== 什么是枚举 ==========
// 枚举: 一组有名字的常量集合
// 避免"魔数" (magic number),提高代码可读性

from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

// 访问枚举成员:
print(Color.RED)               # Color.RED
print(Color.RED.name)          # "RED"
print(Color.RED.value)         # 1

// 比较:
print(Color.RED is Color.RED)  # True (唯一实例)
print(Color.RED == Color.RED)  # True
print(Color.RED is not Color.GREEN)  # True

// 遍历:
for color in Color:
    print(color.name, color.value)
# RED 1
# GREEN 2
# BLUE 3

// 成员类型检查:
print(isinstance(Color.RED, Color))  # True
print(type(Color.RED) == Color)      # True

// ========== 为什么不直接用普通常量 ==========
// ❌ 普通常量:
RED = 1
GREEN = 2
BLUE = 3

def paint(color):
    if color == RED: pass    # 可读但松散,任何值都可以

paint(RED)                   # 可以
paint(999)                   # 也可以! 不合理!

// ✅ 用枚举:
def paint(color: Color):
    if color is Color.RED: pass

paint(Color.RED)             # ✅
paint(999)                   # ❌ 类型提示会警告!
```


## Enum 高级用法


```
// ========== 自定义值 ==========
// 枚举值可以是任意类型

class HttpStatus(Enum):
    OK = 200
    NOT_FOUND = 404
    INTERNAL_ERROR = 500
    REDIRECT = ("redirect", 302)    # 元组作为值

    @property
    def description(self):
        descriptions = {
            200: "成功",
            404: "未找到",
            500: "服务器错误",
        }
        return descriptions.get(self.value, "未知")

print(HttpStatus.OK.description)     # 成功
print(HttpStatus.OK.value)           # 200

// ========== 成员比较 ==========
// 枚举成员通过 identity (is) 比较
// 值相等不意味成员相等!

class StatusA(Enum):
    ACTIVE = 1
    INACTIVE = 0

class StatusB(Enum):
    ACTIVE = 1
    INACTIVE = 0

print(StatusA.ACTIVE == StatusA.ACTIVE)  # True (同一类)
print(StatusA.ACTIVE == StatusB.ACTIVE)  # False (不同类)
print(StatusA.ACTIVE.value == StatusB.ACTIVE.value)  # True (值相同)

// ========== 成员访问 ==========
# 通过 name 访问:
print(Color["RED"])              # Color.RED

# 通过 value 访问:
print(Color(1))                  # Color.RED

# 所有成员列表:
print(list(Color))               # [Color.RED, Color.GREEN, Color.BLUE]

# 成员个数:
print(len(Color))                # 3

// ========== @unique 装饰器 ==========
// 确保没有重复值

from enum import Enum, unique

@unique
class Status(Enum):
    PENDING = 1
    ACTIVE = 2
    INACTIVE = 3
    # ARCHIVED = 3              # ValueError: duplicate value!

// 不加 @unique 可以有重复值 (别名)
class Status(Enum):
    ACTIVE = 1
    APPROVED = 1                # ACTIVE 的别名
    INACTIVE = 2

print(list(Status))              # [Status.ACTIVE, Status.INACTIVE] (别名不重复列出)
print(Status(1))                 # Status.ACTIVE (第一个定义的成员)
```


## IntEnum, StrEnum, Flag


```
// ========== IntEnum ==========
// 继承自 int 的枚举,可以直接用于数学运算

from enum import IntEnum

class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

print(Priority.HIGH > Priority.LOW)   # True (像整数一样比较)
print(Priority.HIGH + 1)              # 4 (整数运算)

tasks = [
    ("A", Priority.MEDIUM),
    ("B", Priority.HIGH),
    ("C", Priority.LOW),
]
sorted_tasks = sorted(tasks, key=lambda t: t[1])
print([t[0] for t in sorted_tasks])   # ["C", "A", "B"] (按优先级)

// IntEnum 可以代替普通整数常量
// 且兼容整数比较

// ========== StrEnum (Python 3.11+) ==========
// 字符串枚举

try:
    from enum import StrEnum

    class Language(StrEnum):
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        GO = "go"

    print(Language.PYTHON.upper())     # PYTHON (字符串方法可用)
    print(Language.PYTHON == "python") # True (和字符串比较)
except ImportError:
    print("StrEnum 需要 Python 3.11+")

// ========== Flag 位标志 ==========
// 支持位运算的枚举

from enum import Flag, auto

class Permission(Flag):
    NONE = 0
    READ = auto()             # 1
    WRITE = auto()            # 2
    EXECUTE = auto()          # 4
    ALL = READ | WRITE | EXECUTE  # 7

# 组合权限:
user_perm = Permission.READ | Permission.WRITE
print(user_perm)               # Permission.READ|WRITE

# 检查:
print(Permission.READ in user_perm)   # True
print(Permission.EXECUTE in user_perm) # False

# 添加权限:
user_perm |= Permission.EXECUTE
print(Permission.EXECUTE in user_perm) # True

# 移除权限:
user_perm &= ~Permission.WRITE
print(user_perm)               # Permission.READ|EXECUTE
```


## auto 与枚举实战


```
// ========== auto() 自动赋值 ==========
// auto 自动分配值 (默认从 1 开始)

from enum import Enum, auto

class Day(Enum):
    MONDAY = auto()           # 1
    TUESDAY = auto()          # 2
    WEDNESDAY = auto()        # 3
    THURSDAY = auto()         # 4
    FRIDAY = auto()           # 5
    SATURDAY = auto()         # 6
    SUNDAY = auto()           # 7

for day in Day:
    print(day.name, day.value)
# MONDAY 1, TUESDAY 2 ...

// 自定义 auto 行为:
from enum import Enum, auto

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()  # 将名字作为值

class Color(AutoName):
    RED = auto()             # value = "red"
    GREEN = auto()           # value = "green"

print(Color.RED.value)       # "red"

// ========== 实战: 状态机 ==========
class OrderStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

    def can_transition_to(self, new_status):
        """状态转换规则"""
        transitions = {
            OrderStatus.PENDING:    {OrderStatus.PAID, OrderStatus.CANCELLED},
            OrderStatus.PAID:       {OrderStatus.SHIPPED, OrderStatus.CANCELLED},
            OrderStatus.SHIPPED:    {OrderStatus.DELIVERED},
            OrderStatus.DELIVERED:  set(),
            OrderStatus.CANCELLED:  set(),
        }
        return new_status in transitions.get(self, set())

order = OrderStatus.PENDING
print(order.can_transition_to(OrderStatus.PAID))      # True
print(order.can_transition_to(OrderStatus.DELIVERED)) # False

// ========== 实战: 配置选项 ==========
class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5

    def should_log(self, level: "LogLevel") -> bool:
        return self.value <= level.value

current = LogLevel.INFO
print(current.should_log(LogLevel.DEBUG))    # False
print(current.should_log(LogLevel.ERROR))    # True
```


> **Note:** 💡 Enum 要点: (1) Enum 成员是单例,用 is 比较; (2) name 是成员名,value 是成员值; (3) @unique 防止重复值; (4) IntEnum 可以和整数比较; (5) Flag 用位运算组合权限,auto 自动赋值。


## 练习


<!-- Converted from: 58_Python枚举类型Enum.html -->
