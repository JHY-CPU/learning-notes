# Python函数综合


## 📊 Python 函数综合


函数设计原则、函数式编程、综合案例、最佳实践、常见模式总结。


## 函数设计原则


```
// ========== 单一职责 ==========
// 一个函数只做一件事

// ❌ 不好的: 做了太多事
def process_user(user):
    # 验证
    if not user.get("name"):
        return False
    # 保存
    database.save(user)
    # 发送邮件
    send_email(user)
    # 记录日志
    log(f"User {user['name']} processed")
    return True

// ✅ 好的: 拆分成多个小函数
def validate_user(user):
    return bool(user.get("name"))

def save_user(user):
    database.save(user)

def notify_user(user):
    send_email(user)

def log_action(action):
    log(action)

// ========== 纯函数 ==========
// 纯函数: 相同输入→相同输出,无副作用

// ❌ 不纯 (依赖全局变量,有副作用):
count = 0
def increment():
    global count
    count += 1
    return count

// ✅ 纯函数:
def increment(count):
    return count + 1

// 纯函数好处:
// - 容易测试
// - 可预测
// - 线程安全
// - 可缓存

// ========== 命名规范 ==========
// 动词 + 名词:
get_user()              # ✅
calculate_total()       # ✅
validate_input()        # ✅
send_email()            # ✅

// 布尔函数:
is_active()             # ✅
has_permission()        # ✅
can_edit()              # ✅
contains_value()        # ✅
```


## 函数式编程模式


```
// ========== 高阶函数 ==========
// 函数作为参数或返回值

def apply_twice(func, value):
    return func(func(value))

result = apply_twice(lambda x: x * 2, 5)
print(result)                 # 20 (5*2=10, 10*2=20)

// ========== 函数组合 ==========
def compose(f, g):
    """返回 f(g(x))"""
    return lambda x: f(g(x))

def double(x): return x * 2
def increment(x): return x + 1

double_then_increment = compose(increment, double)  # double(x) → increment
increment_then_double = compose(double, increment)  # increment(x) → double

print(double_then_increment(5))     # 11 (5*2+1)
print(increment_then_double(5))     # 12 ((5+1)*2)

// ========== 柯里化 ==========
# 将多参数函数转为单参数函数链

def add(a, b):
    return a + b

# 手动柯里化
def curried_add(a):
    def inner(b):
        return a + b
    return inner

add5 = curried_add(5)
print(add5(3))                # 8
print(add5(7))                # 12

// ========== 偏函数 ==========
from functools import partial

def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))              # 25
print(cube(5))                # 125

// 实际应用:
def connect(host, port, timeout):
    print(f"连接 {host}:{port} (超时 {timeout}s)")

connect_local = partial(connect, host="localhost", timeout=30)
connect_local(5432)           # 连接 localhost:5432 (超时 30s)
```


## 综合案例


```
// ========== 案例: 数据处理管道 ==========
from functools import reduce

data = [
    {"name": "  Alice  ", "age": 25, "score": 95},
    {"name": "bob", "age": 30, "score": 52},
    {"name": "CHARLIE", "age": 22, "score": 88},
    {"name": "  david  ", "age": 17, "score": 73},
]

def clean_name(user):
    """清理名字: 去空白、首字母大写"""
    return {**user, "name": user["name"].strip().title()}

def filter_adult(user):
    """只保留成年人"""
    return user["age"] >= 18

def filter_passed(user):
    """只保留及格的"""
    return user["score"] >= 60

def format_output(user):
    """格式化输出"""
    return f"{user['name']}: {user['score']}分"

// 管道处理:
result = [
    format_output(user)
    for user in data
    if filter_adult(user) and filter_passed(user)
]
// 先清理:
cleaned = [clean_name(user) for user in data]
// 再过滤:
result = [format_output(u) for u in cleaned if filter_adult(u) and filter_passed(u)]

print(result)
// ["Alice: 95分", "Charlie: 88分"]

// ========== 案例: 验证器 ==========
def create_validator(rules):
    """创建验证器"""
    def validate(value):
        errors = []
        for name, check, message in rules:
            if not check(value):
                errors.append(message)
        return errors
    return validate

// 定义规则:
age_validator = create_validator([
    ("required", lambda x: x is not None, "年龄不能为空"),
    ("range", lambda x: 0 <= x <= 150, "年龄范围 0-150"),
    ("integer", lambda x: isinstance(x, int), "年龄必须是整数"),
])

print(age_validator(25))      # []
print(age_validator(-1))      # ["年龄范围 0-150"]
print(age_validator("abc"))   # ["年龄必须是整数"]
```


## 函数总结


```
// ========== 函数知识图谱 ==========
// 定义:       def / lambda
// 参数:       位置 / 默认 / *args / **kwargs / keyword-only
// 返回值:     return (单值/多值/None)
// 作用域:     LEGB / global / nonlocal
// 高阶:       map / filter / reduce
// 函数式:     partial / compose / curry
// 递归:       基线条件 / 递归条件 / 尾递归

// ========== 最佳实践清单 ==========
// ✓ 函数命名: 动词 + 名词,有意义
// ✓ 单一职责: 一个函数只做一件事
// ✓ 短小精悍: 尽量不超过 50 行
// ✓ 参数控制: 不超过 3-4 个参数
// ✓ 默认参数: 用 None 而非可变对象
// ✓ 类型提示: 公共函数加类型注解
// ✓ 文档:     公共函数写 docstring
// ✓ 纯函数:   尽量无副作用
// ✓ 提前返回: 避免深层嵌套

// ========== 参数数量建议 ==========
// 0-2 个: 理想
// 3-4 个: 可以接受
// 5+ 个:  考虑用 dataclass 或字典

def create_user(name, email, age):
    pass                     # ✅ 3 个参数,ok

def configure(host, port, db, user, passwd, timeout, retries):
    pass                     # ❌ 太多! 用 dataclass

from dataclasses import dataclass

@dataclass
class Config:
    host: str = "localhost"
    port: int = 5432
    db: str = "test"
    user: str = "admin"
    password: str = ""
    timeout: int = 30
    retries: int = 3

def configure(config: Config):
    pass                     # ✅ 用一个对象封装
```


> **Note:** 💡 函数总结: (1) 函数是 Python 组织代码的基本单元; (2) 单一职责 + 纯函数 = 易测试; (3) 参数多时用 dataclass; (4) 默认参数不要用可变对象; (5) 可读性 > 简洁性 > 性能。


## 练习


<!-- Converted from: 39_Python函数综合.html -->
