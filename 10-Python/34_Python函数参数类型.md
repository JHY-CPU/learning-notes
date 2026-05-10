# Python函数参数类型


## 🔧 Python 函数参数类型


位置参数、关键字参数、默认参数、*args 可变位置参数、**kwargs 可变关键字参数、参数传递机制。


## 参数类型总览


```
// ========== 五种参数类型 ==========
// Python 函数参数按顺序:
//
// 1. 位置参数 (positional)
// 2. 默认参数 (default)
// 3. *args (可变位置参数)
// 4. 关键字参数 (keyword-only)
// 5. **kwargs (可变关键字参数)

// 完整定义顺序:
def func(pos, default="v", *args, kw_only, **kwargs):
    pass

// ========== 参数传递规则 ==========
// 调用时:
// 1. 位置参数按位置匹配
// 2. 关键字参数按名称匹配
// 3. 位置参数必须在关键字参数之前

def example(a, b, c):
    print(a, b, c)

example(1, 2, 3)        # 位置参数
example(1, c=3, b=2)    # 混合: 位置 + 关键字
// example(a=1, 2, 3)   # ❌ SyntaxError! 位置在关键字之后
```


## *args — 可变位置参数


```
// ========== *args ==========
// 收集多余的位置参数为一个元组
// args 是元组 (tuple)

def sum_all(*args):
    print(f"参数: {args}")
    return sum(args)

print(sum_all(1, 2, 3))       # 参数: (1, 2, 3) → 6
print(sum_all(1, 2, 3, 4, 5)) # 参数: (1, 2, 3, 4, 5) → 15

// ========== args + 普通参数 ==========
def greet(greeting, *names):
    for name in names:
        print(f"{greeting}, {name}!")

greet("Hello", "Alice", "Bob", "Charlie")
// Hello, Alice!
// Hello, Bob!
// Hello, Charlie!

// ========== args 解包 ==========
// 调用时也用 * 解包列表/元组
numbers = [1, 2, 3, 4, 5]
print(sum_all(*numbers))       # 15 (解包列表)

// ========== 实用场景 ==========
// 1. 日志函数:
def log(level, *messages):
    for msg in messages:
        print(f"[{level}] {msg}")

log("INFO", "启动", "连接成功", "处理完成")

// 2. 打印多个值带分隔:
def print_sep(sep=", ", *items):
    print(sep.join(str(i) for i in items))

print_sep("-", 1, 2, 3)  # "1-2-3"

// 3. 装饰器参数传递 (后面会学到):
```


## **kwargs — 可变关键字参数


```
// ========== **kwargs ==========
// 收集多余的关键字参数为一个字典
// kwargs 是字典 (dict)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Beijing")
// name: Alice
// age: 25
// city: Beijing

// ========== args + kwargs 组合 ==========
def func(a, b, *args, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, x=10, y=20)
// a=1, b=2
// args=(3, 4)
// kwargs={"x": 10, "y": 20}

// ========== kwargs 解包 ==========
def create_user(name, age, city):
    print(f"用户: {name}, {age}, {city}")

user_data = {"name": "Alice", "age": 25, "city": "Beijing"}
create_user(**user_data)          # 解包字典为关键字参数

// ========== 实用场景 ==========
// 1. 配置传递:
def connect(**options):
    host = options.get("host", "localhost")
    port = options.get("port", 5432)
    print(f"连接到 {host}:{port}")

connect(host="db.example.com", port=5432, timeout=30)

// 2. 继承/包装:
class BaseAPI:
    def request(self, endpoint, **kwargs):
        # kwargs 包含所有请求参数
        url = f"https://api.example.com/{endpoint}"
        return requests.get(url, **kwargs)

// 3. 传递到其他函数:
def wrapper(*args, **kwargs):
    return target_function(*args, **kwargs)
```


## 关键字专用参数


```
// ========== 关键字专用参数 (Keyword-Only) ==========
// * 之后的参数只能用关键字传递

// * 表示后面的参数必须用关键字:
def compare(a, b, *, ignore_case=False):
    if ignore_case:
        return a.lower() == b.lower()
    return a == b

compare("hello", "HELLO")                  # False
compare("hello", "HELLO", ignore_case=True)  # True ✅ 关键字
// compare("hello", "HELLO", True)           # ❌ TypeError!

// ========== 位置专用参数 (Python 3.8+) ==========
// / 之前的参数只能用位置传递

def greet(name, age, /, greeting="Hello"):
    print(f"{greeting}, {name} ({age})")

greet("Alice", 25)                          # ✅ 位置
greet("Alice", 25, greeting="Hi")           # ✅
// greet(name="Alice", age=25)              # ❌ 不能用关键字!

// ========== 完整参数顺序 ==========
def func(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
    """完整参数模式:
    pos1, pos2:          位置专用
    pos_or_kwd:          位置或关键字
    kwd1, kwd2:          关键字专用
    """
    print(pos1, pos2, pos_or_kwd, kwd1, kwd2)

func(1, 2, 3, kwd1=4, kwd2=5)

// ========== 实用模式 ==========
// 强制关键字参数使调用更清晰
def send_email(to, subject, *, cc=None, bcc=None, attachments=None):
    """发送邮件,cc/bcc/attachments 必须用关键字"""
    pass

send_email("a@x.com", "Hello")
send_email("a@x.com", "Hello", cc="b@x.com")
send_email("a@x.com", "Hello", bcc="c@x.com")

// 调用清晰,不会混淆参数顺序
```


> **Note:** 💡 参数要点: (1) 参数顺序: 位置→默认→*args→关键字→**kwargs; (2) * 强制关键字参数,** 收集额外关键字参数; (3) / 强制位置参数 (Python 3.8+); (4) *args 是元组,**kwargs 是字典; (5) 参数太多时考虑用 dataclass。


## 练习


<!-- Converted from: 34_Python函数参数类型.html -->
