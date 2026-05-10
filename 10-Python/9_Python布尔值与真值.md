# Python布尔值与真值


## ✅ Python 布尔值与真值


bool 类型、真值表、falsy 值、bool() 转换、短路求值实战、常见陷阱。


## 布尔类型基础


```
// ========== bool 类型 ==========
// Python 的布尔值: True 和 False
// 注意大写! (不是 true/false, 也不是 TRUE/FALSE)

is_active = True
is_deleted = False

// ========== bool 是 int 的子类 ==========
// True == 1, False == 0
True + True              # 2 (但不推荐!)
True * 5                 # 5
int(True)                # 1
int(False)               # 0

// 布尔运算:
True and True            # True
True and False           # False
True or False            # True
not True                 # False

// ========== bool() 转换 ==========
// 任何对象都可以转换为布尔值
bool(1)                  # True
bool(0)                  # False
bool("hello")            # True
bool("")                 # False
bool([1, 2])             # True
bool([])                 # False
bool(None)               # False

// 等价于直接用在 if 中:
if some_value:           # 等价于 if bool(some_value):
    pass
```


## Falsy 值大全


```
// ========== Python 中所有 Falsy 值 ==========
// 在布尔上下文中为 False 的值:

None                      # 空值
False                     # 布尔假
0                         # 整数零
0.0                       # 浮点零
0j                        # 复数零
""                        # 空字符串
()                        # 空元组
[]                        # 空列表
{}                        # 空字典
set()                     # 空集合
range(0)                  # 空 range

// 其余所有值都是 Truthy!

// ========== 常见判断模式 ==========
// 判断非空:
name = input("输入名字: ")
if name:                  # ✅ 优雅 (等价于 name != "")
    print(f"Hello, {name}")
else:
    print("名字不能为空")

// 判断列表非空:
items = []
if not items:             # ✅ if len(items) == 0 的简化
    print("列表为空")

// 判断不为 None:
result = get_data()
if result is not None:    # ✅ 用 is not None, 不用 if result:
    print(result)

// ⚠️ 注意: 0 和空列表也是 Falsy
// 如果需要区分 0 和 None,用 is not None
count = 0
if count:                 # False! 不对
    print("有数据")

if count is not None:     # True, 正确
    print("count = 0")
```


## 短路求值深入


```
// ========== 短路机制 ==========
// and: 遇到第一个 Falsy 就停止,返回该值
// or:  遇到第一个 Truthy 就停止,返回该值

// ========== and 短路 ==========
0 and print("不会执行")   # 0 (短路,不执行 print)
1 and 42                  # 42 (都真,返回最后一个)
1 and "yes"               # "yes"

// 多层:
1 and "ok" and 0 and "no" # 0 (第三个是 Falsy,短路)
1 and "a" and "b"         # "b" (全部 Truthy)

// ========== or 短路 ==========
1 or print("不会执行")    # 1 (短路,不执行 print)
0 or 42                   # 42 (第一个 Falsy,取第二个)
0 or "" or "default"      # "default" (跳过 Falsy)

// ========== 实用模式 ==========
// 默认值:
name = user_input or "匿名用户"   # 空时用默认值

// 链式默认:
config = os.getenv("MODE") or "development" or ""

// 条件执行:
is_admin and delete_user()        # 只有 admin 才执行

// 可选链替代:
user = get_user()
name = user and user.get("name") or "匿名"

// ========== 优先级与括号 ==========
// not > and > or
True or False and False     # True (等价于 True or (False and False))
(True or False) and False   # False

// 建议: 复杂表达式加括号提高可读性
result = (value > 0) and (value < 100)
```


## 常见陷阱与最佳实践


```
// ========== 陷阱 1: 误用 == 比较布尔 ==========
is_active = True
if is_active == True:      # ❌ 冗余
    pass
if is_active:              # ✅ 直接判断
    pass

// ========== 陷阱 2: 0 和 None 混淆 ==========
count = 0
if not count:              # True, 但本意可能是检查 None
    print("no count")      # 会执行!

if count is None:          # ✅ 明确检查 None
    print("no count")      # 不会执行

// ========== 陷阱 3: 空列表和 None 混淆 ==========
items = []
if not items:              # True (空列表)
    print("empty")

if items is not None:      # True (不是 None)
    print("exists")

// ========== 陷阱 4: all()/any() 的误用 ==========
numbers = [0, 1, 2, 3]
all(numbers)               # False (0 是 Falsy)
any(numbers)               # True (有 Truthy)

// 检查所有值 > 0:
all(x > 0 for x in numbers)  # False

// ========== 最佳实践总结 ==========
// 检查 None:     x is None / x is not None
// 检查空容器:   if not items:
// 检查非空:     if items:
// 默认值:       name = input or "default"
// 条件执行:     is_admin and do_action()
// 布尔方法名:   is_active, has_permission, can_edit
```


> **Note:** 💡 布尔值要点: (1) 记住所有 Falsy 值: None/False/0/空容器; (2) 检查 None 用 is / is not,不要用布尔判断; (3) 短路求值可用于简洁的默认值和条件执行; (4) not > and > or,复杂表达式加括号; (5) bool 方法是 is_xxx / has_xxx / can_xxx 命名。


## 练习


<!-- Converted from: 9_Python布尔值与真值.html -->
