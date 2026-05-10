# Python身份与成员运算符


## 🔍 Python 身份与成员运算符


is vs == 深入对比、整数缓存、None 比较最佳实践、in 运算符、自定义 __contains__。


## 身份运算符 is


```
// ========== is 运算符 ==========
// is: 检查两个变量是否指向同一个对象 (内存地址相同)
// ==: 检查两个对象的值是否相等

a = [1, 2, 3]
b = [1, 2, 3]
c = a

a == b                   # True (值相同)
a is b                   # False (不同对象)
a is c                   # True (同一对象)

// ========== id() 查看内存地址 ==========
id(a)                    # 140234567890... (内存地址)
id(b)                    # 140234567891... (不同)
id(c)                    # 140234567890... (和 a 相同)

// ========== 整数缓存 (-5 到 256) ==========
// Python 缓存了小整数对象,复用相同对象
a = 100
b = 100
a is b                   # True (缓存,同一对象)

a = 1000
b = 1000
a is b                   # False (大整数,不同对象!)
a == b                   # True (值相等)

// ⚠️ 不要依赖整数 is 比较!
// 小整数可能 True,大整数 False
// 比较值始终用 ==

// ========== 字符串驻留 ==========
a = "hello"
b = "hello"
a is b                   # True (短字符串驻留)

a = "hello world!" * 100
b = "hello world!" * 100
a is b                   # False (长字符串不驻留)
a == b                   # True

// ⚠️ 不要依赖字符串 is 比较!
// 始终用 == 比较字符串值
```


## None 比较


```
// ========== None 比较: 用 is! ==========
// None 是单例 (全局只有一个 None 对象)

result = None
result is None           # ✅ 推荐
result == None           # ❌ 不推荐 (虽然可以,但有歧义)
result is not None       # ✅ 推荐

// ========== 为什么用 is 而不是 == ==========
// 1. 性能: is 比 == 快 (比较地址 vs 调用 __eq__)
// 2. 语义: 检查"是否无值"而非"值相等"
// 3. 安全: 避免自定义 __eq__ 的误判

class BadClass:
    def __eq__(self, other):
        return True      # 糟糕的 __eq__ 实现

obj = BadClass()
obj == None              # True! ❌ 错误匹配
obj is None              # False ✅ 正确

// ========== None 常见模式 ==========
def get_user(id):
    # 返回 User 或 None
    if id in database:
        return database[id]
    return None

user = get_user(42)

// 检查:
if user is not None:     # ✅
    print(user.name)

if user:                 # ❌ 如果 User 类 __bool__ 返回 False 会有问题
    print(user.name)

// ========== 默认值 ==========
def fetch_data(cache=None):
    if cache is None:     # ✅ 不能用 if not cache (空列表也是 Falsy)
        cache = []
    return cache
```


## 成员运算符 in


```
// ========== in 基本用法 ==========
// in: 检查是否包含
// not in: 检查是否不包含

// 字符串:
"he" in "hello"          # True
"xyz" in "hello"         # False
"x" not in "hello"       # True

// 列表:
3 in [1, 2, 3]           # True
4 in [1, 2, 3]           # False

// 元组:
3 in (1, 2, 3)           # True

// 字典: 检查的是键!
user = {"name": "Alice", "age": 25}
"name" in user           # True (检查键,不是值)
"Alice" in user          # False (值是 "Alice")
"Alice" in user.values() # True (检查值)

// 集合:
3 in {1, 2, 3}           # True (集合查找 O(1),非常快!)

// ========== 性能对比 ==========
// 列表查找: O(n) 线性扫描
// 集合/字典查找: O(1) 哈希查找

items_list = list(range(1000000))
items_set = set(range(1000000))

999999 in items_list     # O(n) 慢
999999 in items_set      # O(1) 快

// 大量查找时,先转成集合:
search_items = [100, 200, 300]
items_pool = set(large_list)  # 转集合
for item in search_items:
    if item in items_pool:    # O(1) 查找
        print(f"找到 {item}")
```


## 自定义 __contains__


```
// ========== 自定义 in 行为 ==========
// 实现 __contains__ 方法让自定义类支持 in

class Playlist:
    def __init__(self):
        self.songs = []

    def add(self, song):
        self.songs.append(song)

    def __contains__(self, item):
        // 自定义 in 检查逻辑
        for song in self.songs:
            if song["title"].lower() == item.lower():
                return True
        return False

playlist = Playlist()
playlist.add({"title": "Bohemian Rhapsody", "artist": "Queen"})
playlist.add({"title": "Stairway to Heaven", "artist": "Led Zeppelin"})

"bohemian rhapsody" in playlist  # True (忽略大小写)
"Imagine" in playlist            # False

// ========== 没有 __contains__ 时的回退 ==========
// Python 找不到 __contains__ 时会尝试 __iter__
// 再不行会尝试 __getitem__ (索引访问)

class MyContainer:
    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return iter(self.items)

3 in MyContainer([1, 2, 3])  # True (通过迭代检查)

// ========== in 的其他用法 ==========
// 检查子串:
"abc" in "abcdef"        # True

// 检查多个值:
any(x in [1, 2, 3] for x in [2, 5])  # True (2 在列表中)
all(x in [1, 2, 3] for x in [2, 5])  # False (5 不在)
```


> **Note:** 💡 身份与成员要点: (1) None 比较始终用 is/is not; (2) 值比较始终用 ==,不要用 is; (3) 大量查找时用集合替代列表; (4) 字典的 in 检查键而非值; (5) 实现 __contains__ 让自定义类支持 in。


## 练习


<!-- Converted from: 16_Python身份与成员运算符.html -->
