# Python上下文管理器with


## 🔧 Python 上下文管理器 with


with 语句与上下文管理器协议 (__enter__/__exit__)、@contextmanager 装饰器、contextlib 工具 (closing/suppress/redirect_stdout/ExitStack)、实际应用场景。


## with 语句基础


```
// ========== with 语句 ==========
// with 管理资源: 进入时自动获取,退出时自动释放
// 最常见的用途: 文件操作

// ❌ 手动管理:
f = open("file.txt", "w")
try:
    f.write("Hello")
finally:
    f.close()                  # 必须确保关闭

// ✅ with 自动管理:
with open("file.txt", "w") as f:
    f.write("Hello")
    # 退出 with 块自动 f.close()

// ========== with 多个资源 ==========
// Python 3.1+ 支持多个上下文管理器

# 方式 1: 嵌套
with open("a.txt") as fa:
    with open("b.txt") as fb:
        data_a = fa.read()
        data_b = fb.read()

# 方式 2: 并列 (推荐)
with open("a.txt") as fa, open("b.txt") as fb:
    data_a = fa.read()
    data_b = fb.read()

// 方式 3: 多行 (3.10+)
with (
    open("a.txt") as fa,
    open("b.txt") as fb,
    open("c.txt") as fc,
):
    pass

// ========== as 变量作用域 ==========
// as 变量的作用域是整个 with 块
// Python 3.10+: as 变量在 with 块外也可用

with open("test.txt", "w") as f:
    f.write("Hello")

# Python 3.10+, f 在这里仍可访问 (但不建议)
# 文件已关闭,不能再读写
```


## 上下文管理器协议


```
// ========== __enter__ 和 __exit__ ==========
// 任何对象只要有这两个方法就是上下文管理器

class ManagedFile:
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        """进入 with 块时调用"""
        print(f"打开: {self.filename}")
        self.file = open(self.filename, self.mode, encoding="utf-8")
        return self.file        # 绑定到 as 变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用 (总是执行)"""
        print(f"关闭: {self.filename}")
        if self.file:
            self.file.close()

        # exc_type: 异常类型 (没有异常则为 None)
        # exc_val: 异常对象
        # exc_tb: traceback

        # 返回 True → 抑制异常
        # 返回 False/None → 传播异常

        if exc_type is ValueError:
            print(f"忽略 ValueError: {exc_val}")
            return True         # 抑制异常

        return False            # 其他异常继续传播

// 使用:
with ManagedFile("test.txt", "w") as f:
    f.write("Hello!")

// 即使 with 块中有异常,__exit__ 也会执行
with ManagedFile("test.txt", "w") as f:
    raise ValueError("测试")   # 会被 __exit__ 抑制
print("继续执行!")              # 这里还能执行到!
```


## @contextmanager 装饰器


```
// ========== 用生成器实现上下文管理器 ==========
// 比写类更简洁!
// yield 之前的代码 = __enter__, 之后的 = __exit__

from contextlib import contextmanager

@contextmanager
def managed_file(filename, mode="r"):
    """文件上下文管理器 (生成器版)"""
    print(f"进入: 打开 {filename}")
    f = open(filename, mode, encoding="utf-8")
    try:
        yield f                 # as 变量的值
    finally:
        print(f"退出: 关闭 {filename}")
        f.close()

with managed_file("test.txt", "w") as f:
    f.write("Hello!")

// 即使异常也会执行 finally:
with managed_file("test.txt", "w") as f:
    f.write("Hello")
    raise RuntimeError("错误!")

// 输出:
// 进入: 打开 test.txt
// 退出: 关闭 test.txt
// (然后异常继续传播)

// ========== 捕获异常 ==========
@contextmanager
def suppress_error(*errors):
    """抑制指定异常"""
    try:
        yield
    except errors as e:
        print(f"抑制异常: {e}")

with suppress_error(ValueError, TypeError):
    raise ValueError("这个被抑制了")
print("正常继续!")

// ========== 实用例子: 计时器 ==========
import time

@contextmanager
def timer(name="任务"):
    """计时上下文管理器"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} 耗时: {elapsed:.3f}s")

with timer("数据库查询"):
    time.sleep(0.5)             # 模拟耗时操作
```


## contextlib 工具函数


```
// ========== contextlib 实用工具 ==========
from contextlib import closing, suppress, redirect_stdout, ExitStack

// 1. closing — 自动调用 close()
# 用于没有实现上下文管理器但有 close() 的对象
from urllib.request import urlopen

with closing(urlopen("https://example.com")) as resp:
    data = resp.read()
# 等价于: resp.close() 自动调用

// 2. suppress — 忽略指定异常
# 替代 try/except pass
from contextlib import suppress

# 不用:
try:
    os.remove("temp.txt")
except FileNotFoundError:
    pass

# 用 suppress:
with suppress(FileNotFoundError):
    os.remove("temp.txt")

// 3. redirect_stdout — 临时重定向输出
import io

buf = io.StringIO()
with redirect_stdout(buf):
    print("Hello")
    print("World")

output = buf.getvalue()
print(repr(output))            # "Hello\nWorld\n"

// 4. ExitStack — 动态管理多个上下文管理器
# 当你不知道有多少资源需要管理时使用
from contextlib import ExitStack

def process_files(filenames):
    """同时打开多个文件"""
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(f, encoding="utf-8"))
            for f in filenames
        ]
        # 所有文件自动关闭
        return [f.read() for f in files]

// ExitStack 还支持:
// - callback(): 注册退出时调用的函数
// - pop_all(): 转移管理的资源
// - push(): 添加自定义退出函数
```


> **Note:** 💡 with 要点: (1) with 自动管理资源,自动调用 __enter__/__exit__; (2) 自定义上下文管理器: 写类实现协议或用 @contextmanager 装饰器; (3) __exit__ 返回 True 抑制异常; (4) contextlib 提供 closing/suppress/ExitStack 等工具; (5) with 管理文件/锁/数据库连接/网络连接等资源。


## 练习


<!-- Converted from: 62_Python上下文管理器with.html -->
