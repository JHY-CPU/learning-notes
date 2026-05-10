# Python sys模块


## ⚙️ Python sys 模块


sys 模块提供 Python 解释器相关的变量和函数：命令行参数 argv、退出 exit、搜索路径 path、已导入模块 modules、平台信息、标准输入/输出/错误流。


## 命令行参数 sys.argv


```
// ========== sys.argv ==========
import sys

# sys.argv: 命令行参数列表
# argv[0] = 脚本名称
# argv[1:] = 传入的参数

# python script.py --verbose --port 8080
print(sys.argv)
# ['script.py', '--verbose', '--port', '8080']

# 简易参数解析:
if len(sys.argv) > 1:
    if sys.argv[1] == "--help":
        print("用法: python script.py [options]")

# 更专业的参数解析用 argparse 模块
```


## 程序退出 sys.exit


```
// ========== sys.exit([code]) ==========
import sys

# 退出程序,可选退出码
sys.exit(0)          # 成功退出 (默认 0)
sys.exit(1)          # 非零表示错误
sys.exit("错误信息")  # 输出错误信息到 stderr,退出码 1

# exit 实际上抛出 SystemExit 异常
# 可以在 try 中捕获:
try:
    sys.exit(1)
except SystemExit as e:
    print(f"捕获退出: {e.code}")  # 1

# 0 = 成功, 非 0 = 失败
# 常用退出码: 1=一般错误, 2=语法错误
```


## 搜索路径 sys.path


```
// ========== sys.path ==========
import sys

# sys.path: 模块搜索路径列表
# Python import 时按此顺序查找

for p in sys.path:
    print(p)
# 输出类似:
# ''                          # 当前目录
# /usr/lib/python3.10/        # 标准库
# /usr/lib/python3.10/lib-dynload/
# /home/user/.venv/lib/python3.10/site-packages/  # 第三方包

# 动态添加搜索路径:
sys.path.append("/my/custom/path")
sys.path.insert(0, "/priority/path")  # 优先搜索

# 注意: sys.path 包含当前目录 '' (空字符串)
```


```
// ========== sys.modules ==========
import sys

# sys.modules: 已导入模块的缓存字典
# 所有 import 过的模块都在这里

print("math" in sys.modules)  # False (尚未导入)

import math
print("math" in sys.modules)  # True
print(sys.modules["math"])    #

# 查看已导入的模块数量:
print(f"已导入 {len(sys.modules)} 个模块")

# 可以删除缓存强制重新导入:
del sys.modules["some_module"]
# 注意: 一般不需要这样做
```


## 平台与版本信息


```
// ========== 平台信息 ==========
import sys

# 操作系统平台:
print(sys.platform)
# 'win32'   — Windows (即使是 64 位)
# 'linux'   — Linux
# 'darwin'  — macOS

# 跨平台代码:
if sys.platform == "win32":
    import ntpath
else:
    import posixpath

# ========== 版本信息 ==========
# Python 版本:
print(sys.version)
# '3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0]'

print(sys.version_info)
# sys.version_info(major=3, minor=10, micro=12, releaselevel='final', serial=0)

# 版本比较:
if sys.version_info >= (3, 10):
    print("Python 3.10+ 特性可用")

if sys.version_info < (3, 8):
    print("需要升级 Python!")

print(f"Python {sys.version_info.major}.{sys.version_info.minor}")

# 最大整数/递归:
print(sys.maxsize)           # 最大 int 值 (2^63-1 或 2^31-1)
print(sys.getrecursionlimit())  # 递归深度限制 (默认 1000)
sys.setrecursionlimit(2000)     # 设置新的限制
```


## 标准输入输出与错误流


```
// ========== sys.stdin / stdout / stderr ==========
import sys

# 标准输入:
data = sys.stdin.read()       # 读取全部输入
line = sys.stdin.readline()   # 读取一行

for line in sys.stdin:         # 逐行迭代
    print(line.strip())

# 标准输出:
sys.stdout.write("Hello\n")   # 同 print,但不自动换行

# 标准错误:
sys.stderr.write("错误!\n")   # 输出到 stderr

# 重定向输出:
with open("out.log", "w") as f:
    sys.stdout = f            # print 重定向到文件
    print("这条会写入文件")
    sys.stdout = sys.__stdout__  # 恢复原始 stdout

# 缓冲控制:
print("立即输出", flush=True)  # 强制刷新缓冲区
```


## 其他常用功能


```
// ========== 内存与对象 ==========
import sys

# 获取对象大小 (字节):
print(sys.getsizeof(42))          # 28 (小整数)
print(sys.getsizeof("hello"))     # 54
print(sys.getsizeof([]))          # 56 (空列表)
print(sys.getsizeof([1,2,3]))     # 120 (列表+元素)

# 引用计数:
a = []
print(sys.getrefcount(a))         # 2 (a 和 getrefcount 参数)

b = a
print(sys.getrefcount(a))         # 3

# 递归限制:
sys.setrecursionlimit(5000)       # 谨慎: 可能栈溢出

# ========== 编码与 Unicode ==========
print(sys.getdefaultencoding())   # 'utf-8'
print(sys.getfilesystemencoding())# 'utf-8' (Windows: 'mbcs')

# ========== 钩子 ==========
# 捕获未处理的异常:
def my_excepthook(exc_type, exc_value, traceback):
    print(f"未处理异常: {exc_type.__name__}: {exc_value}")

sys.excepthook = my_excepthook

# ========== sys.int_info / sys.float_info ==========
print(sys.int_info)     # 整数内部表示信息
print(sys.float_info)   # float 类型信息 (最大值/精度/最小值)

# ========== sys.getrecursionlimit / setrecursionlimit ==========
print(f"当前递归限制: {sys.getrecursionlimit()}")

# ========== sys.settrace ==========
# 设置追踪函数 (用于调试/性能分析):
def trace(frame, event, arg):
    print(f"{event}: {frame.f_code.co_name}")
    return trace

# sys.settrace(trace)  # 启动追踪
```


> **Note:** 💡 sys 模块要点: (1) sys.argv 获取命令行参数 (argv[0] 是脚本名); (2) sys.exit(code) 退出程序,可捕获 SystemExit; (3) sys.path 模块搜索路径,可动态添加; (4) sys.modules 是已导入模块缓存; (5) sys.platform 判断操作系统; (6) sys.version_info 版本比较; (7) sys.stdin/stdout/stderr 标准流。


## 练习


<!-- Converted from: 75_Python sys模块.html -->
