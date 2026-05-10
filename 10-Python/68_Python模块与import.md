# Python模块与import


## 📦 Python 模块与 import


模块概念、import 语法 (6 种形式)、模块搜索路径 sys.path、__name__ == "__main__"、模块缓存 sys.modules、相对导入。


## 模块基础


```
// ========== 什么是模块 ==========
// 模块 = .py 文件
// 模块名 = 文件名 (不含 .py)
// 模块是 Python 组织代码的基本单位

// ========== 6 种 import 语法 ==========
// 1. 导入整个模块
import math
print(math.sqrt(16))          # 4.0
print(math.pi)                # 3.14159...

// 2. 导入模块并用别名
import datetime as dt
now = dt.datetime.now()

// 3. 导入特定名称
from math import sqrt, pi
print(sqrt(16))               # 4.0 (直接使用,不用 math.)
print(pi)                     # 3.14159

// 4. 导入并别名
from math import sqrt as square_root
print(square_root(16))        # 4.0

// 5. 导入所有 (不推荐!)
from math import *            # 污染命名空间!
# 你不知道导入了什么,可能覆盖已有名称

// 6. 导入包/子模块
import os.path
print(os.path.join("a", "b"))

// ========== 模块搜索路径 ==========
import sys

print(sys.path)               # Python 搜索模块的路径列表

// sys.path 包含:
// 1. 当前脚本所在目录 (或当前工作目录)
// 2. PYTHONPATH 环境变量中的路径
// 3. 标准库路径 (如 /usr/lib/python3.x)
// 4. site-packages (pip 安装的第三方包)

// 临时添加路径:
sys.path.append("/my/custom/path")
import mymodule               # 可以从此路径导入

// 查看模块文件位置:
print(math.__file__)          # 模块文件的实际路径
```


## __name__ == "__main__"


```
// ========== __name__ 的作用 ==========
// 每个模块都有 __name__ 属性
// - 直接运行时: __name__ = "__main__"
// - 被导入时:   __name__ = 模块名

# mymodule.py
print(f"__name__ = {__name__}")

if __name__ == "__main__":
    print("这个文件被直接运行了")
    # 测试代码或命令行入口
else:
    print("这个文件被导入了")

// 运行 python mymodule.py:
// __name__ = __main__
// 这个文件被直接运行了

// 在另一个文件中导入:
// import mymodule
// __name__ = mymodule
// 这个文件被导入了

// ========== 典型结构 ==========
# 每个 Python 文件的推荐结构:

"""
模块文档字符串
"""
import sys
import os

# 常量
DEFAULT_TIMEOUT = 30

# 函数
def main():
    """主函数"""
    print("程序入口")

def helper():
    """辅助函数"""
    pass

# 命令行入口
if __name__ == "__main__":
    main()

// 好处:
// - 文件既可被导入使用函数,也可直接运行
// - 测试代码放在 if 块中不污染导入
// - 清晰的模块入口点
```


## 模块缓存 sys.modules


```
// ========== 模块只导入一次 ==========
// Python 在第一次 import 时执行模块代码
// 后续导入直接从 sys.modules 获取缓存

import sys
print("math" in sys.modules)  # False (还未导入)

import math
print("math" in sys.modules)  # True (已缓存)

# 可以查看所有已导入模块:
print(list(sys.modules.keys())[:10])

// ========== 重新加载模块 ==========
# import 不会重新执行模块
# 用 importlib.reload() 强制重载

import importlib

# 假设修改了 mymodule.py
import mymodule
importlib.reload(mymodule)    # 强制重新加载

// reload 的用途:
// - 开发调试时热重载
// - Jupyter notebook 中重载
// - 动态插件系统

// ========== import 的执行过程 ==========
// 1. 检查 sys.modules 是否已有该模块
//    - 有 → 直接返回,不执行
// 2. 在 sys.path 中搜索模块文件
// 3. 创建模块对象
// 4. 将模块加入 sys.modules
// 5. 执行模块代码 (函数定义、类定义等)
// 6. 返回模块对象

// 这就是为什么模块中的 print 只在第一次 import 时执行
```


## 相对导入与导入陷阱


```
// ========== 相对导入 ==========
// 只能在包内使用 (不能直接在脚本中运行)
// 用 . 表示当前包, .. 表示上级包

# 目录结构:
# mypackage/
#   __init__.py
#   module_a.py
#   subpackage/
#     __init__.py
#     module_b.py

# module_b.py 中导入 module_a:
from .. import module_a         # 上级包
from ..module_a import func     # 上级包的模块

# .  — 当前包
# .. — 上级包
# ... — 上上级包

// 相对导入只能在包中使用
// 直接运行脚本会报错:
// "ImportError: attempted relative import with no known parent package"

// ========== 常见导入陷阱 ==========
// 陷阱 1: 循环导入

# a.py:
from b import func_b
def func_a():
    return func_b()

# b.py:
from a import func_a           # ❌ 循环导入!
def func_b():
    return func_a()

# 修复: 延迟导入 (在函数内导入)
def func_b():
    from a import func_a
    return func_a()

// 陷阱 2: 命名冲突
# from math import *  # 不导入 __ 开头的
# from math import sin
# sin = "覆盖了sin"   # ❌ 覆盖了导入的函数!

// 陷阱 3: 包路径问题
# 直接运行包内脚本可能导致导入失败
# 推荐可执行入口:

# mypackage/__main__.py
from mypackage.cli import main
main()

# 运行: python -m mypackage
```


> **Note:** 💡 模块要点: (1) 每个 .py 文件就是一个模块; (2) __name__=="__main__" 判断是否直接运行; (3) 模块只导入一次,由 sys.modules 缓存; (4) 用 sys.path 查看/添加模块搜索路径; (5) 相对导入在包内使用,避免循环导入。


## 练习


<!-- Converted from: 68_Python模块与import.html -->
