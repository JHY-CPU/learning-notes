# Python包结构与__init__.py


## 📂 Python 包结构与 __init__.py


包 (package) 概念、__init__.py 作用与用法、包设计模式、命名空间包、__all__ 控制导出、python -m 执行包。


## 包基础


```
// ========== 什么是包 ==========
// 包 = 包含 __init__.py 的目录
// 包是"模块的集合",用于组织相关模块

// 典型包结构:
# mypackage/                ← 包 (目录)
#   __init__.py            ← 包的初始化文件
#   module_a.py            ← 子模块
#   module_b.py
#   subpackage/            ← 子包
#       __init__.py
#       module_c.py

// 导入方式:
import mypackage             # 导入包
import mypackage.module_a    # 导入子模块
from mypackage import module_a
from mypackage.module_a import func
from mypackage.subpackage import module_c

// ========== __init__.py 的作用 ==========
// 1. 标记目录为 Python 包 (Python 3.3+ 不是必须的)
// 2. 包初始化代码 (导入时自动执行)
// 3. 控制 from package import * 的内容
// 4. 提供便捷导入

// __init__.py 最简单的形式: 空文件即可
// 但通常包含包文档和便捷导入

// ========== 简单示例 ==========
# mypackage/__init__.py
"""
mypackage - 示例包

提供数学工具函数
"""

from .module_a import add, subtract
from .module_b import multiply, divide

__version__ = "1.0.0"
__all__ = ["add", "subtract", "multiply", "divide"]

# 导入包后可以直接使用:
# import mypackage
# mypackage.add(1, 2)       # ✅ 由 __init__.py 提供
```


## __init__.py 的常见模式


```
// ========== 模式 1: 便捷导入 ==========
// __init__.py 将深层模块的函数提升到包级别

# 没有 __init__.py 时:
from mypackage.subpackage.deep import func
# 太长!

# 有 __init__.py 时:
from .subpackage.deep import func

# 用户可以:
from mypackage import func   # 更简洁!

// ========== 模式 2: 控制 __all__ ==========
# __init__.py
__all__ = ["add", "subtract"]
# from mypackage import * 只导入 add, subtract

// ========== 模式 3: 惰性导入 ==========
# 只有被访问时才导入子模块

# __init__.py
def __getattr__(name):
    """惰性导入: 第一次访问时才导入"""
    import importlib
    if name in ["module_a", "module_b"]:
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 使用:
# import mypackage
# mypackage.module_a     # 第一次访问时自动导入

// ========== 模式 4: 包配置 ==========
# __init__.py
import logging
from .config import load_config
from .version import __version__

# 包初始化时配置日志
logging.getLogger(__name__).addHandler(logging.NullHandler())

# 加载默认配置
config = load_config()
```


## 包设计模式


```
// ========== 推荐包结构 ==========
# myproject/
#   README.md
#   pyproject.toml           ← 项目配置 (现代)
#   setup.py                 ← 旧版配置
#   requirements.txt         ← 依赖列表
#   mypackage/               ← 源码包
#       __init__.py
#       __main__.py          ← python -m 入口
#       cli.py               ← 命令行接口
#       config.py            ← 配置
#       models.py            ← 数据模型
#       utils.py             ← 工具函数
#       subpackage/          ← 子包
#           __init__.py
#           ...
#   tests/                   ← 测试
#       __init__.py
#       test_models.py
#       test_utils.py

// ========== __main__.py ==========
# 让包可以直接运行: python -m mypackage

# mypackage/__main__.py
from mypackage.cli import main
main()

# 运行:
# python -m mypackage
# 相当于执行 __main__.py

// ========== 命名空间包 ==========
// Python 3.3+: 不需要 __init__.py 的包
// 多个目录可以组成同一个包!

# 路径1: /path1/mypackage/module_a.py
# 路径2: /path2/mypackage/module_b.py

# 如果两个目录都在 sys.path 中:
import mypackage.module_a   # 从 /path1 导入
import mypackage.module_b   # 从 /path2 导入

# 这就是命名空间包: 一个包分散在多个目录中
# 适用场景: 插件系统、分布式包

// ========== 包导入顺序 ==========
// import mypackage.sub.module
// 1. 导入 mypackage (执行 __init__.py)
// 2. 导入 mypackage.sub (执行 __init__.py)
// 3. 导入 mypackage.sub.module (执行 module.py)
```


## 包管理最佳实践


```
// ========== 包 vs 模块 选择 ==========
// 用模块: 简单功能,单文件就够了
// 用包: 需要多个相关模块,或大型项目

// 模块 → 包的演进:
# 1. 单模块: utils.py
# 2. 拆包: utils/
#     __init__.py
#     strings.py
#     files.py
#     net.py
# 3. __init__.py 重新导出常用函数

// ========== __init__.py 使用建议 ==========
// ✅ 写文档字符串 (包的说明)
// ✅ 导出常用函数到包级别
// ✅ 定义 __all__ 控制导出
// ✅ 包版本信息 __version__
// ❌ 不要在 __init__.py 中写逻辑代码
// ❌ 不要自动导入所有子模块 (减慢加载)
// ❌ 不要做耗时操作 (包导入多次)

// ========== 包版本管理 ==========
# mypackage/version.py
__version__ = "1.2.3"
__author__ = "Alice"

# mypackage/__init__.py
from .version import __version__, __author__

# 用户查看版本:
# import mypackage
# print(mypackage.__version__)

// ========== 绝对导入 vs 相对导入 ==========
# 推荐: 绝对导入 (清晰,不易错)
from mypackage.module_a import func
from mypackage.subpackage import module_c

# 相对导入 (只能在包内)
from .module_a import func
from ..subpackage import module_c

# 绝对导入更推荐:
# - 移动模块时不易出错
# - 更清晰
# - Python 官方推荐
```


> **Note:** 💡 包要点: (1) 包 = 含 __init__.py 的目录; (2) __init__.py 控制包的导入接口; (3) __main__.py 支持 python -m 执行; (4) __all__ 控制 from xxx import * 的行为; (5) 推荐绝对导入,__init__.py 保持轻量。


## 练习


<!-- Converted from: 69_Python包结构与__init__py.html -->
