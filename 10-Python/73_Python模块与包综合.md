# Python模块与包综合


## 📦 Python 模块与包综合


模块与包知识全景、综合实战: 从零搭建 Python 项目、模块设计原则、导入性能优化、标准库推荐、完整项目模板。


## 模块与包知识全景


```
// ========== 模块与包知识点结构 ==========
// 554 模块基础
// ├── import 6 种语法
// ├── sys.path 模块搜索路径
// ├── __name__ == "__main__"
// ├── sys.modules 模块缓存
// └── 导入陷阱 (循环/命名冲突)
//
// 555 包结构
// ├── __init__.py 作用
// ├── __all__ 控制导出
// ├── __main__.py 入口
// ├── 命名空间包
// └── 相对导入 vs 绝对导入
//
// 556 pip
// ├── install/uninstall/list/show
// ├── freeze > requirements.txt
// ├── 国内镜像源
// └── pip vs pipenv vs poetry
//
// 557 虚拟环境
// ├── python -m venv .venv
// ├── activate/deactivate
// ├── 项目隔离
// └── 工具对比
//
// 558 依赖管理
// ├── 版本锁定策略
// ├── pip-compile / pip-sync
// ├── pyproject.toml
// └── 安全审计 pip-audit
```


## 综合实战: 项目脚手架


```
// ========== 从零搭建 Python 项目 ==========
# myproject/
# ├── pyproject.toml         # 项目配置
# ├── README.md              # 项目说明
# ├── .gitignore             # Git 忽略
# ├── .venv/                 # 虚拟环境 (不提交)
# ├── src/
# │   └── myproject/
# │       ├── __init__.py
# │       ├── __main__.py
# │       ├── config.py
# │       ├── models.py
# │       └── utils.py
# ├── tests/
# │   ├── __init__.py
# │   ├── test_config.py
# │   └── test_models.py
# └── requirements/
#     ├── base.txt
#     ├── dev.txt
#     └── prod.txt

// ========== 搭建步骤 ==========
# 1. 创建项目目录
mkdir myproject && cd myproject

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. 创建项目结构
mkdir -p src/myproject tests requirements

# 4. 创建 pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "0.1.0"
dependencies = []
EOF

# 5. 安装项目 (可编辑模式)
pip install -e .

# 6. 初始化 git
git init
echo ".venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
```


## 模块设计原则


```
// ========== 模块设计原则 ==========
// 1. 单一职责: 一个模块只做一件事
// 2. 接口清晰: 明确导出的内容
// 3. 最小依赖: 减少 import 数量
// 4. 稳定接口: 公共 API 保持向后兼容

// ========== __init__.py 的设计 ==========
# ✅ 好的 __init__.py:
"""mypackage - 简洁的包说明"""

from .models import User, Product
from .utils import format_date, validate_email

__all__ = ["User", "Product", "format_date", "validate_email"]
__version__ = "0.1.0"

# ❌ 不好的 __init__.py:
# 耗时操作
# import 过多非必要模块
# 全局副作用

// ========== 导入性能优化 ==========
# 1. 延迟导入 (需要时才导入)
def process_file(filename):
    from json import load      # 函数内导入
    with open(filename) as f:
        return load(f)

# 2. 只导入需要的
# ✅ from os.path import join
# ❌ import os (然后 os.path.join)

# 3. 用 __all__ 控制导入
# 4. 避免 import *
# 5. 了解 import 的耗时

# 测量导入时间:
import timeit
# python -m timeit "import requests"
```


## 标准库推荐与项目结构


```
// ========== 常用标准库推荐 ==========
# 路径处理: pathlib (优先), os.path
# 系统接口: os, sys, platform
# 日期时间: datetime, time, calendar
# 数学: math, random, statistics
# 文本: re, string, textwrap
# 数据: json, csv, pickle
# 集合: collections, heapq, bisect
# 迭代: itertools, functools
# 加密: hashlib, secrets
# 网络: urllib, http, socket
# 并发: threading, multiprocessing, asyncio
# 测试: unittest, doctest
# 日志: logging
# 配置: configparser, argparse
# 开发: pdb, cProfile, venv

// ========== 项目模板总结 ==========
# 小脚本 (< 100 行):
#   single_file.py
#
# 小项目 (1-5 文件):
#   project/
#   ├── main.py
#   ├── utils.py
#   └── requirements.txt
#
# 中项目 (> 5 文件):
#   project/
#   ├── src/project/
#   │   ├── __init__.py
#   │   ├── __main__.py
#   │   └── *.py
#   ├── tests/
#   ├── requirements/
#   └── pyproject.toml
#
# 库 (发布到 PyPI):
#   project/
#   ├── src/project/   # 源码
#   ├── tests/         # 测试
#   ├── docs/          # 文档
#   ├── pyproject.toml # 构建配置
#   └── README.md

// ========== 模块导出的最佳实践 ==========
# 每个模块应该明确定义它的公共接口

def public_function():
    """公共函数 (在 __all__ 中)"""
    pass

def _private_helper():
    """私有函数 (以下划线开头,不应被外部使用)"""
    pass

# 用户应该通过 __init__.py 的便捷导入使用
# 而不是直接导入深层模块
```


> **Note:** 💡 模块与包总结: (1) 好的项目结构 = 清晰的模块划分 + 明确的接口; (2) __init__.py 提供便捷导入,不要写耗时逻辑; (3) 私有函数用 _ 开头,__all__ 控制导出; (4) 用 pyproject.toml 替代 setup.py; (5) 每个项目创建虚拟环境 + requirements.txt。


## 练习


<!-- Converted from: 73_Python模块与包综合.html -->
