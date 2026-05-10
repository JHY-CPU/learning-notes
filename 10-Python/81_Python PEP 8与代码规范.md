# Python PEP 8与代码规范


## 📐 Python PEP 8 与代码规范


PEP 8 编码风格指南、PEP 257 文档字符串、代码格式化工具 black/autopep8、import 排序 isort、代码检查 flake8/pylint、pre-commit 钩子。


## PEP 8 核心规则


```
// ========== 缩进 ==========
# ✅ 4 个空格 (不要用 Tab)
def hello():
    print("缩进用 4 空格")

# ✅ 换行缩进:
result = (first_variable + second_variable +
          third_variable + fourth_variable)

# ✅ 括号对齐:
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)

// ========== 最大行宽 ==========
# 每行最多 79 字符 (文档/注释 72)
# 超长用括号/反斜杠换行

# ✅ 用括号隐式换行:
total = (price * quantity
         + tax - discount)

# ✅ 字符串拼接:
long_str = ("这是一个很长的字符串，"
            "用括号自动换行拼接")

// ========== 空行 ==========
# 顶层函数/类之间: 2 个空行
# 类中方法之间: 1 个空行
# 相关代码块: 可加空行分组

def func1():
    pass

def func2():
    pass

class MyClass:

    def method1(self):
        pass

    def method2(self):
        pass
```


## 命名规范


```
// ========== 命名约定 ==========
# 模块名:     short_and_snake  (小写+下划线)
# 包名:       short_and_snake  (小写+下划线, 无 __init__)

# 类名:       PascalCase (首字母大写)
# 异常类名:   PascalError 后缀

# 函数名:     snake_case (小写+下划线)
# 变量名:     snake_case
# 方法名:     snake_case

# 常量:       ALL_CAPS (全大写+下划线)
# 私有:       _private  (前导下划线)
# 强私有:     __mangled (双下划线,名称修饰)
# 魔术方法:   __init__, __str__ (双下划线包围)

# 避免: l (小写 L), O (大写 O), I (大写 i)

// ========== 示例 ==========
MAX_RETRIES = 3                  # 常量
DEFAULT_TIMEOUT = 30

class UserAccount:               # 类名 PascalCase
    """用户账户"""

    def __init__(self, name):    # 方法 snake_case
        self._name = name        # 私有属性

    def get_display_name(self) -> str:
        return self._name

def calculate_total(items):      # 函数 snake_case
    pass
```


## 导入规范


```
// ========== 导入顺序 ==========
# 标准库 → 第三方 → 本地模块
# 每组之间空行分隔,每组按字母排序

# ✅ 正确的导入结构:
import os
import sys
from datetime import datetime
from typing import List, Optional

import flask
import requests

from mypackage import config
from myproject.models import User

# ✅ 一行一个 import
# ❌ import os, sys  (不推荐)

# ✅ 绝对导入优先
from mypackage import module  # 而不是 import mypackage.module

# ✅ 从模块导入具体内容
from os.path import join, exists

// ========== isort 自动排序 ==========
# 安装: pip install isort
# 使用: isort myfile.py
# 自动分组排序 import 语句

# isort 配置 (pyproject.toml):
# [tool.isort]
# profile = "black"
```


## 文档字符串 (PEP 257)


```
// ========== docstring 规范 ==========

# ✅ 模块文档:
"""模块功能说明。

这个模块提供了 XXX 功能,
包含 XXX 和 XXX 等类。
"""

# ✅ 函数文档:
def fetch_data(url: str, timeout: int = 30) -> dict:
    """获取远程数据。

    Args:
        url: 请求地址。
        timeout: 超时秒数,默认 30。

    Returns:
        响应数据字典。

    Raises:
        ValueError: url 为空时抛出。
        ConnectionError: 连接失败时抛出。
    """
    ...

# ✅ 类文档:
class User:
    """表示一个用户。

    Attributes:
        name: 用户名。
        email: 邮箱地址。
    """

    def __init__(self, name: str, email: str):
        """初始化 User。

        Args:
            name: 用户名。
            email: 邮箱地址。
        """
        ...
        self.name = name
        self.email = email

# ✅ 单行文档 (简单函数):
def add(a: int, b: int) -> int:
    """返回 a 和 b 的和。"""
    return a + b
```


## 代码格式化工具


```
// ========== black ==========
# "不妥协的代码格式化器"
# pip install black
# black myfile.py
# black .  # 格式化整个项目

# black 自动格式化:
# - 统一缩进/空格/换行
# - 统一引号风格 (双引号)
# - 自动换行到指定宽度 (默认 88)

# 配置 pyproject.toml:
# [tool.black]
# line-length = 88
# target-version = ["py310"]
# include = '\.pyi?$'

// ========== flake8 ==========
# 代码检查 (lint)
# pip install flake8
# flake8 myfile.py

# 检查: 语法错误 / PEP 8 违规 / 逻辑问题
# 配置 .flake8:
# [flake8]
# max-line-length = 88
# extend-ignore = E203, W503
# exclude = .venv,__pycache__

// ========== pylint ==========
# 更深入的代码分析
# pip install pylint
# pylint myfile.py

# 检查: 命名/重复代码/复杂度/未使用变量
# 评分系统: 10/10 是最高分

// ========== pre-commit ==========
# git 提交前自动检查
# pip install pre-commit

# 配置 .pre-commit-config.yaml:
# repos:
#   - repo: https://github.com/psf/black
#     rev: 23.1.0
#     hooks:
#       - id: black

# 安装: pre-commit install
```


## 最佳实践汇总


```
// ========== 好代码 vs 坏代码 ==========

# ❌ 坏代码:
def f(x):
    if x>5:
     print(">5")
        else:
  print("<=5")

# ✅ 好代码:
def compare_value(x: int) -> None:
    """比较 x 与 5 的大小。"""
    if x > 5:
        print("大于 5")
    else:
        print("小于等于 5")

// ========== 更多规范 ==========
# ✅ 用 is 比较 None, 而不是 ==
if x is None:     # ✅
if x == None:     # ❌

# ✅ 用 is not 而不是 not ... is
if x is not None:  # ✅
if not x is None:  # ❌

# ✅ 布尔值判断:
if is_active:      # ✅
if is_active is True:   # ❌

if not items:      # ✅ (空序列 = False)
if len(items) == 0:      # ❌

# ✅ 字符串判空:
if text:           # ✅
if text != "":     # ❌

# ✅ 用 with 打开文件:
with open("f.txt") as f:   # ✅
    data = f.read()

f = open("f.txt")          # ❌
data = f.read()
f.close()

# ✅ 链式比较:
if 0 < x < 10:     # ✅
if x > 0 and x < 10:   # ❌

// ========== 日常工作流 ==========
# 1. 编写代码
# 2. isort file.py    # 排序 import
# 3. black file.py    # 格式化代码
# 4. flake8 file.py   # 检查规范
# 5. mypy file.py     # 类型检查
# 6. pytest           # 运行测试
```


> **Note:** 💡 PEP 8 要点: (1) 4 空格缩进,79 字符行宽; (2) 命名: 类 PascalCase,函数/变量 snake_case,常量 ALL_CAPS; (3) 导入顺序: 标准库→第三方→本地; (4) Pythonic: is None/is not None/if items/with 语句; (5) 工具链: black + isort + flake8 + mypy + pre-commit。


## 练习


<!-- Converted from: 81_Python PEP 8与代码规范.html -->
