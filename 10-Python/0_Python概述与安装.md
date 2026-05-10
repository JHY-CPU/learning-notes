# Python概述与安装


## 🐍 Python 概述与安装


Python 语言特点、安装配置、版本选择、开发环境搭建、Hello World。


## Python 简介


```
// ========== Python 是什么 ==========
// Python 是 Guido van Rossum 于 1991 年创建的高级编程语言
// 设计哲学: 简洁、易读、明确 (The Zen of Python)

// ========== 特点 ==========
// - 解释型语言 (无需编译)
// - 动态类型 (变量无需声明类型)
// - 面向对象 (一切皆对象)
// - 自动内存管理 (垃圾回收 GC)
// - 丰富的标准库 ("内置电池")
// - 跨平台 (Windows/Linux/Mac)
// - 巨大的第三方生态 (PyPI)

// ========== 应用领域 ==========
// Web 开发      Django, Flask, FastAPI
// 数据科学      NumPy, Pandas, Matplotlib
// 机器学习      TensorFlow, PyTorch, scikit-learn
// 自动化运维     Ansible, SaltStack
// 爬虫          Scrapy, BeautifulSoup
// 测试          pytest, Selenium
// 游戏          Pygame
// 桌面应用      PyQt, Tkinter

// ========== Python 2 vs Python 3 ==========
// Python 2: 2000 年发布, 2020 年停止维护
// Python 3: 2008 年发布, 当前主流
// 关键区别:
//   print "hello"   → print("hello")
//   raw_input()     → input()
//   / (整数除法)     → / (浮点除法), // (整数除法)
//   unicode/str      → str (默认 unicode)
//   xrange()         → range()
//
// 注意: 务必使用 Python 3! Python 2 已死
```


## 安装 Python


```
// ========== 安装方式 ==========
// Windows:
//   1. python.org 下载安装包
//   2. 安装时勾选 "Add Python to PATH"
//   3. 或在 Microsoft Store 安装

// Mac:
//   1. brew install python3
//   2. 或 python.org 下载安装

// Linux (Ubuntu/Debian):
sudo apt install python3 python3-pip python3-venv

// Linux (CentOS/RHEL):
sudo yum install python3 python3-pip

// ========== 验证安装 ==========
python --version            # Windows 可能需要 python --version
python3 --version           # Linux/Mac
// Python 3.12.0

pip --version
// pip 23.3.1 from /usr/lib/python3.12/site-packages

// ========== 运行 Python ==========
// 1. REPL (交互式):
python3
>>> print("Hello, World!")
Hello, World!
>>> exit()

// 2. 脚本文件:
# hello.py
print("Hello, World!")

python3 hello.py

// 3. shebang (Linux/Mac):
// #!/usr/bin/env python3
// print("Hello from script")

chmod +x hello.py
./hello.py

// ========== 包管理 pip ==========
pip install requests              # 安装包
pip install flask==2.3.0          # 指定版本
pip install -r requirements.txt   # 从文件安装
pip list                          # 列出已安装
pip show flask                    # 查看包信息
pip uninstall flask               # 卸载
pip freeze > requirements.txt     # 导出依赖
```


## 开发环境


```
// ========== IDE / 编辑器 ==========
// VSCode:        最流行,插件丰富 (Python, Pylance)
// PyCharm:       专业 Python IDE (社区版免费)
// Jupyter Lab:   数据科学首选 (交互式笔记本)
// Vim/Neovim:    终端编辑器
// Thonny:        适合初学者

// VSCode 推荐插件:
// - Python (Microsoft)
// - Pylance (类型检查)
// - Python Docstring Generator
// - GitLens
// - Jupyter

// ========== 虚拟环境 ==========
// 隔离项目依赖,避免冲突
// 每个项目独立的 Python 环境

// Python 内置 venv:
python3 -m venv venv              # 创建虚拟环境
source venv/bin/activate          # 激活 (Linux/Mac)
venv\Scripts\activate             # 激活 (Windows)
deactivate                        # 退出

// 更好的选择: poetry
pip install poetry
poetry new myproject              # 创建新项目
poetry add flask                  # 添加依赖
poetry shell                      # 激活环境

// 或者 pipenv:
pip install pipenv
pipenv install flask              # 安装并自动创建虚拟环境
pipenv shell                      # 激活环境

// ========== Hello World ==========
// 第一个 Python 程序

# hello.py
"""我的第一个 Python 程序"""

print("Hello, World!")

name = input("请输入名字: ")
print(f"你好, {name}!")

# 运行: python3 hello.py
# 输出:
# Hello, World!
# 请输入名字: Alice
# 你好, Alice!
```


> **Note:** 💡 Python 环境最佳实践: (1) 每个项目用独立的虚拟环境; (2) 用 python3 而不是 python 命令; (3) 用 pip freeze > requirements.txt 锁定依赖版本; (4) 初学者建议用 VSCode + Python 插件,上手最快。


## 练习


<!-- Converted from: 0_Python概述与安装.html -->
