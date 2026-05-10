# Python pip包管理器


## 📦 Python pip 包管理器


pip 基础命令 (install/uninstall/list/freeze/show)、版本管理、requirements.txt、国内镜像源、依赖冲突解决、pip vs pipenv/poetry。


## pip 基础命令


```
// ========== pip 是什么 ==========
// pip: Python 包管理器 (Python 3.4+ 自带)
// 从 PyPI (Python Package Index) 安装第三方包
// PyPI 地址: https://pypi.org/

// ========== 常用命令 ==========
# 安装包:
pip install requests           # 安装最新版
pip install requests==2.28.0   # 指定版本
pip install "requests>=2.0"    # 版本范围
pip install requests flask     # 同时安装多个

# 安装可选依赖:
pip install requests[security] # 安装 extras

# 从文件安装:
pip install -r requirements.txt

# 卸载包:
pip uninstall requests

# 列出已安装:
pip list                       # 列出所有
pip list --outdated            # 列出可更新的

# 查看包信息:
pip show requests
# Name: requests
# Version: 2.28.0
# Summary: Python HTTP for Humans.
# Location: /usr/lib/python3.10/site-packages
# Requires: certifi, charset-normalizer, idna, urllib3

# 搜索包:
pip search "web framework"     # 已弃用,用 pypi.org 搜索

// ========== pip freeze ==========
# 导出当前环境的包列表:
pip freeze                     # 标准格式
pip freeze > requirements.txt  # 导出到文件

# pip freeze 输出示例:
# requests==2.28.0
# flask==2.2.0
# numpy>=1.20.0
```


## requirements.txt 详解


```
// ========== requirements.txt 格式 ==========
# 这是注释
requests==2.28.0              # 精确版本
flask>=2.0,<3.0               # 版本范围
numpy                         # 最新版
pandas~=1.4.0                 # 兼容版本 (>=1.4.0, <1.5.0)

# 仅特定平台:
pywin32; sys_platform == "win32"    # 仅 Windows

# 额外选项:
package[extra1,extra2]

# 从其他文件引入:
-r base-requirements.txt

# Git 仓库:
git+https://github.com/user/repo.git@branch
git+https://github.com/user/repo.git@v1.0.0#egg=package

# 本地路径:
./local-packages/mypackage/
../other-project/

// ========== 版本符号 ==========
# == 2.28.0    — 精确版本
# >= 2.28.0    — 大于等于
# <= 2.28.0    — 小于等于
# > 2.28.0     — 大于
# < 2.28.0     — 小于
# != 2.28.0    — 不等于
# ~= 2.28.0    — 兼容版本 (>=2.28.0, ==2.28.*)
# *            — 通配符 (2.28.*)

// ========== 多环境 requirements ==========
# requirements.txt           — 生产依赖
# requirements-dev.txt       — 开发依赖
# requirements-test.txt      — 测试依赖

# requirements-dev.txt:
-r requirements.txt
pytest==7.0.0
black==22.0.0
flake8==5.0.0
```


## 镜像源与环境配置


```
// ========== 国内镜像源 ==========
# 国内下载速度慢,使用镜像源:

# 临时使用:
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 常用镜像:
# 清华: https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里: https://mirrors.aliyun.com/pypi/simple/
# 豆瓣: https://pypi.douban.com/simple/
# 中科大: https://pypi.mirrors.ustc.edu.cn/simple/

# 永久配置:
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 查看配置:
pip config list

# 配置文件位置:
# Linux/macOS: ~/.config/pip/pip.conf
# Windows: %APPDATA%\pip\pip.ini

// ========== pip 配置 ==========
# pip.conf / pip.ini 格式:
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn

// ========== pip 高级用法 ==========
# 列出需要更新的包:
pip list --outdated

# 升级包:
pip install --upgrade requests

# 升级 pip 本身:
pip install --upgrade pip

# 下载但不安装:
pip download requests -d ./packages

# 从本地目录安装:
pip install --no-index --find-links=./packages requests

# 导出依赖树:
pipdeptree                  # 需先安装 pipdeptree
```


## 依赖管理与工具对比


```
// ========== 常见问题 ==========
// 问题 1: 依赖冲突
# pip 不解决深层依赖冲突
# 用 pipdeptree 查看依赖树:
pip install pipdeptree
pipdeptree                   # 显示依赖树

// 问题 2: 版本锁定
# 生产环境锁定所有依赖版本:
pip freeze > requirements.txt
# 会列出所有包 (包括间接依赖)

// 问题 3: 跨平台
# 生成跨平台兼容的 requirements:
pip freeze | grep -v "pywin32" > requirements.txt

// ========== pip vs 其他工具 ==========
// pip:   基础包管理器,简单直接
// pipenv: pip + virtualenv + Pipfile.lock
// poetry: 新一代包管理,依赖解析 + 构建 + 发布
// conda: 科学计算包管理,跨语言

// pipenv 示例:
pipenv install requests       # 安装到虚拟环境
pipenv shell                  # 进入虚拟环境
pipenv lock                   # 生成 Pipfile.lock

// poetry 示例:
poetry new myproject          # 创建项目
poetry add requests           # 添加依赖
poetry install                # 安装依赖
poetry build                  # 构建包

// ========== 包安装位置 ==========
# 查看 site-packages 位置:
python -c "import site; print(site.getsitepackages())"

# 查看当前 Python 路径:
which python                  # Linux/macOS
where python                 # Windows
```


> **Note:** 💡 pip 要点: (1) pip install/uninstall/list/freeze/show 是核心命令; (2) requirements.txt 管理依赖,版本符号 ==/>=/~=/<; (3) 用国内镜像源加速下载; (4) pip freeze > requirements.txt 导出环境; (5) 现代项目推荐 poetry,简单项目用 pip + venv。


## 练习


<!-- Converted from: 70_Python pip包管理器.html -->
