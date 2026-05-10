# Python虚拟环境venv


## 🔒 Python 虚拟环境 venv


为什么需要虚拟环境、venv 创建与使用、激活/退出虚拟环境、pipenv/poetry/conda 对比、最佳实践、常见问题。


## 为什么需要虚拟环境


```
// ========== 问题场景 ==========
// 项目 A: Django 4.0, Python 3.10
// 项目 B: Django 2.2, Python 3.8
//
// 没有虚拟环境: 全局只装一个 Django 版本
// 项目 A 需要 4.0, 项目 B 需要 2.2
// → 冲突! 不能同时满足

// 虚拟环境解决:
// 每个项目有自己的 Python 环境
// 互不干扰的包安装空间

// ========== 虚拟环境原理 ==========
// 虚拟环境 = 独立的 Python 环境副本
//
// 包含:
// - 独立的 Python 解释器 (链接/复制)
// - 独立的 site-packages (包安装位置)
// - 独立的 pip
//
// 激活后:
// - PATH 修改: 优先使用虚拟环境的 Python 和 pip
// - 安装的包只在当前虚拟环境中可用
//
// 结构:
// .venv/
//   ├── bin/ (Windows: Scripts/)
//   │   ├── python          # Python 解释器
//   │   ├── pip             # 包管理器
//   │   └── activate        # 激活脚本
//   ├── lib/
//   │   └── python3.x/
//   │       └── site-packages/  # 包安装位置
//   └── pyvenv.cfg          # 配置文件
```


## venv 基本操作


```
// ========== 创建虚拟环境 ==========
# Python 3.3+ 自带 venv 模块

# 创建:
python -m venv .venv           # 创建名为 .venv 的虚拟环境

# 也可以指定 Python 版本:
python3.10 -m venv .venv       # 用 Python 3.10 创建

# 创建时不安装 pip:
python -m venv .venv --without-pip

# 创建可重定位的环境 (不常用):
python -m venv .venv --relocatable

// ========== 激活/退出 ==========
# Windows (CMD):
.venv\Scripts\activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

# 激活后,终端提示符会显示环境名:
# (.venv) $

# 验证:
which python                   # .venv/bin/python
pip list                       # 空的! (无全局包)

# 退出虚拟环境:
deactivate

// ========== 使用虚拟环境 ==========
# 1. 创建
python -m venv .venv

# 2. 激活
source .venv/bin/activate      # Linux
# .venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install requests flask

# 4. 导出依赖
pip freeze > requirements.txt

# 5. 退出
deactivate
```


## pipenv 与 poetry


```
// ========== pipenv ==========
# pip + virtualenv + Pipfile 的整合
# 自动管理虚拟环境

# 安装:
pip install pipenv

# 创建虚拟环境并安装包:
pipenv install requests        # 自动创建虚拟环境
pipenv install --dev pytest    # 开发依赖

# 激活虚拟环境:
pipenv shell                   # 进入 shell

# 运行命令 (无需激活):
pipenv run python main.py

# 生成 Pipfile.lock (锁定版本):
pipenv lock

# 根据 Pipfile 安装:
pipenv install                 # 生产依赖
pipenv install --dev           # 全部依赖

# 删除虚拟环境:
pipenv --rm

// ========== poetry ==========
# 新一代包管理工具
# 统一: 项目管理 + 依赖管理 + 构建 + 发布

# 安装:
pip install poetry

# 创建项目:
poetry new myproject
# 或: poetry init (现有项目)

# 添加依赖:
poetry add requests
poetry add --dev pytest

# 安装依赖:
poetry install                 # 全部
poetry install --without dev   # 仅生产

# 激活虚拟环境:
poetry shell

# 运行:
poetry run python main.py

# 构建和发布:
poetry build
poetry publish

// ========== conda ==========
# 科学计算场景,跨语言 (Python/R/C++)
# 安装: 下载 Anaconda 或 Miniconda

conda create -n myenv python=3.10
conda activate myenv
conda install numpy pandas
conda deactivate
```


## 最佳实践与常见问题


```
// ========== 最佳实践 ==========
// 1. 每个项目一个虚拟环境
// 2. .venv 目录在项目根目录
// 3. .venv 加入 .gitignore

# .gitignore
.venv/
env/
venv/
*.pyc
__pycache__/

// 4. 用 requirements.txt 锁定依赖
// 5. 不要手动修改 site-packages 中的文件
// 6. 定期清理不用的虚拟环境

// ========== 常用命令速查 ==========
# 查看所有虚拟环境 (需要安装 virtualenvwrapper)
# lsvirtualenv

# 查找 Python 位置:
which python                   # Linux/macOS
where python                  # Windows

# 查看 pip 位置:
which pip

# 查看 site-packages 路径:
python -c "import site; print(site.getsitepackages())"

# 查看已安装的包:
pip list

// ========== 常见问题 ==========
// Q: 激活后 pip 还是全局的?
// A: 检查 PATH: which pip → 应该指向 .venv/bin/pip

// Q: 虚拟环境可以移动位置吗?
// A: 不推荐! 路径硬编码在脚本中

// Q: 退出后 Python 还是虚拟环境的?
// A: 退出后恢复系统 Python

// Q: 如何复制虚拟环境?
// A: 用 requirements.txt:
//    pip freeze > requirements.txt
//    在另一个环境: pip install -r requirements.txt

// ========== VSCode 配置 ==========
# 在 VSCode 中选择虚拟环境的 Python:
# Ctrl+Shift+P → Python: Select Interpreter
# 选择 .venv/bin/python 或 .venv/Scripts/python.exe
#
# VSCode 会自动检测项目中的 .venv 目录
# 终端会自动激活虚拟环境
```


> **Note:** 💡 虚拟环境要点: (1) 使用 venv 隔离项目依赖,避免版本冲突; (2) 核心命令: python -m venv .venv → activate → pip install; (3) 退出用 deactivate; (4) .venv 加入 .gitignore; (5) 现代项目推荐 poetry 或 pipenv 自动管理虚拟环境。


## 练习


<!-- Converted from: 71_Python虚拟环境venv.html -->
