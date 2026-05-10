# Python requirements.txt与依赖管理


## 📋 Python requirements.txt 与依赖管理


requirements.txt 最佳实践、多环境依赖管理、版本锁定策略、依赖冲突解决、pyproject.toml 现代配置、依赖安全审计。


## requirements.txt 深入


```
// ========== 依赖管理策略 ==========
// 三种策略:

// 策略 1: 松散指定 (灵活性高)
requests>=2.0
# 优点: 容易适配新版本
# 缺点: 不同时间安装可能版本不同

// 策略 2: 精确锁定 (可重复)
requests==2.28.0
certifi==2022.6.15
# 优点: 完全可复现
# 缺点: 更新麻烦

// 策略 3: 范围指定 (推荐)
requests>=2.25,<3.0
# 优点: 安全更新自动获得
# 缺点: 仍需测试

// ========== 多环境管理 ==========
# 目录结构:
# project/
# ├── requirements/
# │   ├── base.txt         # 公共依赖
# │   ├── production.txt   # 生产环境
# │   ├── development.txt  # 开发环境
# │   └── test.txt         # 测试环境

# base.txt:
requests>=2.25
flask>=2.0

# development.txt:
-r base.txt                 # 引入公共依赖
pytest>=7.0
black>=22.0
ipython>=8.0

# production.txt:
-r base.txt
gunicorn>=20.0

# 使用:
pip install -r requirements/development.txt
```


## 版本锁定与解析


```
// ========== pip freeze 的陷阱 ==========
# pip freeze 会列出所有包 (包括间接依赖)

# 问题: 输出太多,难以区分直接和间接依赖
pip freeze
# certifi==2022.6.15
# charset-normalizer==2.1.0
# idna==3.3
# requests==2.28.0
# urllib3==1.26.11

# 解决方案: pip-chill
pip install pip-chill
pip-chill                     # 只列出直接依赖
# requests==2.28.0

# pip-chill --no-version      # 不带版本号
# requests

// ========== pip-compile ==========
# pip-tools 工具: 从宽泛要求生成锁定文件

# 安装:
pip install pip-tools

# 创建 requirements.in (宽松):
requests>=2.0
flask>=2.0

# 编译生成 requirements.txt (精确锁定):
pip-compile requirements.in
# 输出:
# click==8.1.3
# flask==2.2.0
# itsdangerous==2.1.2
# jinja2==3.1.2
# markupsafe==2.1.1
# requests==2.28.0
# urllib3==1.26.11
# werkzeug==2.2.0

# 升级所有依赖:
pip-compile --upgrade requirements.in

# 升级特定包:
pip-compile --upgrade-package requests requirements.in

// ========== pip-sync ==========
# 让环境与 requirements.txt 完全一致
# 会卸载多余的包!

pip-sync requirements.txt
```


## pyproject.toml 现代配置


```
// ========== pyproject.toml ==========
// PEP 518/621 定义的现代 Python 项目配置
// 替代 setup.py + requirements.txt

// 最小示例:
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "0.1.0"
description = "项目描述"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Alice", email = "alice@example.com"}
]

dependencies = [
    "requests>=2.25",
    "flask>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=22.0",
]
test = [
    "pytest>=7.0",
    "coverage>=6.0",
]

// ========== 安装使用 ==========
# 从 pyproject.toml 安装:
pip install .
pip install -e .              # 可编辑模式 (开发)

# 安装可选依赖:
pip install ".[dev]"
pip install ".[dev,test]"

# 生成 requirements.txt:
pip freeze > requirements.txt

// ========== setuptools 配置对比 ==========
# setup.py (旧):
from setuptools import setup
setup(
    name="myproject",
    version="0.1.0",
    install_requires=["requests>=2.25"],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)

# setup.cfg (新):
[metadata]
name = myproject
version = 0.1.0

[options]
install_requires =
    requests>=2.25

[options.extras_require]
dev =
    pytest>=7.0
```


## 依赖审计与安全


```
// ========== 安全检查 ==========
# 检查已知漏洞:

# 1. pip-audit
pip install pip-audit
pip-audit                     # 扫描已安装包的漏洞

# 输出:
# Found 2 known vulnerabilities:
# - requests (2.28.0): ID 1234 (高)

# 2. safety
pip install safety
safety check                  # 扫描依赖安全

# 3. GitHub Dependabot
# GitHub 仓库自动检测依赖漏洞
# 自动创建更新 PR

// ========== 依赖更新策略 ==========
# 定期更新依赖:
pip list --outdated           # 查看可更新
pip install --upgrade requests  # 升级特定包

# 大版本更新后一定要:
# 1. 阅读 changelog / 迁移指南
# 2. 运行测试套件
# 3. 在 staging 环境验证

// ========== 依赖管理最佳实践 ==========
// ✅ 虚拟环境隔离项目
// ✅ 依赖文件版本控制
// ✅ 锁定间接依赖版本
// ✅ 定期安全审计
// ✅ CI/CD 中验证依赖安装
// ❌ 不要用 sudo pip install
// ❌ 不要在全局环境装项目依赖
// ❌ 不要忽略依赖版本冲突

// ========== 依赖管理工具选择 ==========
// 小项目 (< 5 人):
//   venv + pip + requirements.txt
//
// 中型项目:
//   pipenv 或 pip-tools
//
// 大型项目/库:
//   poetry 或 conda
//
// 发布到 PyPI:
//   poetry 或 flit
```


> **Note:** 💡 依赖管理要点: (1) 用 pip freeze 锁定版本确保可复现; (2) pip-tools 的 pip-compile 从宽泛要求生成精确锁定; (3) pyproject.toml 是现代化项目配置标准; (4) pip-audit 定期扫描依赖安全漏洞; (5) 选择工具按项目规模: 小项目 pip > 中项目 pipenv > 大项目 poetry。


## 练习


<!-- Converted from: 72_Python requirements.txt与依赖管理.html -->
