# Python pytest入门


## 🧪 pytest 入门


pytest 发现规则、assert 断言、测试函数组织、命令行选项、fixture 基础、测试类和模块。


## pytest 基础


```
// ========== 安装 ==========
# pip install pytest

# ========== 测试发现规则 ==========
# pytest 自动发现测试文件:
# 1. 文件名: test_*.py 或 *_test.py
# 2. 函数名: test_* 开头
# 3. 类名: Test* 开头 (且没有 __init__)

# test_math.py:
def test_add():
    assert 1 + 1 == 2

def test_subtract():
    """带文档字符串的测试"""
    assert 3 - 1 == 2

class TestString:
    """测试类: 按功能分组"""

    def test_upper(self):
        assert "hello".upper() == "HELLO"

    def test_split(self):
        s = "a b c"
        assert s.split() == ["a", "b", "c"]

# 运行:
# pytest                          # 自动发现
# pytest test_math.py            # 指定文件
# pytest test_math.py::test_add  # 指定函数
# pytest -v                      # 详细输出
# pytest -k "add or upper"       # 按名称匹配
# pytest -x                      # 第一个失败就停止
# pytest --tb=short              # 简短的回溯
```


## assert 断言


```
// ========== pytest assert ==========
# pytest 使用 Python 内置 assert,自动提供详细上下文

def test_assertions():
    # 基本断言
    assert True
    assert 1 + 1 == 2

    # 容器断言 (自动显示差异)
    assert [1, 2, 3] == [1, 2, 4]  # 显示具体不同元素

    # 字符串断言 (自动显示 diff)
    assert "hello world" == "hello pytest"

    # 字典断言
    assert {"a": 1, "b": 2} == {"a": 1, "b": 3}

    # 布尔断言
    assert "hello" in "hello world"
    assert 3 > 1

    # 近似相等 (浮点数)
    assert 0.1 + 0.2 == pytest.approx(0.3)

    # 异常断言
    with pytest.raises(ZeroDivisionError):
        1 / 0

    # 检查异常信息
    with pytest.raises(ValueError) as exc_info:
        int("invalid")
    assert "invalid" in str(exc_info.value)

    # 无条件失败
    # pytest.fail("这个测试不应该执行到这里")

    # 跳过测试
    # pytest.skip("跳过这个功能")

    # 预期失败
    # @pytest.mark.xfail(reason="已知 bug #123")
    # def test_known_bug():
```


## 命令行选项


```
// ========== pytest 命令行 ==========
# ===== 基本选项 =====
# pytest                 # 运行所有测试
# pytest -v              # 详细模式 (显示每个测试名)
# pytest -q              # 安静模式 (简略输出)
# pytest -s              # 显示 print 输出 (不捕获 stdout)

# ===== 选择测试 =====
# pytest -k "user"       # 名称包含 "user" 的测试
# pytest -k "not slow"   # 排除包含 "slow" 的测试
# pytest -k "create or delete"  # 匹配多个关键词
# pytest test_api.py     # 指定文件
# pytest test_api.py::TestUsers::test_create  # 精确指定

# ===== 失败处理 =====
# pytest -x              # 第一个失败立即停止
# pytest --maxfail=3     # 3 次失败后停止
# pytest --lf            # 只运行上次失败的测试 (last fail)
# pytest --ff            # 先运行上次失败的,再运行其他的 (fail first)

# ===== 输出格式 =====
# pytest -v              # 详细
# pytest --tb=short      # 短回溯
# pytest --tb=long       # 长回溯
# pytest --tb=line       # 一行回溯
# pytest --tb=no         # 不显示回溯

# ===== 并行运行 =====
# pip install pytest-xdist
# pytest -n auto         # 自动使用所有 CPU 核心
# pytest -n 4            # 使用 4 个进程

# ===== 覆盖率 =====
# pip install pytest-cov
# pytest --cov=myapp     # 生成覆盖率报告
# pytest --cov=myapp --cov-report=html  # HTML 报告

# ===== 标记 =====
# pytest -m "slow"       # 运行标记了 @pytest.mark.slow 的测试
# pytest -m "not slow"   # 排除
```


## 测试组织


```
// ========== 文件结构 ==========
# project/
# ├── src/
# │   ├── __init__.py
# │   ├── calculator.py
# │   └── database.py
# └── tests/
#     ├── __init__.py
#     ├── conftest.py        # 共享 fixture
#     ├── test_calculator.py  # 测试 calculator 模块
#     └── test_database.py    # 测试 database 模块

# test_calculator.py:
import pytest
from src.calculator import Calculator

class TestCalculator:
    """计算器测试"""

    def setup_method(self):
        """每个测试方法前执行"""
        self.calc = Calculator()

    def teardown_method(self):
        """每个测试方法后执行"""
        pass

    def test_add(self):
        assert self.calc.add(2, 3) == 5

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="不能除以零"):
            self.calc.divide(1, 0)
```


## fixture 基础


```
// ========== Fixture 入门 ==========
import pytest

# Fixture: 提供测试所需的固定数据/状态

@pytest.fixture
def sample_data():
    """提供测试数据"""
    return {"name": "Alice", "age": 30, "items": [1, 2, 3]}

def test_sample_data(sample_data):
    """使用 fixture (通过参数注入)"""
    assert sample_data["name"] == "Alice"
    assert len(sample_data["items"]) == 3

# ========== 多个 fixture ==========
@pytest.fixture
def db_connection():
    return {"host": "localhost", "port": 5432, "connected": True}

def test_database(sample_data, db_connection):
    assert db_connection["connected"]
    assert sample_data["name"] == "Alice"

# ========== fixture 中使用 fixture ==========
@pytest.fixture
def user(db_connection):
    """fixture 可以依赖其他 fixture"""
    return {
        "username": "alice",
        "db": db_connection["host"],
    }

def test_user(user):
    assert user["username"] == "alice"

# ========== conftest.py 共享 fixture ==========
# 在 tests/conftest.py 中定义:
# @pytest.fixture
# def client():
#     return TestClient(app)
#
# 所有 tests/ 下的测试文件自动可用
```


## 完整示例: 测试 Calculator


```
// ========== Calculator 测试 ==========
# 被测试代码 (calculator.py):
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("不能除以零")
        return a / b

    def power(self, a, b):
        return a ** b

    def average(self, numbers):
        if not numbers:
            return 0
        return sum(numbers) / len(numbers)

# ========== 完整测试 ==========
import pytest

@pytest.fixture
def calc():
    return Calculator()

class TestCalculator:
    """Calculator 完整测试套件"""

    def test_add(self, calc):
        assert calc.add(2, 3) == 5
        assert calc.add(-1, 1) == 0
        assert calc.add(0, 0) == 0

    def test_subtract(self, calc):
        assert calc.subtract(5, 3) == 2
        assert calc.subtract(1, 5) == -4

    def test_multiply(self, calc):
        assert calc.multiply(3, 4) == 12
        assert calc.multiply(-2, 3) == -6
        assert calc.multiply(0, 100) == 0

    def test_divide(self, calc):
        assert calc.divide(10, 2) == 5.0
        assert calc.divide(7, 2) == 3.5

    def test_divide_by_zero(self, calc):
        with pytest.raises(ValueError, match="不能除以零"):
            calc.divide(1, 0)

    def test_power(self, calc):
        assert calc.power(2, 3) == 8
        assert calc.power(5, 0) == 1

    def test_average(self, calc):
        assert calc.average([1, 2, 3, 4, 5]) == 3.0
        assert calc.average([]) == 0

    def test_average_single(self, calc):
        assert calc.average([42]) == 42
```


> **Note:** 💡 pytest 入门要点: assert 自动提供详细上下文; fixture 通过参数注入; conftest.py 共享 fixture; -v/-k/-x 命令行选项; 测试文件使用 test_ 前缀。


## 练习


<!-- Converted from: 114_Python pytest入门.html -->
