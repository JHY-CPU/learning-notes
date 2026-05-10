# Python测试覆盖率与模式


## 📊 Python 测试覆盖率与模式


coverage.py 覆盖率分析、pytest-cov 集成、测试金字塔策略、TDD 流程、数据驱动测试、测试工厂模式。


## 覆盖率 coverage.py


```
// ========== coverage ==========
# pip install coverage pytest-cov

# 方式 1: 命令行
# coverage run -m pytest tests/
# coverage report              # 终端报告
# coverage html                # HTML 报告 (cov_report/)
# coverage xml                 # XML 报告 (CI 集成)

# 方式 2: pytest-cov 插件
# pytest --cov=src             # 测量 src/ 的覆盖率
# pytest --cov=src --cov-report=term  # 终端输出
# pytest --cov=src --cov-report=html  # HTML 输出
# pytest --cov=src --cov-report=xml   # XML 输出
# pytest --cov=src --cov-branch       # 分支覆盖率

# ========== .coveragerc 配置文件 ==========
# [run]
# source = src
# omit = */tests/*, */migrations/*
# branch = True
#
# [report]
# exclude_lines =
#     pragma: no cover
#     def __repr__
#     raise NotImplementedError
#     if __name__ == "__main__":
#     pass

# ========== 覆盖率解读 ==========
# Name           Stmts   Miss  Cover
# ----------------------------------
# src/app.py        50      5    90%
# src/db.py         30      0   100%
# src/utils.py      20     10    50%
# ----------------------------------
# TOTAL            100     15    85%
#
# Stmts: 总语句数
# Miss: 未执行的语句数
# Cover: 覆盖率
# 目标: 通常 80%+ (项目要求不同)
```


## 测试金字塔


```
// ========== 测试金字塔 ==========
#        ⬆️                   E2E 测试 (少)
#      ⬆️ ⬆️              集成测试 (中)
#    ⬆️ ⬆️ ⬆️           单元测试 (多)
#  ⬆️ ⬆️ ⬆️ ⬆️        静态分析 (最多)

# 单元测试 (Unit Test):
# - 测试单个函数/方法
# - 快速,毫秒级运行
# - 使用 mock 隔离外部依赖
# - 占比: 70-80%

# 集成测试 (Integration Test):
# - 测试多个组件协作
# - 测试数据库/API 交互
# - 可能使用 Testcontainers
# - 占比: 15-20%

# E2E 测试 (End-to-End Test):
# - 从用户视角测试完整功能
# - 真实浏览器/客户端
# - 速度慢,维护成本高
# - 占比: 5-10%

# ========== 测试策略建议 ==========
# 1. 优先写单元测试 (快速反馈)
# 2. 关键路径写集成测试
# 3. 核心流程写少量 E2E 测试
# 4. 测试边界条件 (空值/越界/异常)
# 5. 测试覆盖而非测试代码
```


## TDD 流程


```
// ========== TDD (测试驱动开发) ==========
# 红 → 绿 → 重构

# 步骤 1: 写一个失败的测试 (红)
def test_is_even():
    assert is_even(2) is True
    assert is_even(3) is False
    assert is_even(0) is True

# (此时代码不存在,测试失败!)

# 步骤 2: 写最少代码通过测试 (绿)
def is_even(n):
    return n % 2 == 0

# 步骤 3: 重构(改进代码,测试保持通过)

# ========== TDD 示例 ==========
# 需求: 实现斐波那契数列

# 步骤 1: 测试
def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    assert fibonacci(10) == 55

# 步骤 2: 实现
def fibonacci(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# 步骤 3: 重构 (添加缓存优化)
from functools import lru_cache

@lru_cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试仍然通过 ✅ 可以进行下一步
```


## 数据驱动测试


```
// ========== 数据驱动模式 ==========
import pytest
import json

# 方式 1: parametrize (推荐)
@pytest.mark.parametrize("input,expected", [
    (3, "Fizz"),
    (5, "Buzz"),
    (15, "FizzBuzz"),
    (7, "7"),
    (0, "FizzBuzz"),
])
def test_fizzbuzz(input, expected):
    assert fizzbuzz(input) == expected

# 方式 2: 外部数据文件
def load_test_data():
    """从 JSON 文件加载测试数据"""
    with open("tests/data/test_cases.json") as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_test_data())
def test_from_file(case):
    result = process(case["input"])
    assert result == case["expected"]

# 方式 3: fixture 返回多组数据
@pytest.fixture(params=[
    {"input": [3, 1, 2], "expected": [1, 2, 3]},
    {"input": [], "expected": []},
    {"input": [1], "expected": [1]},
    {"input": [3, 2, 1], "expected": [1, 2, 3]},
])
def sort_case(request):
    return request.param

def test_sort(sort_case):
    result = sorted(sort_case["input"])
    assert result == sort_case["expected"]

# ========== 组合参数 ==========
# 笛卡尔积自动生成所有组合
@pytest.mark.parametrize("separator", [",", "|", "\t"])
@pytest.mark.parametrize("values", [
    ["a", "b", "c"],
    ["1", "2", "3"],
])
def test_join(separator, values):
    result = separator.join(values)
    assert isinstance(result, str)
    assert separator in result
```


## 工厂模式与 Fixture


```
// ========== 测试工厂 ==========
import pytest
from datetime import datetime, timedelta

# 工厂: 快速创建测试对象

# 方式 1: fixture 工厂
@pytest.fixture
def make_user():
    """返回一个创建用户的函数"""
    def _make_user(name="Default", age=18, active=True):
        return {
            "name": name,
            "age": age,
            "active": active,
            "created_at": datetime.now(),
        }
    return _make_user

def test_user_defaults(make_user):
    user = make_user()
    assert user["name"] == "Default"
    assert user["active"] is True

def test_user_custom(make_user):
    user = make_user(name="Alice", age=30)
    assert user["name"] == "Alice"
    assert user["age"] == 30

# 方式 2: factory_boy (需要安装)
# pip install factory_boy
# import factory
#
# class UserFactory(factory.Factory):
#     class Meta:
#         model = dict
#     name = "Alice"
#     email = factory.LazyAttribute(lambda o: f"{o.name}@example.com")
#     age = factory.Faker("random_int", min=18, max=60)
#
# user = UserFactory()  # {"name": "Alice", "email": "Alice@example.com", ...}

# ========== Faker 生成测试数据 ==========
# pip install Faker
from faker import Faker

fake = Faker("zh_CN")

@pytest.fixture
def random_user():
    """生成随机用户数据"""
    return {
        "name": fake.name(),
        "email": fake.email(),
        "phone": fake.phone_number(),
        "address": fake.address(),
    }

def test_random_user(random_user):
    assert "@" in random_user["email"]
    assert len(random_user["name"]) > 0
```


## 完整示例: 测试策略实战


```
// ========== 分层测试示例 ==========
# 被测试代码: src/order_service.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Order:
    id: int
    user_id: int
    items: list
    total: float
    status: str
    created_at: datetime

class OrderService:
    def __init__(self, db, payment, notification):
        self.db = db
        self.payment = payment
        self.notification = notification

    def create_order(self, user_id, items):
        total = sum(item["price"] * item["quantity"] for item in items)

        order = Order(
            id=0,
            user_id=user_id,
            items=items,
            total=total,
            status="pending",
            created_at=datetime.now(),
        )

        order_id = self.db.save(order)
        order.id = order_id

        # 支付处理
        payment_result = self.payment.charge(user_id, total)
        if payment_result["success"]:
            order.status = "paid"
            self.db.update(order)
            self.notification.send(user_id, f"订单 #{order_id} 支付成功")
        else:
            order.status = "failed"

        return order

# ========== 单元测试 ==========
from unittest.mock import MagicMock

def test_create_order_success():
    mock_db = MagicMock()
    mock_db.save.return_value = 1

    mock_payment = MagicMock()
    mock_payment.charge.return_value = {"success": True}

    mock_notify = MagicMock()

    service = OrderService(mock_db, mock_payment, mock_notify)
    items = [{"price": 100, "quantity": 2}, {"price": 50, "quantity": 1}]

    order = service.create_order(1, items)

    assert order.total == 250.0
    assert order.status == "paid"
    mock_notify.send.assert_called_once()

def test_create_order_payment_failed():
    mock_db = MagicMock()
    mock_db.save.return_value = 1

    mock_payment = MagicMock()
    mock_payment.charge.return_value = {"success": False}

    mock_notify = MagicMock()

    service = OrderService(mock_db, mock_payment, mock_notify)
    items = [{"price": 100, "quantity": 1}]

    order = service.create_order(1, items)

    assert order.status == "failed"
    mock_notify.send.assert_not_called()
```


> **Note:** 💡 测试最佳实践: 覆盖率 80%+ 目标; 测试金字塔单元>集成>E2E; TDD 红绿重构循环; factory 模式复用测试数据; @parametrize 覆盖多场景。


## 练习


<!-- Converted from: 118_Python测试覆盖率与模式.html -->
