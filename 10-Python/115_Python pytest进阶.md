# Python pytest进阶


## 🧪 pytest 进阶


Fixture scope (function/class/module/session)、yield fixture 清理、conftest.py 层级覆盖、parametrize 参数化、monkeypatch 模拟、autouse 与标记。


## Fixture Scope


```
// ========== Fixture 作用域 ==========
import pytest

# scope 控制 fixture 的生命周期:
# function (默认): 每个测试函数创建一次
# class:    每个测试类创建一次
# module:   每个模块创建一次
# session:  整个测试会话创建一次

@pytest.fixture(scope="function")
def fresh_data():
    """每个测试独立的数据"""
    return {"count": 0}

@pytest.fixture(scope="class")
def db_connection():
    """类级别: 所有测试方法共享同个连接"""
    print("\n[建立数据库连接]")
    conn = {"connected": True}
    yield conn
    print("[关闭数据库连接]")

@pytest.fixture(scope="module")
def config():
    """模块级别: 模块内所有测试共享"""
    return {"debug": False, "version": "1.0"}

@pytest.fixture(scope="session")
def app():
    """会话级别: 整个测试运行一次"""
    print("\n[启动应用]")
    yield {"name": "MyApp"}
    print("[关闭应用]")

class TestDB:
    def test_insert(self, db_connection, config):
        assert db_connection["connected"]
        assert config["version"] == "1.0"

    def test_query(self, db_connection):
        assert db_connection["connected"]
```


## yield Fixture 与清理


```
// ========== yield fixture ==========
import pytest

# yield fixture: 在 yield 之前设置,之后清理
# 适合: 数据库连接/文件/网络资源

@pytest.fixture
def resource():
    print("\n  设置资源")
    # setup 代码
    data = {"status": "ready"}
    yield data  # 测试使用这个值
    # teardown 代码
    print("  清理资源")

def test_use_resource(resource):
    assert resource["status"] == "ready"

# ========== 带异常的清理 ==========
@pytest.fixture
def database():
    """即使测试失败,清理代码也会执行"""
    db = Database()
    db.connect()
    yield db
    db.close()  # 一定执行

# ========== 安全清理 ==========
@pytest.fixture
def temp_dir():
    """创建临时目录并在测试后删除"""
    import tempfile
    import shutil

    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)  # 测试后清理

def test_file_creation(temp_dir):
    import os
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("hello")
    assert os.path.exists(test_file)
```


## conftest.py 层级覆盖


```
// ========== conftest.py 层级 ==========
# conftest.py 按目录层级组织 fixture
# 内层 conftest 覆盖外层

# tests/conftest.py:
import pytest

@pytest.fixture
def app():
    """全局应用 fixture"""
    return {"name": "App", "env": "test"}

@pytest.fixture
def db():
    """全局数据库 fixture"""
    return {"host": "localhost"}

# tests/api/conftest.py:
import pytest

@pytest.fixture
def app():
    """覆盖全局 fixture (仅 api 目录有效)"""
    return {"name": "API-App", "env": "test", "api_version": "v1"}

# tests/api/test_users.py:
def test_user(app, db):
    assert app["name"] == "API-App"  # 使用内层 conftest
    assert db["host"] == "localhost"  # 从父 conftest 继承

# ========== conftest.py 常用模式 ==========
# tests/conftest.py:
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def app():
    from main import create_app
    return create_app()

@pytest.fixture
def client(app):
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    """认证头 fixture (被多个测试文件共享)"""
    resp = client.post("/auth/login", json={
        "username": "testuser",
        "password": "testpass"
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```


## parametrize 参数化


```
// ========== @pytest.mark.parametrize ==========
import pytest

# 参数化: 同一测试函数运行多组数据
# 避免写重复测试代码

# 基本参数化:
@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, -50, 50),
])
def test_add(a, b, expected):
    assert a + b == expected

# 单个参数:
@pytest.mark.parametrize("text", ["hello", "world", "pytest"])
def test_upper(text):
    assert text.upper() == text.upper()

# 组合参数 (笛卡尔积):
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x, y):
    # 运行 2 × 2 = 4 次
    pass

# 带 ID 的参数 (可读性):
@pytest.mark.parametrize("input,expected", [
    ("1+1", 2),
    ("2*3", 6),
    ("10/2", 5),
], ids=["add", "multiply", "divide"])
def test_eval(input, expected):
    assert eval(input) == expected

# ========== Fixture 参数化 ==========
@pytest.fixture(params=["sqlite", "postgresql"])
def database(request):
    """根据参数创建不同的数据库"""
    if request.param == "sqlite":
        return {"type": "sqlite", "url": "sqlite:///test.db"}
    elif request.param == "postgresql":
        return {"type": "postgresql", "url": "postgresql://localhost/test"}

def test_connect(database):
    assert "url" in database
    print(f"测试: {database['type']}")
```


## monkeypatch 模拟


```
// ========== monkeypatch ==========
import pytest

# monkeypatch: 临时修改对象/环境变量
# 内置 fixture,无需导入

# 1. 修改属性:
class Config:
    DEBUG = True
    DATABASE_URL = "production.db"

def test_config(monkeypatch):
    monkeypatch.setattr(Config, "DEBUG", False)
    monkeypatch.setattr(Config, "DATABASE_URL", "test.db")

    assert Config.DEBUG is False
    assert Config.DATABASE_URL == "test.db"

# 2. 修改函数/方法:
def get_user():
    return {"name": "real_user"}

def test_get_user(monkeypatch):
    def mock_get_user():
        return {"name": "mock_user"}

    monkeypatch.setattr("module_name.get_user", mock_get_user)
    result = get_user()
    assert result["name"] == "mock_user"

# 3. 修改环境变量:
def test_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///test.db")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    import os
    assert os.environ["DATABASE_URL"] == "sqlite:///test.db"

# 4. 修改字典:
def test_dict(monkeypatch):
    config = {"key": "value"}
    monkeypatch.setitem(config, "key", "new_value")
    assert config["key"] == "new_value"

# 5. 修改 import:
def test_import(monkeypatch):
    import socket
    def mock_create_connection(*args, **kwargs):
        raise OSError("网络不可用")

    monkeypatch.setattr(socket, "create_connection", mock_create_connection)
    # 所有 socket 连接都会失败

# 6. 应用: 模拟网络请求
def test_fetch_data(monkeypatch):
    import requests

    class MockResponse:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {"data": "mocked"}

    monkeypatch.setattr(requests, "get", lambda url: MockResponse())
    resp = requests.get("https://api.example.com")
    assert resp.json()["data"] == "mocked"
```


## 标记与 autouse


```
// ========== 内置标记 ==========
import pytest

@pytest.mark.skip(reason="功能未实现")
def test_not_ready():
    assert False  # 跳过,不会执行

@pytest.mark.skipif(
    True,  # 条件
    reason="只在 Windows 上运行"
)
def test_windows_only():
    pass

@pytest.mark.xfail(reason="已知 bug #42")
def test_known_bug():
    assert 1 + 1 == 3  # 预期失败

@pytest.mark.xfail(strict=True)
def test_xfail_strict():
    """strict=True: 如果通过了反而失败"""
    assert 1 == 2

@pytest.mark.timeout(5)  # 需要 pytest-timeout
def test_slow():
    import time
    time.sleep(10)  # 5 秒后超时

# ========== 自定义标记 ==========
@pytest.mark.slow
def test_heavy_computation():
    pass

@pytest.mark.smoke
def test_critical():
    pass

# pytest -m "smoke"          # 只运行冒烟测试
# pytest -m "not slow"       # 跳过慢测试
# pytest -m "smoke or slow"  # 或关系

# ========== autouse ==========
@pytest.fixture(autouse=True)
def auto_log():
    """每个测试自动执行,无需显式依赖"""
    print(f"\n[测试开始]")

@pytest.fixture(autouse=True)
def setup_database():
    """每个测试自动重置数据库"""
    # 清理数据库
    # 插入测试数据
    yield

def test_one():
    pass  # auto_log 和 setup_database 自动生效
```


> **Note:** 💡 pytest 进阶要点: scope 控制生命周期, yield 做清理, conftest.py 按层级共享, parametrize 避免重复代码, monkeypatch 模拟环境, @pytest.mark 分类测试。


## 练习


<!-- Converted from: 115_Python pytest进阶.html -->
