# Python Flask测试


## 🧪 Flask 测试


pytest + Flask 测试客户端、测试数据库配置、API 端点测试、数据库 fixture 管理、认证测试、覆盖率报告。


## pytest + Flask 测试


```
// ========== 安装 ==========
# pip install pytest pytest-cov

// ========== conftest.py: 共享 fixture ==========
import pytest
from app import create_app, db as _db

@pytest.fixture
def app():
    """创建测试应用"""
    app = create_app("config.TestConfig")

    with app.app_context():
        _db.create_all()          # 创建测试表
        yield app
        _db.drop_all()            # 清理

@pytest.fixture
def client(app):
    """测试客户端"""
    return app.test_client()

@pytest.fixture
def db(app):
    """数据库实例"""
    return _db

@pytest.fixture
def auth_headers(client):
    """生成认证请求头"""
    # 先注册用户
    client.post("/api/register", json={
        "username": "testuser",
        "password": "testpass"
    })
    # 登录获取 token
    resp = client.post("/api/login", json={
        "username": "testuser",
        "password": "testpass"
    })
    token = resp.get_json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```


## 测试配置


```
// ========== config.py ==========
class Config:
    SECRET_KEY = "dev-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///app.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class TestConfig(Config):
    TESTING = True                     # 启用测试模式
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"  # 内存数据库
    WTF_CSRF_ENABLED = False           # 禁用 CSRF (测试表单)
    SERVER_NAME = "localhost"          # 用于 url_for

// ========== conftest.py 补充 ==========
@pytest.fixture(scope="session")
def app():
    """session 级 fixture (所有测试共享)"""
    app = create_app("config.TestConfig")
    with app.app_context():
        _db.create_all()
        yield app
        _db.drop_all()

@pytest.fixture(autouse=True)
def clean_db(app):
    """每个测试后自动清理数据"""
    yield
    for table in reversed(_db.metadata.sorted_tables):
        _db.session.execute(table.delete())
    _db.session.commit()
```


## 测试 API 端点


```
// ========== test_posts.py ==========
from flask import url_for
import json

class TestPosts:
    """文章 API 测试"""

    def test_create_post(self, client, auth_headers):
        """测试创建文章"""
        resp = client.post("/api/posts",
            json={"title": "测试文章", "content": "内容"},
            headers=auth_headers
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["title"] == "测试文章"
        assert "id" in data

    def test_list_posts(self, client, auth_headers):
        """测试文章列表"""
        # 先创建两篇文章
        client.post("/api/posts",
            json={"title": "文章1", "content": "内容1"},
            headers=auth_headers
        )
        client.post("/api/posts",
            json={"title": "文章2", "content": "内容2"},
            headers=auth_headers
        )

        resp = client.get("/api/posts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["items"]) == 2

    def test_get_post_not_found(self, client):
        """测试不存在的文章"""
        resp = client.get("/api/posts/999")
        assert resp.status_code == 404

    def test_create_post_no_auth(self, client):
        """测试未认证创建文章"""
        resp = client.post("/api/posts",
            json={"title": "文章"}
        )
        assert resp.status_code == 401

    def test_update_post(self, client, auth_headers):
        """测试更新文章"""
        # 创建
        create_resp = client.post("/api/posts",
            json={"title": "原文", "content": "原内容"},
            headers=auth_headers
        )
        post_id = create_resp.get_json()["id"]

        # 更新
        resp = client.put(f"/api/posts/{post_id}",
            json={"title": "修改后"},
            headers=auth_headers
        )
        assert resp.status_code == 200
        assert resp.get_json()["title"] == "修改后"
```


## 测试数据库模型


```
// ========== test_models.py ==========
from app import db
from models import User, Post

class TestUserModel:

    def test_create_user(self, app):
        with app.app_context():
            user = User(username="alice", email="alice@test.com")
            user.set_password("secret123")
            db.session.add(user)
            db.session.commit()

            assert user.id is not None
            assert user.check_password("secret123") is True
            assert user.check_password("wrong") is False

    def test_user_unique_username(self, app):
        with app.app_context():
            user1 = User(username="bob", email="bob@test.com")
            db.session.add(user1)
            db.session.commit()

            user2 = User(username="bob", email="bob2@test.com")
            db.session.add(user2)
            with pytest.raises(Exception):  # 违反唯一约束
                db.session.commit()

// ========== 测试表单 ==========
class TestAuthForm:

    def test_login_form_valid(self, client):
        """测试登录表单验证"""
        resp = client.post("/auth/login", data={
            "username": "testuser",
            "password": "testpass"
        })
        assert resp.status_code == 302  # 重定向

    def test_login_form_missing(self, client):
        """测试缺少字段"""
        resp = client.post("/auth/login", data={
            "username": ""
        })
        assert resp.status_code == 200  # 留在登录页
```


## 运行测试与覆盖率


```
// ========== 运行测试 ==========
# 运行所有测试:
pytest

# 详细输出:
pytest -v

# 运行特定文件:
pytest tests/test_posts.py -v

# 运行特定测试类:
pytest tests/test_posts.py::TestPosts -v

# 运行特定测试函数:
pytest tests/test_posts.py::TestPosts::test_create_post -v

# 打印更多信息:
pytest -v --tb=long            # 长回溯信息
pytest -v -s                   # 显示 print 输出

// ========== 覆盖率 ==========
# 运行并测量覆盖率:
pytest --cov=app tests/

# HTML 覆盖率报告:
pytest --cov=app --cov-report=html tests/
# 生成 htmlcov/index.html

# 分支覆盖率:
pytest --cov=app --cov-branch tests/

// ========== pytest 配置 (pyproject.toml) ==========
# [tool.pytest.ini_options]
# testpaths = ["tests"]
# python_files = ["test_*.py"]
# addopts = "-v --cov=app --cov-report=term-missing"

// ========== 测试结构 ==========
# tests/
# ├── conftest.py         # 共享 fixture
# ├── test_models.py      # 模型测试
# ├── test_auth.py        # 认证测试
# ├── test_posts.py       # API 测试
# └── test_forms.py       # 表单测试
```


> **Note:** 💡 Flask 测试要点: (1) conftest.py 共享 app/client fixture; (2) app.test_client() 模拟 HTTP 请求; (3) 内存数据库 sqlite:///:memory: 快速测试; (4) autouse fixture 自动清理测试数据; (5) pytest --cov=app 衡量代码覆盖率。


## 练习


<!-- Converted from: 91_Python Flask测试.html -->
