# Python FastAPI测试


## 🧪 FastAPI 测试


TestClient 测试客户端、pytest + FastAPI 测试、覆盖依赖注入、异步测试、数据库测试、HTTP 请求模拟。


## TestClient 基础


```
// ========== TestClient ==========
# pip install httpx  # TestClient 需要 httpx
# pip install pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

# 创建测试客户端:
client = TestClient(app)

# 测试:
def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_read_item():
    response = client.get("/items/42?q=test")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42, "q": "test"}

def test_read_item_no_query():
    response = client.get("/items/1")
    assert response.json() == {"item_id": 1, "q": None}
```


## 覆盖依赖注入


```
// ========== 依赖覆盖 ==========
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

app = FastAPI()

# 正常依赖:
async def get_current_user():
    return {"username": "real_user"}

@app.get("/users/me")
async def read_me(user: dict = Depends(get_current_user)):
    return user

// ========== 测试时覆盖依赖 ==========
# 创建测试用依赖:
async def override_get_current_user():
    return {"username": "test_user"}

# 覆盖:
app.dependency_overrides[get_current_user] = override_get_current_user

client = TestClient(app)

def test_read_me():
    response = client.get("/users/me")
    assert response.json() == {"username": "test_user"}

# ========== 使用 conftest 管理覆盖 ==========
# conftest.py:
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_user
    yield TestClient(app)
    app.dependency_overrides.clear()  # 清理

# test_api.py:
def test_create_item(client):
    resp = client.post("/items", json={"title": "test"})
    assert resp.status_code == 201
```


## 测试 CRUD API


```
// ========== CRUD 测试 ==========
import pytest
from fastapi.testclient import TestClient

class TestItems:
    """商品 API 测试"""

    def test_create_item(self, client, auth_headers):
        resp = client.post("/items/", json={
            "title": "测试商品",
            "price": 29.9
        }, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "测试商品"
        assert data["price"] == 29.9
        assert "id" in data

    def test_list_items(self, client, auth_headers):
        # 创建两个商品
        client.post("/items/", json={"title": "A", "price": 10}, headers=auth_headers)
        client.post("/items/", json={"title": "B", "price": 20}, headers=auth_headers)

        resp = client.get("/items/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 2

    def test_get_nonexistent_item(self, client):
        resp = client.get("/items/99999")
        assert resp.status_code == 404

    def test_delete_item(self, client, auth_headers):
        # 创建 → 删除
        create = client.post("/items/", json={"title": "临时", "price": 1}, headers=auth_headers)
        item_id = create.json()["id"]

        resp = client.delete(f"/items/{item_id}", headers=auth_headers)
        assert resp.status_code == 204

        resp = client.get(f"/items/{item_id}")
        assert resp.status_code == 404
```


## 测试认证与授权


```
// ========== 认证测试 ==========
class TestAuth:

    def test_login_success(self, client):
        resp = client.post("/token", data={
            "username": "testuser",
            "password": "testpass"
        })
        assert resp.status_code == 200
        token = resp.json().get("access_token")
        assert token is not None

    def test_login_failure(self, client):
        resp = client.post("/token", data={
            "username": "wrong",
            "password": "wrong"
        })
        assert resp.status_code == 401

    def test_unauthorized_access(self, client):
        resp = client.get("/users/me")
        assert resp.status_code == 401  # 无 token

    def test_invalid_token(self, client):
        resp = client.get("/users/me", headers={
            "Authorization": "Bearer invalid_token"
        })
        assert resp.status_code == 401

    def test_protected_route_with_token(self, client, auth_headers):
        resp = client.get("/users/me", headers=auth_headers)
        assert resp.status_code == 200

// ========== Fixture 管理 ==========
@pytest.fixture
def auth_headers(client):
    """创建认证请求头 fixture"""
    # 先注册用户
    client.post("/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass"
    })
    # 登录
    resp = client.post("/token", data={
        "username": "testuser",
        "password": "testpass"
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# 管理员 token:
@pytest.fixture
def admin_headers(client):
    resp = client.post("/token", data={
        "username": "admin",
        "password": "adminpass"
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
```


## 异步测试


```
// ========== Async 测试 ==========
# pip install pytest-asyncio

import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_async_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200

# ========== 数据库测试 ==========
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db

# 使用内存数据库:
TEST_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def setup_db():
    """每个测试前创建表,测试后删除"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
```


> **Note:** 💡 FastAPI 测试要点: (1) TestClient(app) 模拟 HTTP 请求; (2) app.dependency_overrides 在测试中替换依赖; (3) conftest.py 共享 client/auth_headers fixture; (4) 内存数据库 (SQLite) 快速测试; (5) httpx.AsyncClient 测试异步端点。


## 练习


<!-- Converted from: 104_Python FastAPI测试.html -->
