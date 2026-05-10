# Python FastAPI项目实战


## 🏗️ FastAPI 项目实战


从零搭建完整 FastAPI 项目：项目结构、应用工厂、路由组织、数据库集成、认证、测试、Docker 部署。FastAPI 知识全景回顾。


## 推荐项目结构


```
// ========== FastAPI 项目结构 ==========
# fastapi-blog/
# ├── app/
# │   ├── __init__.py         # 应用工厂
# │   ├── main.py             # 入口 (uvicorn)
# │   ├── config.py           # 配置 (环境变量)
# │   ├── database.py         # 数据库连接
# │   ├── models/
# │   │   ├── __init__.py
# │   │   ├── user.py         # SQLAlchemy 模型
# │   │   └── post.py
# │   ├── schemas/
# │   │   ├── __init__.py
# │   │   ├── user.py         # Pydantic 模式
# │   │   └── post.py
# │   ├── api/
# │   │   ├── __init__.py
# │   │   ├── users.py        # 用户路由
# │   │   └── posts.py        # 文章路由
# │   ├── core/
# │   │   ├── __init__.py
# │   │   ├── security.py     # 认证/密码
# │   │   └── dependencies.py # 共享依赖
# │   └── utils/
# │       └── __init__.py
# ├── tests/
# │   ├── __init__.py
# │   ├── conftest.py
# │   ├── test_users.py
# │   └── test_posts.py
# ├── .env
# ├── .gitignore
# ├── Dockerfile
# ├── docker-compose.yml
# ├── requirements.txt
# └── pyproject.toml
```


## 应用工厂与配置


```
// ========== config.py ==========
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "FastAPI Blog"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://user:pass@localhost:5432/dbname"
    )
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

settings = Settings()

# ========== database.py ==========
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```


## 应用入口与路由


```
// ========== main.py ==========
from fastapi import FastAPI
from app.core.config import settings
from app.api import users, posts
from app.database import engine, Base

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
    )

    # 创建表:
    @app.on_event("startup")
    def on_startup():
        Base.metadata.create_all(bind=engine)

    # 注册路由:
    app.include_router(users.router, prefix=f"{settings.API_V1_PREFIX}/users", tags=["users"])
    app.include_router(posts.router, prefix=f"{settings.API_V1_PREFIX}/posts", tags=["posts"])

    # 健康检查:
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app

app = create_app()

# ========== api/users.py ==========
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter()

@router.get("/", response_model=list[schemas.User])
def list_users(db: Session = Depends(get_db)):
    return crud.get_users(db)

@router.post("/", response_model=schemas.User, status_code=201)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(400, "邮箱已存在")
    return crud.create_user(db, user)

@router.get("/{user_id}", response_model=schemas.User)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(404, "用户不存在")
    return user
```


## 部署与测试


```
// ========== Dockerfile ==========
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

// ========== docker-compose.yml ==========
# version: "3.8"
# services:
#   api:
#     build: .
#     ports:
#       - "8000:8000"
#     env_file: .env
#     depends_on:
#       - db
#     restart: unless-stopped
#
#   db:
#     image: postgres:15
#     environment:
#       POSTGRES_DB: myapp
#       POSTGRES_USER: myapp
#       POSTGRES_PASSWORD: secret
#     volumes:
#       - pgdata:/var/lib/postgresql/data
#
# volumes:
#   pgdata:

// ========== 测试 conftest.py ==========
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.database import Base, get_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def app():
    app = create_app()
    # 覆盖数据库为内存 SQLite
    engine = create_engine("sqlite:///./test.db")
    TestingSessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield app

@pytest.fixture
def client(app):
    return TestClient(app)
```


## FastAPI 知识全景


```
// ========== FastAPI 知识体系 ==========
// 基础 (581-585):
// 581 - 介绍与安装 (uvicorn/路径操作/自动文档)
// 582 - 路径与查询参数 (Path/Query/验证)
// 583 - 请求体与 Pydantic (模型/Field/嵌套)
// 584 - 响应模型 (response_model/status_code)
// 585 - 自动文档 (Tags/OpenAPI/Swagger定制)

// 进阶 (586-592):
// 586 - 依赖注入 (Depends/嵌套/yield)
// 587 - 数据库集成 (SQLAlchemy/Session/CRUD)
// 588 - 认证与安全 (OAuth2/JWT/角色控制)
// 589 - 中间件 (自定义/CORS/事件处理器)
// 590 - 后台任务与文件 (BackgroundTasks/Upload)
// 591 - 测试 (TestClient/依赖覆盖/异步测试)
// 592 - 项目实战 (结构/部署/全景)

// ========== FastAPI vs Flask 选择 ==========
// FastAPI → JSON API / 微服务 / 高性能
// Flask   → 模板页面 / 简单应用 / 成熟生态

// ========== 学习路线 ==========
// 1. 基础路由和参数 (581-583)
// 2. 响应和文档 (584-585)
// 3. 依赖注入模式 (586)
// 4. 数据库集成 (587)
// 5. 认证安全 (588)
// 6. 中间件/文件/测试 (589-591)
// 7. 项目实战 (592)
```


> **Note:** 💡 FastAPI 综合要点: (1) 模块化: APIRouter + 应用工厂 + Pydantic Schema; (2) 依赖注入: Depends 管理认证/数据库/共享逻辑; (3) 自动文档: Swagger/ReDoc 零配置生成; (4) 异步支持: async/await 高性能; (5) 适合: JSON API / 微服务 / 前后端分离。


## 练习


<!-- Converted from: 105_Python FastAPI项目实战.html -->
