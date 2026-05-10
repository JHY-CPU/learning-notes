# Python Web开发综合回顾


## 🌐 Python Web 开发综合回顾


Python Web 生态全景、Flask vs FastAPI 选型、项目最佳实践、知识体系地图、常见陷阱与经验。


## Python Web 生态全景


```
// ========== Python Web 生态 ==========
# ┌─────────────────────────────────────────────────────┐
# │                  Python Web 生态                     │
# ├─────────┬───────────┬──────────┬────────────────────┤
# │  框架   │  ORM      │  异步    │  工具              │
# ├─────────┼───────────┼──────────┼────────────────────┤
# │ Flask   │ SQLAlchemy│ asyncio  │ Celery             │
# │ FastAPI │ Alembic   │ aiohttp  │ APScheduler        │
# │ Django  │ Peewee    │ httpx    │ Flower             │
# │ Tornado │ Tortoise  │ websockets│ pytest            │
# │ Sanic   │ PonyORM   │ aiocache │ pytest-asyncio     │
# │ Starlette│          │ asyncpg │ coverage           │
# └─────────┴───────────┴──────────┴────────────────────┘

# 使用场景:
# Flask:    简单 API / 中小项目 / 模板渲染
# FastAPI:  高性能 API / 自动文档 / 前后端分离
# Django:  大型项目 / 全功能框架 / 管理后台
# Tornado:  WebSocket / 长连接 / 实时通信
```


## Flask vs FastAPI 选型


```
// ========== 对比分析 ==========
# 特性          Flask              FastAPI
# -------------------------------------------
# 性能          中等               高 (ASGI)
# 异步          可选 (3.6+)        原生 async
# 自动文档      需扩展 (Flask-    ✅ 自动 OpenAPI
#               RESTx/Spectree)
# 数据验证      SQLAlchemy/     ✅ Pydantic
#               marshmallow
# 依赖注入      无 (手动)        ✅ Depends
# WebSocket     扩展              ✅ 原生
# 模板          Jinja2            Jinja2 (可选)
# 生态          成熟 (扩展多)     较新
# 学习曲线      低 (同步)         中 (async)
# 社区          大                快速增长

# ========== 选型建议 ==========
# 选 Flask 当:
# - 团队熟悉同步 Python
# - 需要模板渲染 (SSR)
# - 需要 Flask 扩展生态
# - 简单项目快速开发

# 选 FastAPI 当:
# - 构建 REST/GraphQL API
# - 需要高性能
# - 前后端分离项目
# - 需要自动 API 文档
# - 需要 WebSocket
# - 微服务架构

# 选 Django 当:
# - 大型全栈项目
# - 内置管理后台
# - ORM + Migration 一体化
# - 模板 + API 都重要
```


## 项目最佳实践


```
// ========== 项目结构 ==========
# myproject/
# ├── app/
# │   ├── __init__.py          # 应用工厂
# │   ├── main.py              # 入口点
# │   ├── config.py            # 配置 (pydantic-settings)
# │   ├── database.py          # 数据库连接
# │   ├── models/              # SQLAlchemy 模型
# │   ├── schemas/             # Pydantic 模式
# │   ├── api/                 # 路由
# │   ├── services/            # 业务逻辑
# │   ├── core/                # 认证/依赖
# │   └── utils/               # 工具函数
# ├── tests/
# │   ├── conftest.py
# │   ├── test_api/
# │   └── test_services/
# ├── migrations/              # Alembic
# ├── .env.example
# ├── Dockerfile
# ├── docker-compose.yml
# └── pyproject.toml

# ========== 分层架构 ==========
# 路由层 (API)    → 接收请求,参数验证
# 服务层 (Service) → 业务逻辑
# 数据层 (Model)  → 数据库操作
# Schema          → 请求/响应格式

# ========== 开发流程 ==========
# 1. 定义 Schema (Pydantic)
# 2. 编写 Service 层 (业务逻辑)
# 3. 编写 API 路由 (端点)
# 4. 编写测试 (pytest)
# 5. 数据库迁移 (Alembic)
# 6. 部署 (Docker + CI/CD)
```


## 常见陷阱


```
// ========== Python Web 常见陷阱 ==========
# 1. 忽略异步阻塞
@app.get("/users")
async def get_users():
    # ❌ 阻塞事件循环!
    result = some_sync_function()  # 同步 SQLAlchemy
    return result

    # ✅ 非阻塞
    result = await some_async_function()
    return result

# 2. N+1 查询
users = db.query(User).all()
for user in users:  # N+1!
    print(user.posts)

# ✅ 急加载
users = db.query(User).options(joinedload(User.posts)).all()

# 3. 不处理 CancelledError
try:
    await long_task()
except asyncio.TimeoutError:
    pass  # ❌ 任务已被取消,但未处理 CancelledError
    # ✅ 检查 task.cancelled()

# 4. Secret 硬编码
# ❌ SECRET_KEY = "hardcoded"
# ✅ SECRET_KEY = os.getenv("SECRET_KEY")

# 5. 不验证输入
# ❌ user = User(name=request.json["name"])
# ✅ user = UserCreate(**request.json)  # Pydantic 验证

# 6. 不写测试
# ❌ 手动测试
# ✅ pytest + coverage

# 7. 无类型提示
# ❌ def get_user(id):
# ✅ def get_user(id: int) -> User | None:
```


## 知识体系地图


```
// ========== Python Web 知识体系 ==========
// Phase 5 已完成 50 个文件 (571-620)

// ┌──────────────────────────────────────────────┐
// │            Flask (571-580)                   │
// │  基础 → 路由/模板 → SQLAlchemy → REST        │
// │  → 认证 → 文件上传 → 测试 → 部署 → 实战      │
// └──────────────────────────────────────────────┘
//
// ┌──────────────────────────────────────────────┐
// │           FastAPI (581-592)                  │
// │  基础 → 参数 → Pydantic → 响应 → 文档        │
// │  → 依赖注入 → 数据库 → 认证 → 中间件         │
// │  → 后台任务/文件 → 测试 → 项目实战           │
// └──────────────────────────────────────────────┘
//
// ┌──────────────────────────────────────────────┐
// │      异步 & 网络 & 测试 (593-620)            │
// │  asyncio基础 → 进阶 → 上下文管理器/迭代器     │
// │  → aiohttp → Queue/同步原语 → requests      │
// │  → httpx → BeautifulSoup → pytest入门→进阶  │
// │  → unittest.mock → pytest-asyncio           │
// │  → 覆盖率 → SQLAlchemy深入 → Alembic        │
// │  → Celery → APScheduler → WebSocket        │
// │  → 异步DB → 缓存/限流 → 性能分析 → 安全    │
// │  → 日志 → 并发对比 → 类型提示 → 配置管理    │
// │  → 设计模式 → 综合回顾                       │
// └──────────────────────────────────────────────┘
//
// 下一步: SQL & 关系型数据库 (621-660)
```


> **Note:** 💡 Phase 5 完成! 从 Flask/FastAPI Web 框架到 asyncio 异步编程,从网络客户端到测试,覆盖了 Python Web 开发的完整生态。下一步进入关系型数据库。


## 练习


<!-- Converted from: 133_Python Web开发综合回顾.html -->
