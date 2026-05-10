# Python FastAPI中间件与CORS


## 🔧 FastAPI 中间件与 CORS


自定义中间件（请求处理/响应处理/计时/日志）、ASGI 中间件、CORSMiddleware 配置、TrustedHostMiddleware、GZip 压缩、HTTPS 重定向。


## 自定义中间件


```
// ========== 中间件基础 ==========
# 中间件: 在请求到达路径操作前 / 响应返回前执行
# 每个请求都会经过中间件

from fastapi import FastAPI, Request
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # 请求前
    start_time = time.perf_counter()

    # 处理请求 (调用下一个中间件或路径操作)
    response = await call_next(request)

    # 响应后
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 访问任意路由都会添加 X-Process-Time 头
```


## 中间件常见场景


```
// ========== 请求日志中间件 ==========
from fastapi import Request
import logging

logger = logging.getLogger("api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # 记录请求
    logger.info(f"→ {request.method} {request.url.path}")

    # 处理请求
    response = await call_next(request)

    # 记录响应
    logger.info(f"← {request.method} {request.url.path} → {response.status_code}")
    return response

# ========== 限流中间件 (简单版) ==========
from collections import defaultdict
from datetime import datetime, timedelta

request_counts = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # 基于 IP 的简单限流
    client_ip = request.client.host
    now = datetime.now()

    # 清除 60 秒前的记录
    request_counts[client_ip] = [
        t for t in request_counts[client_ip]
        if now - t < timedelta(seconds=60)
    ]

    # 检查限制
    if len(request_counts[client_ip]) >= 60:  # 每分钟 60 次
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"error": "请求太频繁"}
        )

    request_counts[client_ip].append(now)
    response = await call_next(request)
    return response

# ========== 认证中间件 ==========
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 公开路径不检查
    public_paths = ["/docs", "/redoc", "/openapi.json", "/token", "/register"]
    if request.url.path in public_paths:
        return await call_next(request)

    # 检查 Authorization 头
    auth = request.headers.get("Authorization")
    if not auth:
        return JSONResponse(status_code=401, content={"error": "未认证"})

    response = await call_next(request)
    return response
```


## 内置中间件


```
// ========== CORSMiddleware ==========
# 处理跨域请求 (前后端分离必需)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://myapp.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# allow_origins 可选值:
# ["*"] — 允许所有 (开发)
# ["http://localhost:3000"] — 特定域名
# ["https://*.example.com"] — 通配符 (不支持!)

// ========== TrustedHostMiddleware ==========
# 限制允许的 Host 头 (防 Host 头攻击)
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "example.com",
        "*.example.com",
        "localhost",
        "127.0.0.1",
    ]
)

// ========== GZipMiddleware ==========
# 压缩响应 (减少传输大小)
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
# 只压缩大于 1000 字节的响应

// ========== HTTPSRedirectMiddleware ==========
# 将所有 HTTP 重定向到 HTTPS
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

# app.add_middleware(HTTPSRedirectMiddleware)  # 生产启用

// ========== 中间件顺序 ==========
# 中间件按添加顺序执行 (先进先出)
# 请求: 先添加的中间件先处理
# 响应: 后添加的中间件先处理 (栈)
```


## 事件处理器


```
// ========== 启动/关闭事件 ==========
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    print("应用启动...")
    # 初始化数据库连接
    # 加载配置
    # 连接外部服务

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    print("应用关闭...")
    # 关闭数据库连接
    # 清理资源
    # 断开外部服务

# ========== 启动/关闭的生命周期 ==========
# 1. 应用启动
# 2. startup_event 执行
# 3. 处理请求...
# 4. shutdown_event 执行
# 5. 应用关闭

# ========== Lifespan (FastAPI 2.0+) ==========
# 更推荐的方式 (替代 on_event):
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动:
    print("应用启动")
    db.connect()
    yield
    # 关闭:
    print("应用关闭")
    db.disconnect()

app = FastAPI(lifespan=lifespan)
```


## 异常处理器


```
// ========== 全局异常处理 ==========
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# 自定义异常:
class AppException(Exception):
    def __init__(self, message: str, code: int = 400):
        self.message = message
        self.code = code

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.code,
        content={"error": exc.message, "code": exc.code}
    )

@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id == 0:
        raise AppException("商品不存在", 404)
    return {"item_id": item_id}

# 覆盖默认异常:
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def custom_http_handler(request, exc):
    # 自定义 404 页面
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"message": "资源不存在", "code": "NOT_FOUND"}
        )
    return await http_exception_handler(request, exc)

# 捕获所有未处理异常:
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误"}
    )
```


> **Note:** 💡 中间件要点: (1) @app.middleware("http") 自定义请求/响应处理; (2) call_next(request) 传递请求到下一个中间件/路由; (3) CORSMiddleware 处理跨域,允许特定来源; (4) @app.on_event("startup"/"shutdown") 生命周期钩子; (5) exception_handler 全局异常处理。


## 练习


<!-- Converted from: 102_Python FastAPI中间件与CORS.html -->
