# FastAPI 核心


## FastAPI 核心概念


FastAPIPydantic异步


FastAPI 是基于 Starlette 和 Pydantic 的现代 Python Web 框架，原生支持异步，自动生成 OpenAPI 文档，性能接近 Node.js/Go。


## 路径参数与查询参数


```
基本路由：
  from fastapi import FastAPI
  app = FastAPI(title="我的API", version="1.0.0")

  @app.get("/")
  async def root():
      return {"message": "Hello World"}

  @app.get("/items/{item_id}")
  async def get_item(item_id: int):
      return {"item_id": item_id}

路径参数类型：
  @app.get("/users/{user_id}")         # int
  @app.get("/files/{file_path:path}")  # 路径匹配
  @app.get("/items/{item_id:int}")     # 显式类型

查询参数：
  from typing import Optional

  @app.get("/items")
  async def list_items(
      skip: int = 0,               # 必填，有默认值
      limit: int = 10,
      q: Optional[str] = None,     # 可选参数
      sort: str = "-created_at",
  ):
      return {"skip": skip, "limit": limit, "q": q}

  # /items?skip=0&limit=20&q=python&sort=-created_at

参数验证（Query）：
  from fastapi import Query

  @app.get("/items")
  async def list_items(
      q: str = Query(
          None,
          min_length=3,
          max_length=50,
          pattern="^[a-zA-Z]+$",
          title="搜索关键词",
          description="用于搜索商品名称",
      ),
      page: int = Query(1, ge=1),        # >= 1
      size: int = Query(20, ge=1, le=100),  # 1-100
  ):
      return {"q": q, "page": page, "size": size}

路径参数验证（Path）：
  from fastapi import Path

  @app.get("/items/{item_id}")
  async def get_item(
      item_id: int = Path(..., title="商品ID", ge=1),
  ):
      return {"item_id": item_id}

请求头与 Cookie：
  from fastapi import Header, Cookie

  @app.get("/items")
  async def get_items(
      user_agent: Optional[str] = Header(None),
      token: Optional[str] = Cookie(None),
      x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
  ):
      return {"user_agent": user_agent, "token": token}
```


## Pydantic 数据模型


```
请求体模型：
  from pydantic import BaseModel, Field, EmailStr, validator
  from typing import Optional, List
  from datetime import datetime
  from enum import Enum

  class PostStatus(str, Enum):
      draft = "draft"
      published = "published"

  class PostCreate(BaseModel):
      title: str = Field(..., min_length=5, max_length=200)
      content: str
      status: PostStatus = PostStatus.draft
      tags: List[str] = []

      @validator('title')
      def title_not_empty(cls, v):
          if not v.strip():
              raise ValueError('标题不能为空')
          return v.strip()

  class PostResponse(BaseModel):
      id: int
      title: str
      content: str
      status: PostStatus
      author_id: int
      created_at: datetime
      updated_at: datetime

      class Config:
          from_attributes = True   # 支持 ORM 模型

  @app.post("/posts", response_model=PostResponse, status_code=201)
  async def create_post(post: PostCreate):
      # post 是已验证的 Pydantic 对象
      db_post = Post(**post.dict())
      await db_post.save()
      return db_post

响应模型：
  from pydantic import BaseModel
  from typing import List, Generic, TypeVar

  T = TypeVar('T')

  class PaginatedResponse(BaseModel, Generic[T]):
      total: int
      page: int
      size: int
      items: List[T]

  class PostListResponse(PaginatedResponse[PostResponse]):
      pass

  @app.get("/posts", response_model=PostListResponse)
  async def list_posts(page: int = 1, size: int = 20):
      total = await Post.count()
      items = await Post.offset((page-1)*size).limit(size).all()
      return PostListResponse(total=total, page=page, size=size, items=items)

  # 排除字段
  @app.get("/posts/{id}", response_model=PostResponse, response_model_exclude={"content"})
  async def get_post(id: int):
      return await Post.get(id)

  # 只返回指定字段
  @app.get("/posts/{id}", response_model=PostResponse, response_model_include={"id", "title"})
  async def get_post_brief(id: int):
      return await Post.get(id)

嵌套模型：
  class Address(BaseModel):
      street: str
      city: str
      country: str

  class UserCreate(BaseModel):
      username: str
      email: EmailStr
      addresses: List[Address] = []
      metadata: Optional[dict] = None  # 任意字典
```


## 异步端点与依赖注入


```
异步端点：
  # async def - 异步视图（推荐用于 IO 密集型）
  @app.get("/posts")
  async def get_posts():
      posts = await Post.objects.all()  # 异步数据库查询
      return posts

  # def - 同步视图（FastAPI 会自动在线程池中运行）
  @app.get("/sync-posts")
  def get_posts_sync():
      posts = Post.objects.all()  # 同步操作
      return posts

依赖注入（Dependency Injection）：
  from fastapi import Depends

  # 数据库会话依赖
  async def get_db():
      async with AsyncSessionLocal() as session:
          try:
              yield session
          finally:
              await session.close()

  @app.get("/posts")
  async def get_posts(db: AsyncSession = Depends(get_db)):
      result = await db.execute(select(Post))
      return result.scalars().all()

  # 分页依赖
  class PaginationParams:
      def __init__(self, page: int = 1, size: int = 20):
          self.page = page
          self.size = size
          self.offset = (page - 1) * size

  @app.get("/posts")
  async def get_posts(pagination: PaginationParams = Depends()):
      posts = await Post.offset(pagination.offset).limit(pagination.size).all()
      return posts

  # 认证依赖
  from fastapi import Security
  from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

  security = HTTPBearer()

  async def get_current_user(
      credentials: HTTPAuthorizationCredentials = Security(security)
  ):
      token = credentials.credentials
      payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
      user = await User.get(payload["sub"])
      if not user:
          raise HTTPException(status_code=401, detail="用户不存在")
      return user

  @app.get("/me")
  async def get_me(user: User = Depends(get_current_user)):
      return user

  # 权限依赖
  def require_role(role: str):
      async def role_checker(user: User = Depends(get_current_user)):
          if user.role != role:
              raise HTTPException(status_code=403, detail="权限不足")
          return user
      return role_checker

  @app.delete("/users/{user_id}")
  async def delete_user(
      user_id: int,
      admin: User = Depends(require_role("admin")),
  ):
      await User.filter(id=user_id).delete()

依赖层级：
  # 依赖可以嵌套
  async def common_params(q: Optional[str] = None, page: int = 1):
      return {"q": q, "page": page}

  async def advanced_params(
      commons: dict = Depends(common_params),
      sort: str = "-created_at",
  ):
      commons["sort"] = sort
      return commons

  @app.get("/items")
  async def get_items(params: dict = Depends(advanced_params)):
      return params
```


## 中间件与异常处理


```
中间件：
  from fastapi import FastAPI, Request
  from starlette.middleware.base import BaseHTTPMiddleware
  import time

  app = FastAPI()

  # 方法1：装饰器中间件
  @app.middleware("http")
  async def add_process_time_header(request: Request, call_next):
      start = time.time()
      response = await call_next(request)
      process_time = time.time() - start
      response.headers["X-Process-Time"] = str(process_time)
      return response

  # 方法2：类中间件
  class LoggingMiddleware(BaseHTTPMiddleware):
      async def dispatch(self, request: Request, call_next):
          print(f"Request: {request.method} {request.url}")
          response = await call_next(request)
          print(f"Response: {response.status_code}")
          return response

  app.add_middleware(LoggingMiddleware)

  # CORS 中间件
  from fastapi.middleware.cors import CORSMiddleware

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:3000"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

异常处理：
  from fastapi import HTTPException, Request
  from fastapi.responses import JSONResponse

  # HTTP 异常
  @app.get("/items/{item_id}")
  async def get_item(item_id: int):
      item = await Item.get(item_id)
      if not item:
          raise HTTPException(
              status_code=404,
              detail="商品不存在",
              headers={"X-Error": "Item not found"},
          )
      return item

  # 全局异常处理器
  class AppException(Exception):
      def __init__(self, code: int, message: str):
          self.code = code
          self.message = message

  @app.exception_handler(AppException)
  async def app_exception_handler(request: Request, exc: AppException):
      return JSONResponse(
          status_code=exc.code,
          content={"error": exc.message, "code": exc.code},
      )

  @app.exception_handler(404)
  async def not_found_handler(request: Request, exc):
      return JSONResponse(
          status_code=404,
          content={"error": "资源未找到", "path": str(request.url)},
      )

生命周期事件：
  @app.on_event("startup")
  async def startup():
      await database.connect()
      print("应用启动")

  @app.on_event("shutdown")
  async def shutdown():
      await database.disconnect()
      print("应用关闭")

  # 新的 lifespan 方式（推荐）
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      # 启动
      await database.connect()
      yield
      # 关闭
      await database.disconnect()

  app = FastAPI(lifespan=lifespan)
```


> **Note:** FastAPI 基于 Pydantic 和 Starlette，原生支持异步和自动 OpenAPI 文档。路径/查询/请求体参数通过类型注解自动验证。依赖注入系统支持数据库会话、认证、权限等功能的解耦复用。中间件处理请求/响应生命周期，异常处理通过 HTTPException 和自定义异常处理器实现。


<!-- Converted from: 01_FastAPI核心.html -->
