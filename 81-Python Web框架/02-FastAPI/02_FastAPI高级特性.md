# FastAPI 高级特性


## FastAPI 高级特性


WebSocket流式响应SQLAlchemy


FastAPI 高级特性包括 WebSocket 实时通信、后台任务、流式响应、OpenAPI 文档定制、SQLAlchemy ORM 集成和测试。


## WebSocket 实时通信


```
基本 WebSocket：
  from fastapi import FastAPI, WebSocket, WebSocketDisconnect
  from typing import List

  app = FastAPI()

  class ConnectionManager:
      def __init__(self):
          self.active_connections: List[WebSocket] = []

      async def connect(self, websocket: WebSocket):
          await websocket.accept()
          self.active_connections.append(websocket)

      def disconnect(self, websocket: WebSocket):
          self.active_connections.remove(websocket)

      async def send_personal(self, message: str, websocket: WebSocket):
          await websocket.send_text(message)

      async def broadcast(self, message: str):
          for connection in self.active_connections:
              await connection.send_text(message)

  manager = ConnectionManager()

  @app.websocket("/ws/{client_id}")
  async def websocket_endpoint(websocket: WebSocket, client_id: int):
      await manager.connect(websocket)
      await manager.broadcast(f"客户端 {client_id} 已连接")
      try:
          while True:
              data = await websocket.receive_text()
              await manager.send_personal(f"你发送: {data}", websocket)
              await manager.broadcast(f"客户端 {client_id}: {data}")
      except WebSocketDisconnect:
          manager.disconnect(websocket)
          await manager.broadcast(f"客户端 {client_id} 已断开")

带认证的 WebSocket：
  from fastapi import Query

  @app.websocket("/ws")
  async def websocket_endpoint(
      websocket: WebSocket,
      token: str = Query(...),
  ):
      # 验证 token
      try:
          payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
          user_id = payload["sub"]
      except jwt.JWTError:
          await websocket.close(code=1008)  # Policy Violation
          return

      await manager.connect(websocket)
      try:
          while True:
              data = await websocket.receive_json()
              await handle_message(user_id, data)
      except WebSocketDisconnect:
          manager.disconnect(websocket)

广播与房间管理：
  class RoomManager:
      def __init__(self):
          self.rooms: dict[str, List[WebSocket]] = {}

      async def join_room(self, room_id: str, websocket: WebSocket):
          if room_id not in self.rooms:
              self.rooms[room_id] = []
          self.rooms[room_id].append(websocket)

      async def leave_room(self, room_id: str, websocket: WebSocket):
          self.rooms[room_id].remove(websocket)

      async def broadcast_to_room(self, room_id: str, message: dict):
          for ws in self.rooms.get(room_id, []):
              await ws.send_json(message)

  room_manager = RoomManager()

  @app.websocket("/ws/room/{room_id}")
  async def room_endpoint(websocket: WebSocket, room_id: str):
      await websocket.accept()
      await room_manager.join_room(room_id, websocket)
      try:
          while True:
              data = await websocket.receive_json()
              await room_manager.broadcast_to_room(room_id, data)
      except WebSocketDisconnect:
          await room_manager.leave_room(room_id, websocket)
```


## 后台任务与流式响应


```
后台任务（Background Tasks）：
  from fastapi import BackgroundTasks

  def send_email(to: str, subject: str, body: str):
      # 模拟发送邮件
      import time
      time.sleep(2)
      print(f"邮件已发送: {to}")

  def write_log(message: str):
      with open("log.txt", "a") as f:
          f.write(f"{message}\n")

  @app.post("/register")
  async def register(
      user: UserCreate,
      background_tasks: BackgroundTasks,
  ):
      db_user = await User.create(**user.dict())
      # 在响应返回后执行
      background_tasks.add_task(send_email, user.email, "欢迎", "注册成功!")
      background_tasks.add_task(write_log, f"用户注册: {user.username}")
      return {"message": "注册成功", "user_id": db_user.id}

流式响应（Streaming Response）：
  from fastapi.responses import StreamingResponse
  import asyncio

  # 生成器流式响应
  async def generate_numbers():
      for i in range(100):
          yield f"data: {i}\n\n"
          await asyncio.sleep(0.1)

  @app.get("/stream")
  async def stream():
      return StreamingResponse(
          generate_numbers(),
          media_type="text/event-stream",
      )

  # 文件流式下载
  async def file_iterator(file_path: str, chunk_size: int = 1024):
      with open(file_path, "rb") as f:
          while chunk := f.read(chunk_size):
              yield chunk

  @app.get("/download/{file_path:path}")
  async def download(file_path: str):
      return StreamingResponse(
          file_iterator(file_path),
          media_type="application/octet-stream",
          headers={"Content-Disposition": f"attachment; filename={file_path}"},
      )

  # AI 流式输出
  async def generate_ai_response(prompt: str):
      # 模拟逐 token 输出
      response = "这是一个AI生成的回复..."
      for char in response:
          yield f"data: {json.dumps({'token': char})}\n\n"
          await asyncio.sleep(0.05)
      yield "data: [DONE]\n\n"

  @app.post("/chat/stream")
  async def chat_stream(request: ChatRequest):
      return StreamingResponse(
          generate_ai_response(request.prompt),
          media_type="text/event-stream",
      )

文件上传：
  from fastapi import File, UploadFile

  @app.post("/upload")
  async def upload_file(file: UploadFile = File(...)):
      contents = await file.read()
      # 保存文件
      with open(f"uploads/{file.filename}", "wb") as f:
          f.write(contents)
      return {"filename": file.filename, "size": len(contents)}

  # 多文件上传
  @app.post("/upload-multiple")
  async def upload_files(files: List[UploadFile] = File(...)):
      return {"filenames": [f.filename for f in files]}

  # 大文件流式上传
  @app.post("/upload-large")
  async def upload_large(file: UploadFile = File(...)):
      with open(f"uploads/{file.filename}", "wb") as f:
          while chunk := await file.read(1024 * 1024):  # 1MB chunks
              f.write(chunk)
      return {"filename": file.filename}
```


## OpenAPI 文档与 SQLAlchemy 集成


```
OpenAPI 文档定制：
  from fastapi import FastAPI

  app = FastAPI(
      title="我的 API",
      description="""
      ## 用户管理 API
      提供用户注册、登录、CRUD 功能。
      ### 认证
      使用 Bearer Token (JWT) 认证。
      """,
      version="1.0.0",
      terms_of_service="https://example.com/terms",
      contact={"name": "API 支持", "email": "support@example.com"},
      license_info={"name": "MIT"},
  )

  # 标签分组
  from fastapi import APIRouter, Tag

  users_router = APIRouter(prefix="/api/users", tags=["用户管理"])
  posts_router = APIRouter(prefix="/api/posts", tags=["文章管理"])

  @users_router.get(
      "/",
      summary="获取用户列表",
      description="分页获取用户列表，支持搜索和过滤",
      response_description="用户列表",
      responses={
          200: {"description": "成功"},
          401: {"description": "未认证"},
      },
  )
  async def list_users():
      """获取所有用户的列表"""
      pass

  # 文档访问
  # Swagger UI:  http://localhost:8000/docs
  # ReDoc:       http://localhost:8000/redoc
  # OpenAPI JSON: http://localhost:8000/openapi.json

SQLAlchemy 2.0 集成：
  from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
  from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column
  from sqlalchemy import String, Integer, Text, ForeignKey

  DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/db"
  engine = create_async_engine(DATABASE_URL, echo=True)
  AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

  Base = declarative_base()

  class User(Base):
      __tablename__ = "users"

      id: Mapped[int] = mapped_column(primary_key=True)
      username: Mapped[str] = mapped_column(String(50), unique=True)
      email: Mapped[str] = mapped_column(String(100), unique=True)
      hashed_password: Mapped[str] = mapped_column(String(128))
      is_active: Mapped[bool] = mapped_column(default=True)

      posts: Mapped[list["Post"]] = relationship(back_populates="author")

  class Post(Base):
      __tablename__ = "posts"

      id: Mapped[int] = mapped_column(primary_key=True)
      title: Mapped[str] = mapped_column(String(200))
      content: Mapped[str] = mapped_column(Text)
      author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

      author: Mapped["User"] = relationship(back_populates="posts")

  # 依赖注入
  async def get_db():
      async with AsyncSessionLocal() as session:
          try:
              yield session
          finally:
              await session.close()

  # CRUD 操作
  @app.post("/users", response_model=UserResponse)
  async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
      db_user = User(**user.dict())
      db.add(db_user)
      await db.commit()
      await db.refresh(db_user)
      return db_user

  @app.get("/users/{user_id}", response_model=UserResponse)
  async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
      result = await db.execute(select(User).where(User.id == user_id))
      user = result.scalar_one_or_none()
      if not user:
          raise HTTPException(404, "用户不存在")
      return user

  @app.get("/users", response_model=List[UserResponse])
  async def list_users(
      skip: int = 0,
      limit: int = 20,
      db: AsyncSession = Depends(get_db),
  ):
      result = await db.execute(select(User).offset(skip).limit(limit))
      return result.scalars().all()

Alembic 数据库迁移：
  # 安装: pip install alembic
  # 初始化: alembic init migrations

  # alembic.ini
  sqlalchemy.url = postgresql+asyncpg://user:pass@localhost/db

  # env.py (支持异步)
  from sqlalchemy.ext.asyncio import async_engine_from_config

  # 命令
  # alembic revision --autogenerate -m "create users table"
  # alembic upgrade head
  # alembic downgrade -1
  # alembic history
```


## FastAPI 测试


```
使用 TestClient 测试：
  from fastapi.testclient import TestClient
  from main import app

  client = TestClient(app)

  # 基本测试
  def test_read_main():
      response = client.get("/")
      assert response.status_code == 200
      assert response.json() == {"message": "Hello World"}

  # POST 测试
  def test_create_user():
      response = client.post("/users", json={
          "username": "testuser",
          "email": "test@example.com",
          "password": "testpass",
      })
      assert response.status_code == 201
      data = response.json()
      assert data["username"] == "testuser"
      assert "id" in data

  # 带认证测试
  def test_get_current_user():
      # 先获取 token
      response = client.post("/token", data={
          "username": "testuser", "password": "testpass"
      })
      token = response.json()["access_token"]

      # 带 token 请求
      response = client.get("/me", headers={
          "Authorization": f"Bearer {token}"
      })
      assert response.status_code == 200

pytest fixtures：
  import pytest
  from httpx import AsyncClient
  from main import app

  @pytest.fixture
  async def client():
      async with AsyncClient(app=app, base_url="http://test") as ac:
          yield ac

  @pytest.fixture
  async def auth_client():
      async with AsyncClient(app=app, base_url="http://test") as ac:
          # 登录获取 token
          response = await ac.post("/token", data={
              "username": "testuser", "password": "testpass"
          })
          token = response.json()["access_token"]
          ac.headers["Authorization"] = f"Bearer {token}"
          yield ac

  # 使用
  @pytest.mark.asyncio
  async def test_create_post(auth_client):
      response = await auth_client.post("/posts", json={
          "title": "测试文章",
          "content": "测试内容",
      })
      assert response.status_code == 201

  # 数据库测试
  @pytest.fixture
  async def db_session():
      async with engine.begin() as conn:
          await conn.run_sync(Base.metadata.create_all)
      async with AsyncSessionLocal() as session:
          yield session
      async with engine.begin() as conn:
          await conn.run_sync(Base.metadata.drop_all)

测试命令：
  # 运行测试
  pytest tests/ -v

  # 覆盖率
  pytest tests/ --cov=app --cov-report=html

  # 只运行标记的测试
  pytest -m "not slow"

  # 异步测试
  pytest tests/ -v --asyncio-mode=auto

项目结构推荐：
  project/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py           # FastAPI 实例
  │   ├── config.py         # 配置
  │   ├── database.py       # 数据库
  │   ├── models/           # SQLAlchemy 模型
  │   ├── schemas/          # Pydantic 模型
  │   ├── routers/          # API 路由
  │   ├── services/         # 业务逻辑
  │   ├── dependencies.py   # 依赖注入
  │   └── utils/            # 工具函数
  ├── tests/
  ├── alembic/
  ├── requirements.txt
  ├── Dockerfile
  └── docker-compose.yml
```


> **Note:** FastAPI 高级特性：WebSocket 实时通信（连接管理/房间/广播）、后台任务（BackgroundTasks）、流式响应（StreamingResponse，适合 AI 输出/大文件下载）。SQLAlchemy 2.0 异步集成通过 Depends(get_db) 注入数据库会话。OpenAPI 文档支持丰富的元数据定制。测试用 TestClient（同步）或 AsyncClient（异步），配合 pytest fixtures 管理测试数据。


<!-- Converted from: 02_FastAPI高级特性.html -->
