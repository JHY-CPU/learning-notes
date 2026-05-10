# Python FastAPI后台任务与文件上传


## 📎 FastAPI 后台任务与文件上传


BackgroundTasks 后台任务、文件上传 (UploadFile/File)、文件验证、静态文件挂载、WebSocket 基础、Server-Sent Events。


## BackgroundTasks


```
// ========== 后台任务 ==========
# BackgroundTasks: 在返回响应后执行的任务
# 适合: 发送邮件/处理图片/日志记录/通知推送

from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

# 定义后台任务函数:
def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(f"{message}\n")

def send_email(email: str, body: str):
    # 模拟发送邮件
    import time
    time.sleep(2)  # 耗时操作
    print(f"发送邮件到 {email}: {body}")

@app.post("/send-notification")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    # 立即返回响应,后台执行任务
    background_tasks.add_task(write_log, f"通知发送到 {email}")
    background_tasks.add_task(send_email, email, "你好!")
    return {"message": "通知将在后台发送"}

# 依赖注入中使用:
def get_logger(background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, "请求已记录")
    return "logger"

@app.get("/items")
async def read_items(logger: str = Depends(get_logger)):
    return {"items": [1, 2, 3]}
```


## 文件上传 (UploadFile)


```
// ========== 单文件上传 ==========
from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # UploadFile 属性:
    print(file.filename)       # 原始文件名
    print(file.content_type)   # MIME 类型 (如 image/png)
    print(file.size)           # 文件大小

    # 保存文件:
    with open(f"uploads/{file.filename}", "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size
    }

# ========== 文件验证 ==========
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "application/pdf"}
MAX_SIZE = 5 * 1024 * 1024  # 5 MB

@app.post("/upload-validated")
async def upload_validated(file: UploadFile = File(...)):
    # 验证类型:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"不支持的文件类型: {file.content_type}")

    # 验证大小:
    contents = await file.read()
    if len(contents) > MAX_SIZE:
        raise HTTPException(400, f"文件过大 (> {MAX_SIZE} bytes)")

    # 重置文件指针 (读取后需要)
    await file.seek(0)

    # 保存:
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename": file.filename, "size": len(contents)}
```


## 多文件上传


```
// ========== 多文件上传 ==========
from fastapi import FastAPI, File, UploadFile
from typing import List

app = FastAPI()

@app.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(content),
            "type": file.content_type
        })
    return {"files": results}

# ========== 表单字段 + 文件 ==========
from fastapi import Form

@app.post("/upload-with-fields")
async def create_article(
    title: str = Form(...),
    content: str = Form(...),
    image: UploadFile = File(...)
):
    # 保存图片
    with open(f"uploads/{image.filename}", "wb") as f:
        shutil.copyfileobj(image.file, f)

    return {
        "title": title,
        "content": content,
        "image": image.filename
    }

# ========== FileResponse ==========
from fastapi.responses import FileResponse

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"uploads/{filename}"
    return FileResponse(
        path=file_path,
        filename=filename,              # 下载时的文件名
        media_type="application/octet-stream"  # 强制下载
    )
```


## 静态文件服务


```
// ========== 挂载静态文件 ==========
from fastapi.staticfiles import StaticFiles

# 挂载静态文件目录:
app.mount("/static", StaticFiles(directory="static"), name="static")

# 现在可以访问: /static/style.css, /static/js/app.js

# ========== HTML 响应 ==========
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def index():
    return """


FastAPI

Hello from FastAPI!


    """

# ========== Jinja2 模板 (需安装) ==========
# pip install jinja2
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")

@app.get("/hello/{name}")
async def hello(request: Request, name: str):
    return templates.TemplateResponse(
        "hello.html",
        {"request": request, "name": name}
    )

# templates/hello.html:
# <h1>Hello {{ name }}!</h1>
```


## WebSocket 基础


```
// ========== WebSocket ==========
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收消息:
            data = await websocket.receive_text()
            # 发送消息:
            await websocket.send_text(f"收到: {data}")
    except WebSocketDisconnect:
        print("客户端断开连接")

# ========== WebSocket 管理器 ==========
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"用户: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("用户离开")
```


> **Note:** 💡 后台任务与文件要点: (1) BackgroundTasks.add_task(func, *args) 响应后执行; (2) UploadFile = File(...) 异步文件上传,read/seek 操作; (3) FileResponse 文件下载; (4) StaticFiles 挂载静态目录; (5) WebSocket 实时双向通信。


## 练习


<!-- Converted from: 103_Python FastAPI后台任务与文件上传.html -->
