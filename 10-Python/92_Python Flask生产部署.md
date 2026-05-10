# Python Flask生产部署


## 🚀 Flask 生产部署


WSGI 服务器 (Gunicorn/uWSGI)、Nginx 反向代理、环境变量管理、Docker 容器化部署、多环境配置、日志与监控、HTTPS 配置。


## WSGI 服务器


```
// ========== Flask 开发服务器 vs 生产 ==========
# Flask 自带的开发服务器:
# - 单进程,单线程
# - 不适合生产 (性能差,不安全)
# - 用于开发调试 (debug=True)

# 生产需要 WSGI 服务器:
# - Gunicorn (推荐,简单)
# - uWSGI (更强大,配置复杂)
# - Waitress (Windows 友好)

// ========== Gunicorn ==========
# pip install gunicorn

# 运行:
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# -w 4: 4 个工作进程
# -b: 绑定地址
# app:app: 模块名:应用变量名

# 生产用配置文件:
gunicorn app:app \
    -w 4 \
    -b 0.0.0.0:8000 \
    --timeout 30 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info \
    --daemon

# 配置文件 gunicorn.conf.py:
# workers = 4
# bind = "0.0.0.0:8000"
# timeout = 30
# accesslog = "logs/access.log"
# errorlog = "logs/error.log"
# daemon = True

// ========== Waitress (Windows) ==========
# pip install waitress
# waitress-serve --port=8000 app:app
```


## Nginx 反向代理


```
// ========== Nginx 配置 ==========
# /etc/nginx/sites-available/myapp

# server {
#     listen 80;
#     server_name example.com;
#
#     location / {
#         proxy_pass http://127.0.0.1:8000;
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto $scheme;
#     }
#
#     location /static/ {
#         alias /var/www/myapp/static/;
#         expires 30d;
#     }
#
#     location /uploads/ {
#         alias /var/www/myapp/uploads/;
#     }
# }

// ========== Flask 配置 (在 Nginx 后) ==========
from flask import Flask

app = Flask(__name__)

# 确保 Flask 知道真实 IP 和协议:
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

@app.route("/")
def index():
    # 现在 request.remote_addr 返回真实客户端 IP
    return "Hello, Production!"
```


## 环境变量与配置


```
// ========== 环境变量管理 ==========
import os

class Config:
    # 从环境变量读取,带默认值:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-key")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///app.db"
    )
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME")

# ========== .env 文件 ==========
# pip install python-dotenv

# .env 文件 (不提交到 Git):
# SECRET_KEY=your-secret-key-here
# DATABASE_URL=postgresql://user:pass@localhost/dbname
# FLASK_DEBUG=0

# app.py:
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件到环境变量

# .env.example (提交到 Git,不含真实密码):
# SECRET_KEY=change-me
# DATABASE_URL=postgresql://user:pass@localhost/dbname
# FLASK_DEBUG=0

# .gitignore:
# .env
# *.pyc
# __pycache__/
# .venv/
# instance/
```


## Docker 部署


```
// ========== Dockerfile ==========
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖:
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码:
COPY . .

# 暴露端口:
EXPOSE 8000

# 运行 (不 root):
RUN useradd -m appuser
USER appuser

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]

// ========== docker-compose.yml ==========
version: "3.8"

services:
  web:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - db
      - redis
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: myapp
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  pgdata:
```


## 生产最佳实践


```
// ========== 生产 Checklist ==========
// [ ] SECRET_KEY 使用高强度随机密钥
// [ ] DEBUG = False
// [ ] 使用环境变量配置
// [ ] 数据库使用 PostgreSQL
// [ ] Gunicorn 多进程
// [ ] Nginx 反向代理 + 静态文件
// [ ] HTTPS (Let's Encrypt)
// [ ] 日志轮转
// [ ] 数据库迁移 (Flask-Migrate)
// [ ] 监控与告警

// ========== 日志配置 ==========
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    handler = RotatingFileHandler(
        "logs/app.log", maxBytes=1024*1024, backupCount=10
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

// ========== 性能优化 ==========
# 静态文件由 Nginx 服务 (不经过 Flask)
# 缓存频繁查询 (Flask-Caching + Redis)
# 数据库连接池 (SQLAlchemy 默认处理)
# 大文件异步处理 (Celery)
# GZip 压缩:
from flask import Flask
from flask_compress import Compress
Compress(app)
```


> **Note:** 💡 部署要点: (1) 生产用 Gunicorn (Linux) 或 Waitress (Windows); (2) Nginx 反向代理 + 静态文件服务; (3) 环境变量 + .env 管理配置; (4) Docker 容器化一键部署; (5) DEBUG=False, SECRET_KEY 强随机,HTTPS。


## 练习


<!-- Converted from: 92_Python Flask生产部署.html -->
