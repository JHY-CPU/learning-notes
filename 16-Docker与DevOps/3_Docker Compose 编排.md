# Docker Compose 编排


## 🐙 Docker Compose 编排


docker-compose.yml 详解、services/volumes/networks 配置、多容器应用 (app+db+redis)、开发/生产环境分离、常用命令。


## Compose 文件结构


```
// ========== docker-compose.yml 完整示例 ==========

services:
  # ===== API 服务 =====
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
      args:
        VERSION: 1.0.0
    image: myapp/api:latest
    container_name: myapp-api
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - GIN_MODE=release
    env_file:
      - ./api/.env
    volumes:
      - uploads:/app/uploads
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    restart: unless-stopped
    networks:
      - backend
      - frontend

  # ===== PostgreSQL 数据库 =====
  db:
    image: postgres:16-alpine
    container_name: myapp-db
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secret}
      POSTGRES_DB: myapp
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d  # 初始化脚本
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app"]
      interval: 10s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"  # 开发时暴露
    networks:
      - backend

  # ===== Redis 缓存 =====
  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    ports:
      - "6379:6379"
    networks:
      - backend

  # ===== Nginx 反向代理 =====
  nginx:
    image: nginx:alpine
    container_name: myapp-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - static-files:/usr/share/nginx/html:ro
    depends_on:
      - api
    networks:
      - frontend

volumes:
  postgres-data:
  redis-data:
  uploads:
  static-files:

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
```


## Compose 指令详解


```
// ========== 核心指令 ==========
// services:    服务定义
//   build:     构建配置 (context, dockerfile, args)
//   image:     使用的镜像
//   ports:     端口映射
//   volumes:   卷挂载
//   environment: 环境变量
//   env_file:   环境变量文件
//   depends_on: 依赖关系
//   command:   覆盖默认命令
//   restart:   重启策略
//   healthcheck: 健康检查
//   networks:  网络连接
//   container_name: 容器名

// volumes:     卷定义
// networks:    网络定义
// configs:     配置 (Swarm)
// secrets:     密钥 (Swarm)

// ========== restart 策略 ==========
// no:           不自动重启 (默认)
// always:       总是重启
// on-failure:   失败时重启
// unless-stopped: 除非手动停止

// ========== depends_on ==========
// 简单依赖:
//   depends_on:
//     - db
//     - redis

// 条件依赖:
//   depends_on:
//     db:
//       condition: service_healthy
//     redis:
//       condition: service_started

// ========== 环境变量 ==========
// 方式 1: 直接指定
//   environment:
//     - DB_HOST=db
//     - DB_PORT=5432

// 方式 2: 字典格式
//   environment:
//     DB_HOST: db
//     DB_PORT: "5432"

// 方式 3: 从宿主机传递
//   environment:
//     - DB_PASSWORD=${DB_PASSWORD}

// 方式 4: 从 .env 文件
//   env_file: .env

// ========== 变量替换 ==========
// .env 文件:
// DB_PASSWORD=secret
// IMAGE_TAG=latest

// compose 中使用:
//   image: myapp:${IMAGE_TAG:-latest}
//   environment:
//     DB_PASSWORD: ${DB_PASSWORD}

// 默认值: ${VAR:-default}
```


## 多环境配置


```
// ========== 环境覆盖 ==========
// 默认: docker-compose.yml
// 覆盖: docker-compose.override.yml (自动加载)

// 开发环境 docker-compose.override.yml:
// services:
//   api:
//     build:
//       context: .
//       dockerfile: Dockerfile.dev
//     volumes:
//       - .:/app          # 热重载
//     ports:
//       - "2345:2345"     # dlv 调试
//     environment:
//       - GIN_MODE=debug

// 生产环境 docker-compose.prod.yml:
// services:
//   api:
//     image: registry.example.com/myapp:${TAG:-latest}
//     deploy:
//       replicas: 3
//       resources:
//         limits:
//           memory: 512M
//     restart: always

// 使用:
// docker compose up -d                              # 开发 (自动加载 override)
// docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d  # 生产

// ========== 扩展服务 ==========
// 使用 & 和 * 复用配置

// x-logging: &logging
//   logging:
//     driver: "json-file"
//     options:
//       max-size: "10m"
//       max-file: "3"
//
// services:
//   api:
//     <<: *logging
//   worker:
//     <<: *logging

// ========== Compose 命令 ==========
// docker compose up -d               # 后台启动
// docker compose up -d --build       # 重新构建
// docker compose down                # 停止并删除
// docker compose down -v             # 停止并删除卷
// docker compose ps                  # 查看状态
// docker compose logs -f             # 日志
// docker compose logs -f api         # 特定服务
// docker compose exec api bash       # 进入容器
// docker compose restart api         # 重启服务
// docker compose pull                # 拉取镜像
// docker compose build               # 构建镜像
// docker compose config              # 验证配置
// docker compose top                 # 进程
```


## 实战示例


```
// ========== Go API + PostgreSQL + Redis ==========
services:
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      DATABASE_DSN: host=db user=postgres password=postgres dbname=todo sslmode=disable
      REDIS_ADDR: redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: pg_isready -U postgres

  redis:
    image: redis:7-alpine
    healthcheck:
      test: redis-cli ping

volumes:
  pgdata:

// ========== Python FastAPI + PostgreSQL ==========
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql+asyncpg://app:secret@db:5432/app
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: app
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:

// ========== Node.js + MongoDB + Redis ==========
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      MONGODB_URI: mongodb://mongo:27017/app
      REDIS_URL: redis://redis:6379
    depends_on:
      - mongo
      - redis

  mongo:
    image: mongo:7
    volumes:
      - mongodata:/data/db

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

volumes:
  mongodata:
  redisdata:

// ========== .env 文件建议 ==========
// 生产环境使用 .env 文件:
// DB_PASSWORD=
// JWT_SECRET=
// TAG=v1.0.0

// .env 不上传 git
// .env.example 上传 git (不含敏感信息)
```


> **Note:** 💡 Compose 要点: docker-compose.yml 定义多服务; build/image/ports/volumes/environment/depends_on; depends_on condition: service_healthy; 环境变量 4 种方式; .env 文件管理敏感信息; 多环境覆盖 (override/prod); x-logging 复用配置; docker compose up/down/logs/exec; 端口映射注意生产不暴露 DB; Volumes 持久化数据。


## 练习


<!-- Converted from: 3_Docker Compose 编排.html -->
