# Dockerfile最佳实践

## 一、层缓存优化

```dockerfile
# 不好 - 每次修改代码都重新安装依赖
COPY . .
RUN npm install

# 好 - 依赖层缓存
COPY package*.json ./
RUN npm ci
COPY . .
```

## 二、最小化镜像

```dockerfile
# 使用Alpine基础镜像
FROM node:20-alpine

# 清理缓存
RUN apk add --no-cache git

# 合并RUN命令
RUN apk add --no-cache python3 make g++ && \
    npm install && \
    apk del python3 make g++
```

## 三、安全实践

```dockerfile
# 不要以root运行
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001
USER nextjs

# 不要存储敏感信息
# 使用构建参数
ARG NPM_TOKEN
RUN echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > .npmrc && \
    npm ci && \
    rm .npmrc
```

## 四、.dockerignore

```
node_modules
.git
.env
*.md
Dockerfile
docker-compose.yml
```

## 五、注意事项

1. **COPY顺序**：变化少的文件先COPY
2. **不要使用latest**：指定具体版本
3. **健康检查**：添加HEALTHCHECK
4. **文档**：添加LABEL说明镜像信息

## 六、层缓存优化详解

```dockerfile
# 坏的实践 - 每次修改代码都重新安装依赖
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "index.js"]

# 好的实践 - 依赖层缓存
FROM node:20-alpine
WORKDIR /app
# 1. 先复制依赖声明文件
COPY package.json package-lock.json ./
# 2. 安装依赖（这层会被缓存）
RUN npm ci --only=production
# 3. 最后复制应用代码（这层经常变化，但不影响依赖层）
COPY . .
CMD ["node", "index.js"]
```

## 七、RUN命令最佳实践

```dockerfile
# 合并RUN命令减少层数
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# 使用apk (Alpine)
RUN apk add --no-cache \
    curl \
    ca-certificates \
    git

# 清理包管理器缓存
RUN pip install --no-cache-dir -r requirements.txt
RUN npm ci --only=production && npm cache clean --force
RUN go build -ldflags="-s -w" -o app .  # 去除调试信息
```

## 八、安全强化Dockerfile

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS production
# 安全更新
RUN apk upgrade --no-cache
# 创建非root用户
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup
# 复制文件
WORKDIR /app
COPY --from=builder --chown=appuser:appgroup /app/dist ./dist
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --from=builder --chown=appuser:appgroup /app/package.json ./
# 安全配置
USER appuser
EXPOSE 3000
# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD wget -q --spider http://localhost:3000/health || exit 1
# 不要使用npm start，直接使用node
CMD ["node", "dist/main.js"]
```

## 九、.dockerignore完整模板

```gitignore
# 版本控制
.git
.gitignore
.gitmodules

# 依赖（Docker构建时安装）
node_modules
vendor
__pycache__
*.pyc

# IDE配置
.vscode
.idea
*.swp
*.swo

# 环境文件
.env
.env.*
!.env.example

# 构建产物
dist
build
*.o
*.exe

# 文档
README.md
CHANGELOG.md
LICENSE
docs/

# 测试
tests
test
coverage
*.test.js
*.spec.js

# Docker文件（避免递归）
Dockerfile
docker-compose*.yml
.dockerignore

# 临时文件
tmp
.temp
*.log
*.tmp

# 操作系统文件
.DS_Store
Thumbs.db
```

## 十、镜像元数据

```dockerfile
LABEL org.opencontainers.image.title="My Application"
LABEL org.opencontainers.image.description="A production-ready web application"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="team@example.com"
LABEL org.opencontainers.image.source="https://github.com/org/repo"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
```
