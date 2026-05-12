# Docker在CI-CD中的角色

## 一、概念说明

Docker为CI/CD提供一致的构建和运行环境，消除"在我机器上能运行"的问题。

## 二、Docker在CI/CD中的应用

```bash
# 1. 构建环境
# 使用Docker提供一致的构建环境

# 2. 测试环境
# 使用Docker运行测试环境

# 3. 制品管理
# Docker镜像作为部署制品

# 4. 运行环境
# 生产环境使用Docker运行应用
```

## 三、CI/CD流程

```
代码提交 → 构建Docker镜像 → 推送镜像仓库 → 部署到测试环境
    → 运行测试 → 部署到生产环境
```

## 四、基本Dockerfile

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

## 五、CI中的Docker

```yaml
# GitHub Actions
- uses: docker/build-push-action@v5
  with:
    push: true
    tags: myapp:${{ github.sha }}
```

## 六、注意事项

1. **镜像大小**：使用多阶段构建减小镜像
2. **安全扫描**：构建后扫描镜像漏洞
3. **缓存利用**：利用Docker层缓存
4. **版本标签**：使用Git SHA作为镜像标签

## 七、Docker CI/CD完整架构

```
开发流程:
  代码推送 → CI触发 → 多阶段构建 → 安全扫描 → 推送仓库 → 部署

┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  开发者      │     │  CI/CD平台    │     │  镜像仓库     │
│  git push    │────▶│  Build       │────▶│  Push        │
│              │     │  Test        │     │  Registry    │
│              │     │  Scan        │     │              │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │  部署平台      │
                                          │  K8s/ECS/Swarm│
                                          └──────────────┘
```

## 八、完整的Dockerfile CI/CD模板

```dockerfile
# Dockerfile - 生产就绪的多阶段构建
# === 构建阶段 ===
FROM node:20-alpine AS builder
WORKDIR /app

# 安装依赖（利用层缓存）
COPY package*.json ./
RUN npm ci --only=production && \
    cp -R node_modules /prod_modules && \
    npm ci

# 构建应用
COPY . .
RUN npm run build

# === 生产阶段 ===
FROM node:20-alpine AS production
WORKDIR /app

# 安全配置
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# 只复制必要文件
COPY --from=builder --chown=nextjs:nodejs /prod_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

USER nextjs
EXPOSE 3000
CMD ["node", "dist/main.js"]

# === 开发阶段 ===
FROM node:20-alpine AS development
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
CMD ["npm", "run", "dev"]
```

## 九、Docker Compose CI/CD测试环境

```yaml
# docker-compose.ci.yml
version: '3.8'
services:
  app:
    build:
      context: .
      target: test
    environment:
      - NODE_ENV=test
      - DATABASE_URL=postgres://postgres:postgres@db:5432/testdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./coverage:/app/coverage

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: testdb
      POSTGRES_PASSWORD: postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    tmpfs:
      - /var/lib/postgresql/data  # 内存中运行，测试完即删

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
```
