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
