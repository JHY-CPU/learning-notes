# Docker 化应用实战


## 🐳 Docker 化应用实战


Node.js/Python/Spring Boot/Go 的 Dockerfile 与 Compose 最佳实践、安全策略、镜像优化、CI/CD 集成。


## Node.js Docker 化


```
// ========== Node.js 生产 Dockerfile ==========
// Dockerfile (多阶段)

// 阶段 1: 安装依赖
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production --ignore-scripts

// 阶段 2: 构建
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

// 阶段 3: 运行
FROM node:20-alpine AS runner
WORKDIR /app

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV NODE_ENV=production
CMD ["node", "server.js"]

// ========== Node.js .dockerignore ==========
// node_modules
// .next
// .git
// .env
// .env.local
// .env.*.local
// *.md
// Dockerfile
// docker-compose*.yml
// .vscode
// .idea

// ========== Node.js 开发 Dockerfile ==========
// Dockerfile.dev
// FROM node:20-alpine
// WORKDIR /app
// COPY package*.json ./
// RUN npm install
// COPY . .
// CMD ["npm", "run", "dev"]

// docker compose 开发:
// services:
//   app:
//     build:
//       context: .
//       dockerfile: Dockerfile.dev
//     volumes:
//       - .:/app          # 热重载
//       - /app/node_modules  # 使用容器内的 node_modules
//     ports:
//       - "3000:3000"
```


## Python Docker 化


```
// ========== Python FastAPI Dockerfile ==========
// Dockerfile

FROM python:3.12-slim AS builder

WORKDIR /app

// 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

// 阶段 2: 运行
FROM python:3.12-slim

RUN adduser --system --uid 1001 appuser

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

// ========== Python Compose ==========
// services:
//   api:
//     build: .
//     ports:
//       - "8000:8000"
//     env_file: .env
//     depends_on:
//       db:
//         condition: service_healthy
//     volumes:
//       - .:/app  # 开发热重载
//
//   db:
//     image: postgres:16-alpine
//     environment:
//       POSTGRES_USER: app
//       POSTGRES_PASSWORD: ${DB_PASSWORD}
//       POSTGRES_DB: app
//     volumes:
//       - pgdata:/var/lib/postgresql/data
//     healthcheck:
//       test: pg_isready -U app

// ========== Python 安全注意事项 ==========
// 1. requirements.txt 固定版本
// 2. --no-cache-dir 减镜像体积
// 3. PYTHONUNBUFFERED=1 日志实时输出
// 4. 非 root 用户运行
// 5. .dockerignore 排除 __pycache__ .git
```


## Java Spring Boot Docker 化


```
// ========== Spring Boot 分层 Dockerfile ==========
// 利用 Spring Boot 分层 jar 优化缓存

FROM eclipse-temurin:21-jre-alpine AS builder
WORKDIR /app
COPY target/*.jar app.jar
RUN java -Djarmode=layertools -jar app.jar extract

FROM eclipse-temurin:21-jre-alpine
RUN adduser -D appuser
WORKDIR /app

// 分层复制 (利用缓存)
COPY --from=builder /app/dependencies/ ./
COPY --from=builder /app/spring-boot-loader/ ./
COPY --from=builder /app/snapshot-dependencies/ ./
COPY --from=builder /app/application/ ./

USER appuser
EXPOSE 8080
ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]

// ========== Maven 构建 Compose ==========
// services:
//   app:
//     build: .
//     ports:
//       - "8080:8080"
//     environment:
//       SPRING_DATASOURCE_URL: jdbc:postgresql://db:5432/app
//       SPRING_REDIS_HOST: redis
//     depends_on:
//       - db
//       - redis
//
//   db:
//     image: postgres:16-alpine
//     environment:
//       POSTGRES_DB: app
//       POSTGRES_PASSWORD: secret

// ========== Maven 编译优化 ==========
// 利用 Docker 层缓存

// FROM maven:3.9-eclipse-temurin-21 AS builder
// WORKDIR /build
// COPY pom.xml .
// RUN mvn dependency:go-offline   // 下载依赖 (缓存)
// COPY src ./src
// RUN mvn package -DskipTests     // 编译
```


## 安全最佳实践


```
// ========== Docker 安全清单 ==========
// 1. 非 root 用户
//    RUN adduser -D appuser
//    USER appuser

// 2. 最小基础镜像
//    alpine (~7MB) 或 distroless (~2MB)
//    distroless 无 shell, 攻击面最小

// 3. 镜像漏洞扫描
//    docker scout quickstart        // Docker Scout
//    trivy image myapp:latest       // Trivy
//    grype myapp:latest             // Grype
//    snyk container test myapp:latest

// 4. 不存储 secrets
//    ❌ ARG DB_PASSWORD
//    ✅ docker run -e DB_PASSWORD=xxx
//    ✅ docker secrets (Swarm)

// 5. 只复制必要文件
//    .dockerignore 排除敏感文件

// 6. 只读根文件系统
//    docker run --read-only --tmpfs /tmp nginx

// 7. 内核能力限制
//    docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE
//    删除所有能力, 只添加需要的

// 8. 安全选项
//    --security-opt=no-new-privileges:true
//    --security-opt seccomp=default.json

// ========== 镜像大小优化 ==========
// Go  (scratch):   ~10MB
// Go  (alpine):    ~15MB
// Node (alpine):   ~120MB
// Python (slim):   ~150MB
// Java (jre):      ~200MB

// 优化技巧:
// 1. 多阶段构建
// 2. alpine/distroless 基础镜像
// 3. --no-cache / --no-install-recommends
// 4. 合并 RUN 命令 (减少层)
// 5. 清理包管理缓存
```


> **Note:** 💡 应用 Docker 化: Node.js (npm ci + 多阶段); Python (pip --no-cache-dir + slim); Spring Boot (分层 jar + jre-alpine); Go (静态编译 + scratch); 安全: 非 root, 最小镜像, 只读文件系统, cap-drop, 不存 secrets; 镜像体积: Go


## 练习


## 练习


<!-- Converted from: 4_Docker 化应用实战.html -->
