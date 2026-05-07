# Docker 容器化

## 一、Dockerfile 最佳实践

```dockerfile
# 多阶段构建 - Spring Boot 应用
FROM eclipse-temurin:17-jdk-alpine AS builder
WORKDIR /app
COPY pom.xml .
COPY src ./src
RUN apk add --no-cache maven && \
    mvn package -DskipTests

FROM eclipse-temurin:17-jre-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
RUN chown -R app:app /app
USER app

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://localhost:8080/actuator/health || exit 1

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
```

## 二、Docker Compose 编排

```yaml
# docker-compose.yml
version: '3.8'
services:
  order-service:
    build: ./order-service
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=docker
      - NACOS_ADDR=nacos:8848
    depends_on:
      mysql:
        condition: service_healthy
      nacos:
        condition: service_started
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: root
    volumes:
      - mysql-data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping"]
      interval: 10s
      retries: 5

  nacos:
    image: nacos/nacos-server:v2.3.0
    environment:
      MODE: standalone
    ports:
      - "8848:8848"

volumes:
  mysql-data:
```

## 三、镜像优化

```yaml
优化策略:
  多阶段构建:
    - 分离构建和运行环境
    - 最终镜像只包含运行时

  基础镜像选择:
    - Alpine 版本体积小
    - Distroless 更安全

  层缓存:
    - 先复制依赖文件
    - 再复制源代码

  安全加固:
    - 非 root 用户运行
    - 只读文件系统
    - 最小化安装包
```

## 四、注意事项

1. **不要在镜像中存放敏感信息**
2. **每次构建使用固定版本标签**
3. **镜像大小要控制在合理范围**
4. **健康检查是必须的**
5. **定期更新基础镜像修复安全漏洞**
