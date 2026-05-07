# Redis安装（Docker）

## 一、概念说明

使用Docker安装Redis是最便捷的方式之一，可以快速部署、环境隔离、易于管理。Docker方式适合开发测试环境，也可用于生产环境（配合数据卷和网络配置）。

## 二、具体用法

### 方式一：docker run快速启动

```bash
# 拉取最新Redis镜像
docker pull redis:7.2-alpine

# 启动Redis容器
docker run -d \
  --name my-redis \
  -p 6379:6379 \
  redis:7.2-alpine

# 连接到Redis
docker exec -it my-redis redis-cli

# 带密码启动
docker run -d \
  --name my-redis \
  -p 6379:6379 \
  redis:7.2-alpine \
  redis-server --requirepass "yourpassword"

# 带持久化启动（挂载数据卷）
docker run -d \
  --name my-redis \
  -p 6379:6379 \
  -v /data/redis:/data \
  redis:7.2-alpine \
  redis-server --appendonly yes
```

### 方式二：docker-compose部署（推荐）

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7.2-alpine
    container_name: redis-server
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
      - ./redis.conf:/etc/redis/redis.conf
    command: redis-server /etc/redis/redis.conf
    restart: always
    networks:
      - redis-net

volumes:
  redis-data:

networks:
  redis-net:
    driver: bridge
```

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f redis

# 停止服务
docker-compose down

# 数据持久化验证
docker exec -it redis-server redis-cli SET test "hello"
docker-compose down && docker-compose up -d
docker exec -it redis-server redis-cli GET test
# 输出: "hello"
```

## 三、注意事项与常见陷阱

1. **数据持久化**：必须挂载数据卷或使用AOF，否则容器删除数据丢失
2. **网络配置**：生产环境不要暴露6379端口到公网
3. **内存限制**：通过`--memory`限制容器内存使用
4. **镜像版本**：alpine版本更小但可能缺少调试工具
5. **配置文件**：可通过挂载自定义redis.conf覆盖默认配置
6. **密码安全**：不要在docker-compose.yml中明文写密码，使用环境变量或secrets
