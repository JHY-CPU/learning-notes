# Docker Compose流水线

## 一、概念说明

Docker Compose用于定义和运行多容器Docker应用，在CI/CD中用于集成测试环境。

## 二、基本配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgres://db:5432/myapp
    depends_on:
      - db
  
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_PASSWORD=secret
  
  redis:
    image: redis:7-alpine
```

## 三、CI集成

```yaml
# GitHub Actions
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker-compose up -d
      - run: sleep 10  # 等待服务启动
      - run: npm run test:integration
      - run: docker-compose down
```

## 四、测试专用配置

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  app:
    build:
      context: .
      target: test
    command: npm test
    environment:
      - NODE_ENV=test
```

## 五、注意事项

1. **服务依赖**：使用depends_on和健康检查
2. **数据清理**：测试后清理数据
3. **网络隔离**：使用独立网络
4. **资源限制**：设置资源限制
