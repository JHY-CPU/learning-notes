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

## 六、完整的CI测试环境

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
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    networks:
      - test-net
    volumes:
      - ./coverage:/app/coverage
      - ./test-results:/app/test-results

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
      - /var/lib/postgresql/data
    networks:
      - test-net

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - test-net

  rabbitmq:
    image: rabbitmq:3-management-alpine
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "check_running"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - test-net

networks:
  test-net:
    driver: bridge
```

## 七、CI中的Docker Compose使用

```yaml
# GitHub Actions
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Start services
      run: docker-compose -f docker-compose.ci.yml up -d
    - name: Wait for services
      run: |
        until docker-compose -f docker-compose.ci.yml exec -T db pg_isready -U postgres; do
          sleep 2
        done
    - name: Run tests
      run: docker-compose -f docker-compose.ci.yml run --rm app npm test
    - name: Collect artifacts
      if: always()
      run: |
        docker-compose -f docker-compose.ci.yml logs > docker-compose.log
    - name: Cleanup
      if: always()
      run: docker-compose -f docker-compose.ci.yml down -v --remove-orphans
```

```yaml
# GitLab CI
test:
  image: docker/compose:latest
  services:
    - docker:dind
  variables:
    DOCKER_TLS_CERTDIR: ""
  before_script:
    - docker-compose -f docker-compose.ci.yml up -d
    - sleep 10
  script:
    - docker-compose -f docker-compose.ci.yml run --rm app npm test
  after_script:
    - docker-compose -f docker-compose.ci.yml logs
    - docker-compose -f docker-compose.ci.yml down -v
  artifacts:
    paths:
      - coverage/
      - test-results/
    reports:
      junit: test-results/*.xml
```

## 八、Profile和覆盖

```yaml
# docker-compose.yml - 基础配置
version: '3.8'
services:
  app:
    build: .

# docker-compose.override.yml - 开发覆盖
version: '3.8'
services:
  app:
    volumes:
      - .:/app
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development

# docker-compose.prod.yml - 生产覆盖
version: '3.8'
services:
  app:
    image: myapp:latest
    restart: always
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
```

```bash
# 使用覆盖文件
docker-compose -f docker-compose.yml -f docker-compose.ci.yml up -d
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
