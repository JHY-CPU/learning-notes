# 流水线Pipeline

## 一、概念说明

CI/CD流水线是自动化执行的一系列阶段（Stage）和任务（Job），从代码提交到生产部署。

## 二、流水线阶段

```yaml
# 典型流水线
stages:
  - checkout     # 拉取代码
  - build        # 构建
  - test         # 测试
  - package      # 打包
  - deploy-dev   # 部署到开发环境
  - deploy-staging  # 部署到预生产
  - deploy-prod  # 部署到生产
```

## 三、GitHub Actions示例

```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm install
      - run: npm test
      - run: npm run build
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - run: echo "Deploy to production"
```

## 四、触发方式

```bash
# 1. 代码提交
on: push

# 2. Pull Request
on: pull_request

# 3. 定时触发
on:
  schedule:
    - cron: '0 2 * * *'

# 4. 手动触发
on: workflow_dispatch

# 5. 标签触发
on:
  push:
    tags: ['v*']
```

## 五、注意事项

1. **阶段依赖**：通过needs定义依赖关系
2. **并行执行**：独立任务并行运行
3. **失败处理**：配置失败通知和重试
4. **缓存加速**：缓存依赖和构建产物

## 六、完整流水线设计模式

### GitLab CI完整流水线

```yaml
stages:
  - lint
  - test
  - build
  - deploy-staging
  - acceptance-test
  - deploy-production

variables:
  NODE_VERSION: "20"

lint:
  stage: lint
  image: node:${NODE_VERSION}-alpine
  cache:
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - node_modules/
  script:
    - npm ci
    - npm run lint
    - npm run type-check

test:unit:
  stage: test
  image: node:${NODE_VERSION}-alpine
  script:
    - npm ci
    - npm run test:unit -- --coverage
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml
    paths:
      - coverage/

test:integration:
  stage: test
  image: node:${NODE_VERSION}-alpine
  services:
    - postgres:15-alpine
    - redis:7-alpine
  variables:
    DATABASE_URL: "postgres://postgres:postgres@postgres:5432/testdb"
    REDIS_URL: "redis://redis:6379"
  script:
    - npm ci
    - npm run test:integration

build:
  stage: build
  image: docker:24.0.5
  services:
    - docker:24.0.5-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy-staging:
  stage: deploy-staging
  image: alpine:latest
  environment:
    name: staging
    url: https://staging.example.com
  before_script:
    - apk add --no-cache openssh-client
  script:
    - ssh deploy@staging-server "cd /app && docker-compose pull && docker-compose up -d"
  only:
    - main

deploy-production:
  stage: deploy-production
  image: alpine:latest
  environment:
    name: production
    url: https://example.com
  when: manual
  script:
    - ssh deploy@prod-server "cd /app && docker-compose pull && docker-compose up -d"
  only:
    - main
```

## 七、流水线优化策略

| 优化方向 | 具体措施 | 预期效果 |
|----------|----------|----------|
| 并行化 | 独立测试并行运行 | 构建时间减少50%+ |
| 缓存 | 缓存依赖和构建产物 | 依赖安装时间减少80% |
| 增量构建 | 只构建变更部分 | 构建时间减少30-70% |
| 条件执行 | 按变更文件决定运行哪些阶段 | 避免不必要的构建 |
| 资源优化 | 选择合适的Runner规格 | 降低成本 |

## 八、流水线通知集成

```yaml
# Slack通知示例
after_script:
  - |
    if [ "$CI_JOB_STATUS" == "failed" ]; then
      curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\" Pipeline失败!\n项目: $CI_PROJECT_NAME\n分支: $CI_COMMIT_REF_NAME\n提交: $CI_COMMIT_SHORT_SHA\n详情: $CI_PIPELINE_URL\"}" \
        $SLACK_WEBHOOK_URL
    fi
```
