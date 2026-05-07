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
