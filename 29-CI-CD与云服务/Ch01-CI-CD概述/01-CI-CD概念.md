# CI-CD概念

## 一、概念说明

CI/CD是持续集成（Continuous Integration）、持续交付（Continuous Delivery）和持续部署（Continuous Deployment）的简称，是现代软件开发的核心实践。

## 二、核心概念

### 持续集成（CI）

```bash
# 开发者频繁地将代码合并到主分支
# 每次合并自动触发构建和测试

# CI流程
1. 开发者提交代码
2. 自动触发构建
3. 运行单元测试
4. 运行集成测试
5. 生成构建报告
6. 通知团队结果
```

### 持续交付（CD）

```bash
# 代码随时可以部署到生产环境
# 部署需要手动触发

# 持续交付 = CI + 自动化部署到预生产环境
```

### 持续部署（CD）

```bash
# 代码自动部署到生产环境
# 完全自动化，无需人工干预

# 持续部署 = CI + 自动化部署到生产环境
```

## 三、CI/CD流程

```
代码提交 → 构建 → 单元测试 → 集成测试 → 打包 → 部署到测试环境
    → 验收测试 → 部署到预生产 → 部署到生产
```

## 四、关键指标

```bash
# DORA指标
# 1. 部署频率 - 每天/每周/每月
# 2. 变更前置时间 - 代码提交到生产的时间
# 3. 变更失败率 - 导致故障的部署比例
# 4. 恢复时间 - 从故障恢复的时间
```

## 五、注意事项

1. **渐进式实施**：从CI开始，逐步实现CD
2. **自动化测试**：没有测试的CI没有意义
3. **快速反馈**：构建时间应控制在10分钟内
4. **版本控制**：所有代码和配置都要版本控制

## 六、CI/CD流水线实战示例

### 完整的GitHub Actions CI/CD流水线

```yaml
name: Complete CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  NODE_VERSION: '20'
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint:
    name: Code Linting
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      - run: npm ci
      - run: npm run lint

  test:
    name: Unit & Integration Tests
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      - run: npm ci
      - run: npm run test:coverage
      - uses: codecov/codecov-action@v3

  build:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    permissions:
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:latest

  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy via SSH
        run: |
          ssh deploy@production-server << 'EOF'
            docker pull ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            docker-compose up -d
          EOF
```

## 七、CI/CD成熟度模型

```
Level 0: 无自动化 - 手动构建和部署
Level 1: 基础CI - 自动构建，手动部署
Level 2: CI + 自动化测试 - 自动构建和测试，手动部署
Level 3: CI/CD - 自动构建、测试、部署到预生产
Level 4: 完全自动化 - 自动部署到生产，带监控和回滚
Level 5: 优化 - 持续优化，A/B测试，特性开关
```

## 八、常见CI/CD反模式

| 反模式 | 问题 | 解决方案 |
|--------|------|----------|
| 构建时间过长 | 开发者等待，降低效率 | 并行化、缓存、增量构建 |
| 测试不稳定 | 误报导致信任下降 | 隔离测试、固定测试数据 |
| 环境不一致 | "在我机器上能运行" | 使用Docker统一环境 |
| 缺少回滚方案 | 故障恢复慢 | 每次部署保留可回滚版本 |
| Secret硬编码 | 安全风险 | 使用Secret管理服务 |
