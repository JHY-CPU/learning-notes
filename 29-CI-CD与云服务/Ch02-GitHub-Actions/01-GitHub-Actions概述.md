# GitHub Actions概述

## 一、概念说明

GitHub Actions是GitHub内置的CI/CD平台，通过YAML文件定义工作流，支持事件触发、矩阵构建等功能。

## 二、核心概念

```bash
# Workflow（工作流）
# 一个YAML文件定义一个工作流

# Event（事件）
# 触发工作流的事件：push、pull_request等

# Job（任务）
# 一组Step的集合，可以并行执行

# Step（步骤）
# 单个命令或Action

# Action（动作）
# 可复用的代码单元
```

## 三、基本结构

```yaml
name: CI Pipeline
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
```

## 四、运行环境

```bash
# GitHub托管Runner
# ubuntu-latest - Ubuntu最新版
# windows-latest - Windows最新版
# macos-latest - macOS最新版

# 自托管Runner
# 企业自己的服务器
```

## 五、注意事项

1. **免费额度**：公共仓库无限，私有仓库2000分钟/月
2. **YAML语法**：严格遵循YAML语法
3. **缓存**：利用缓存加速构建
4. **Secrets**：敏感信息使用Secrets存储
