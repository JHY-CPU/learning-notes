# Actions市场

## 一、概念说明

GitHub Marketplace提供了数千个可复用的Action，覆盖各种常见任务。

## 二、常用Actions

```yaml
# 检出代码
- uses: actions/checkout@v4

# 设置Node.js
- uses: actions/setup-node@v4
  with:
    node-version: '20'

# 设置Python
- uses: actions/setup-python@v5
  with:
    python-version: '3.11'

# 设置Java
- uses: actions/setup-java@v4
  with:
    java-version: '17'

# 设置Go
- uses: actions/setup-go@v5
  with:
    go-version: '1.21'

# 缓存依赖
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: node-${{ hashFiles('package-lock.json') }}

# 上传制品
- uses: actions/upload-artifact@v4
  with:
    name: build
    path: dist/

# Docker构建
- uses: docker/build-push-action@v5
  with:
    push: true
    tags: user/app:latest
```

## 三、安全注意事项

```bash
# 1. 使用版本标签（v4）而不是master
# 2. 优先使用官方Actions
# 3. 审查第三方Action代码
# 4. 使用SHA固定版本
- uses: actions/checkout@8ade135a41bc03ea155e62e844d188df1ea18608  # v4
```

## 四、创建自己的Action

```bash
# JavaScript Action
# /action.yml
name: 'My Action'
description: 'Custom action'
inputs:
  name:
    description: 'Name'
    required: true
runs:
  using: 'node20'
  main: 'index.js'
```
