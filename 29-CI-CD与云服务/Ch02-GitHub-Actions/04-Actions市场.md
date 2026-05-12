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

## 五、精选第三方Actions

```yaml
# 代码质量
- uses: github/super-linter/slim@v5    # 多语言Linter
- uses: sonarsource/sonarqube-scan-action@master  # SonarQube

# 安全扫描
- uses: github/codeql-action/init@v3    # CodeQL安全分析
- uses: aquasecurity/trivy-action@master  # 镜像扫描

# 部署
- uses: aws-actions/configure-aws-credentials@v4  # AWS认证
- uses: azure/login@v1                  # Azure认证
- uses: google-github-actions/auth@v2   # GCP认证

# 通知
- uses: slackapi/slack-github-action@v1  # Slack通知
- uses: dawidd6/action-send-mail@v3     # 邮件通知

# 发布
- uses: softprops/action-gh-release@v1   # GitHub Release
- uses: pypa/gh-action-pypi-publish@release/v1  # PyPI发布

# Docker
- uses: docker/setup-buildx-action@v3   # Buildx设置
- uses: docker/metadata-action@v5       # 元数据提取
- uses: docker/build-push-action@v5     # 构建推送

# 文档
- uses: peaceiris/actions-gh-pages@v3   # GitHub Pages部署
- uses: JamesIves/github-pages-deploy-action@v4  # Pages部署
```

## 六、Action版本管理策略

```yaml
# 使用语义化版本标签（推荐）
- uses: actions/checkout@v4

# 使用主版本标签（自动获取最新补丁）
- uses: actions/checkout@v4

# 使用SHA固定（最安全，供应链攻击防护）
- uses: actions/checkout@8ade135a41bc03ea155e62e844d188df1ea18608  # v4.1.1

# 使用Dependabot自动更新Actions版本
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```
