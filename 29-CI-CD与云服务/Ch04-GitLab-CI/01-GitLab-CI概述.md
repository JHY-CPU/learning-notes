# GitLab CI概述

## 一、概念说明

GitLab CI/CD是GitLab内置的CI/CD平台，通过`.gitlab-ci.yml`文件定义流水线。

## 二、基本结构

```yaml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - npm install
    - npm run build

test:
  stage: test
  script:
    - npm test

deploy:
  stage: deploy
  script:
    - ./deploy.sh
  only:
    - main
```

## 三、核心概念

```bash
# Pipeline（流水线）
# 一次CI/CD执行

# Stage（阶段）
# 流水线的阶段：build/test/deploy

# Job（任务）
# 阶段中的具体任务

# Runner（运行器）
# 执行Job的服务器
```

## 四、GitLab Runner

```bash
# 安装Runner
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash
sudo apt-get install gitlab-runner

# 注册Runner
sudo gitlab-runner register
# 输入GitLab URL
# 输入Registration Token
# 输入Runner描述
# 输入Tag
# 选择Executor（docker/shell等）
```

## 五、注意事项

1. **文件位置**：`.gitlab-ci.yml`必须在仓库根目录
2. **语法正确**：YAML语法必须正确
3. **Runner可用**：确保有可用的Runner
4. **缓存配置**：配置依赖缓存加速构建

## 六、GitLab CI完整工作流程

```
开发者推送代码
    │
    ▼
GitLab检测到.gitlab-ci.yml
    │
    ▼
创建Pipeline
    │
    ├─ Stage: build (Job 1, Job 2...)
    ├─ Stage: test (Job 1, Job 2...)
    ├─ Stage: deploy (Job 1, Job 2...)
    │
    ▼
选择可用Runner执行Job
    │
    ├─ Shell Executor → 直接执行
    ├─ Docker Executor → 容器内执行
    ├─ Kubernetes Executor → Pod内执行
    │
    ▼
收集结果、制品、报告
```

## 七、GitLab CI/CD架构

```
┌──────────────────────────────────────────────────────┐
│                   GitLab Server                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │   Git     │  │ CI/CD    │  │ Container Registry │ │
│  │  仓库     │  │ 调度器    │  │ (镜像仓库)         │ │
│  └────┬─────┘  └────┬─────┘  └────────────────────┘ │
│       │              │                                │
└───────┼──────────────┼────────────────────────────────┘
        │              │
   ┌────▼──────────────▼────┐
   │      Runner Pool        │
   │  ┌───────┐  ┌────────┐ │
   │  │Runner1│  │Runner2 │ │
   │  │(Linux)│  │(Docker)│ │
   │  └───────┘  └────────┘ │
   └────────────────────────┘
```

## 八、GitLab CI与GitHub Actions对比

| 特性 | GitLab CI | GitHub Actions |
|------|-----------|----------------|
| 配置文件 | .gitlab-ci.yml | .github/workflows/*.yml |
| Runner | GitLab Runner | GitHub Runner |
| 内置注册表 | GitLab Registry | GHCR |
| 安全扫描 | 内置SAST/DAST | 需第三方Action |
| 环境管理 | Environments | Environments |
| 制品管理 | Artifacts + Packages | Artifacts |
| 免费额度 | 400分钟/月 | 2000分钟/月 |
