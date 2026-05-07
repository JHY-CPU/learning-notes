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
