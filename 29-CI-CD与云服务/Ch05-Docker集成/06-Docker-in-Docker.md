# Docker-in-Docker

## 一、概念说明

Docker-in-Docker（DinD）允许在Docker容器内运行Docker，常用于CI/CD中构建Docker镜像。

## 二、GitLab CI DinD

```yaml
build:
  image: docker:24.0.5
  services:
    - docker:24.0.5-dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  script:
    - docker build -t myapp .
    - docker push myapp
```

## 三、GitHub Actions DinD

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: docker:24.0.5
    services:
      docker:
        image: docker:24.0.5-dind
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t myapp .
```

## 四、安全注意事项

```bash
# 1. 使用TLS
DOCKER_TLS_CERTDIR: "/certs"

# 2. 限制权限
# 不要使用privileged: true除非必要

# 3. 隔离网络
# 使用独立的Docker网络

# 4. 清理资源
# 构建后清理镜像和容器
```

## 五、替代方案

```bash
# Kaniko - 无特权构建
# Buildah - 无守护进程构建
# img - 无守护进程构建
# 推荐使用Kaniko替代DinD
```

## 六、注意事项

1. **安全风险**：DinD有安全风险
2. **资源消耗**：DinD消耗额外资源
3. **替代方案**：优先使用Kaniko
4. **版本匹配**：Docker版本与DinD版本匹配
