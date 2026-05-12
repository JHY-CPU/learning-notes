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

## 七、DinD vs Docker Socket挂载

```yaml
# 方案1: Docker-in-Docker (独立Docker守护进程)
# 优点: 完全隔离
# 缺点: 需要privileged模式，无层缓存
build-dind:
  services:
    - docker:24.0.5-dind
  script:
    - docker build -t myapp .

# 方案2: Docker Socket挂载 (共享主机Docker)
# 优点: 共享层缓存，无需privileged
# 缺点: 安全风险，Job间不隔离
build-socket:
  services:
    - docker:dind
  variables:
    DOCKER_HOST: tcp://localhost:2375
  script:
    - docker build -t myapp .
```

## 八、安全的DinD配置

```yaml
# 安全加固的DinD配置
build:
  image: docker:24.0.5
  services:
    - name: docker:24.0.5-dind
      alias: docker
      variables:
        DOCKER_TLS_CERTDIR: "/certs"
  variables:
    DOCKER_HOST: tcp://docker:2376
    DOCKER_TLS_CERTDIR: "/certs"
    DOCKER_TLS_VERIFY: "1"
    DOCKER_CERT_PATH: "/certs/client"
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## 九、Buildah替代方案

```yaml
# 使用Buildah构建镜像（无守护进程）
build-buildah:
  image: quay.io/buildah/stable:latest
  variables:
    STORAGE_DRIVER: vfs
    BUILDAH_FORMAT: docker
  script:
    - buildah bud -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - buildah push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  tags:
    - docker

# Buildah in Kubernetes
build-k8s:
  image: quay.io/buildah/stable:latest
  variables:
    STORAGE_DRIVER: overlay
  script:
    - buildah bud --isolation chroot -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - buildah push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
```

## 十、方案选型对比

```
┌──────────────────┬────────────┬────────────┬──────────────┐
│ 方案              │ 安全性     │ 性能       │ 适用场景      │
├──────────────────┼────────────┼────────────┼──────────────┤
│ DinD             │ 中         │ 中         │ 需要完全隔离  │
│ Socket挂载       │ 低         │ 高         │ 受信环境      │
│ Kaniko           │ 高         │ 中         │ Kubernetes   │
│ Buildah          │ 高         │ 高         │ 无守护进程    │
│ 本地Docker       │ 高         │ 最高       │ 自托管Runner  │
└──────────────────┴────────────┴────────────┴──────────────┘
```
