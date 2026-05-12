# Kaniko无Docker构建

## 一、概念说明

Kaniko在容器内构建Docker镜像，不需要Docker守护进程，更安全。

## 二、GitLab CI Kaniko

```yaml
build:
  image:
    name: gcr.io/kaniko-project/executor:v1.9.1-debug
    entrypoint: [""]
  script:
    - /kaniko/executor
      --context "${CI_PROJECT_DIR}"
      --dockerfile "${CI_PROJECT_DIR}/Dockerfile"
      --destination "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}"
```

## 三、GitHub Actions Kaniko

```yaml
- name: Build with Kaniko
  run: |
    docker run --rm -v $(pwd):/workspace \
      gcr.io/kaniko-project/executor:latest \
      --dockerfile=/workspace/Dockerfile \
      --destination=myapp:latest
```

## 四、Kaniko配置

```yaml
# 构建参数
--build-arg NODE_ENV=production
--cache=true
--cache-repo=myregistry/kaniko-cache
--snapshot-mode=redo
--compressed-caching=false
```

## 五、注意事项

1. **无需特权**：Kaniko不需要特权模式
2. **上下文**：需要配置正确的构建上下文
3. **缓存**：使用缓存加速构建
4. **认证**：配置镜像仓库认证

## 六、Kaniko完整配置

```yaml
# Kaniko with advanced options
build:
  image:
    name: gcr.io/kaniko-project/executor:v1.9.1-debug
    entrypoint: [""]
  variables:
    DOCKER_CONFIG: /kaniko/.docker
  before_script:
    # 配置仓库认证
    - mkdir -p /kaniko/.docker
    - |
      echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" \
        > /kaniko/.docker/config.json
  script:
    - |
      /kaniko/executor \
        --context "${CI_PROJECT_DIR}" \
        --dockerfile "${CI_PROJECT_DIR}/Dockerfile" \
        --destination "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}" \
        --destination "${CI_REGISTRY_IMAGE}:latest" \
        --cache=true \
        --cache-repo="${CI_REGISTRY_IMAGE}/kaniko-cache" \
        --snapshot-mode=redo \
        --compressed-caching=false \
        --build-arg NODE_ENV=production \
        --label "org.opencontainers.image.revision=${CI_COMMIT_SHA}" \
        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

## 七、Kaniko多架构构建

```yaml
# 使用Kaniko构建多架构镜像
build-multi-arch:
  image:
    name: gcr.io/kaniko-project/executor:v1.9.1-debug
    entrypoint: [""]
  script:
    - |
      /kaniko/executor \
        --context "${CI_PROJECT_DIR}" \
        --dockerfile "${CI_PROJECT_DIR}/Dockerfile" \
        --destination "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}" \
        --custom-platform=linux/amd64,linux/arm64 \
        --cache=true
```

## 八、Kaniko缓存优化

```yaml
# 使用WARMLAYER缓存加速
build:
  image:
    name: gcr.io/kaniko-project/executor:v1.9.1-debug
    entrypoint: [""]
  script:
    - |
      /kaniko/executor \
        --context "${CI_PROJECT_DIR}" \
        --dockerfile "${CI_PROJECT_DIR}/Dockerfile" \
        --destination "${CI_REGISTRY_IMAGE}:${CI_COMMIT_SHA}" \
        --cache=true \
        --cache-repo="${CI_REGISTRY_IMAGE}/cache" \
        --cache-copy-layers=true \
        --snapshot-mode=redo \
        --use-new-run=true
```

## 九、GitHub Actions中使用Kaniko

```yaml
name: Build with Kaniko
on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Kaniko Build
        run: |
          docker run --rm \
            -v $(pwd):/workspace \
            -v $HOME/.docker/config.json:/kaniko/.docker/config.json \
            gcr.io/kaniko-project/executor:latest \
            --dockerfile=/workspace/Dockerfile \
            --context=/workspace \
            --destination=ghcr.io/${{ github.repository }}:${{ github.sha }} \
            --cache=true
```

## 十、Kaniko故障排查

```bash
# 常见问题及解决方案

# 1. 上下文路径错误
# 确保 --context 指向包含Dockerfile的目录

# 2. 认证失败
# 检查 /kaniko/.docker/config.json 格式是否正确

# 3. 缓存未命中
# 使用 --cache-repo 指定缓存仓库
# 检查Dockerfile中COPY/ADD指令的顺序

# 4. 构建慢
# 使用 --snapshot-mode=redo
# 使用 --compressed-caching=false
# 使用 --use-new-run=true
```
