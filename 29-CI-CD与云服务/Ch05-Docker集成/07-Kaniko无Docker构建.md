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
