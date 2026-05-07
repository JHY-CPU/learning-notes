# Runner配置

## 一、注册Runner

```bash
# 注册命令
sudo gitlab-runner register \
  --non-interactive \
  --url "https://gitlab.com/" \
  --registration-token "TOKEN" \
  --executor "docker" \
  --docker-image "alpine:latest" \
  --description "docker-runner"
```

## 二、Runner类型

```bash
# Shared - 共享Runner（所有项目可用）
# Group - 组Runner（组内项目可用）
# Specific - 项目Runner（仅指定项目可用）
```

## 三、Executor类型

```bash
# shell - 直接在主机执行
# docker - 在Docker容器执行
# docker+machine - 动态Docker机器
# kubernetes - 在K8s Pod执行
```

## 四、配置文件

```toml
# /etc/gitlab-runner/config.toml
[[runners]]
  name = "docker-runner"
  url = "https://gitlab.com/"
  token = "TOKEN"
  executor = "docker"
  [runners.docker]
    image = "alpine:latest"
    privileged = false
```

## 五、标签和Runner选择

```yaml
build:
  tags:
    - docker
    - linux
  script: echo "Runs on docker+linux runner"
```

## 六、注意事项

1. **安全**：保护Runner的Token
2. **资源限制**：配置Runner的资源限制
3. **并发数**：设置合理的并发数
4. **自动清理**：配置Docker自动清理
