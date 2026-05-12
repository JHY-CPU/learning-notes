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

## 七、Kubernetes Executor配置

```toml
# /etc/gitlab-runner/config.toml (Kubernetes Executor)
[[runners]]
  name = "k8s-runner"
  url = "https://gitlab.com/"
  token = "TOKEN"
  executor = "kubernetes"
  [runners.kubernetes]
    namespace = "gitlab-runner"
    image = "alpine:latest"
    cpu_request = "100m"
    cpu_limit = "500m"
    memory_request = "128Mi"
    memory_limit = "512Mi"
    service_cpu_request = "50m"
    service_memory_request = "128Mi"
    poll_interval = 5
    poll_timeout = 3600
```

## 八、Runner自动缩放

```toml
# Docker Machine Executor (已弃用，推荐Kubernetes Executor)
[[runners]]
  name = "docker-machine-runner"
  executor = "docker+machine"
  [runners.machine]
    IdleCount = 0
    IdleTime = 600
    MaxBuilds = 100
    MachineDriver = "amazonec2"
    MachineName = "runner-%s"
    MachineOptions = [
      "amazonec2-region=us-east-1",
      "amazonec2-instance-type=t3.medium",
      "amazonec2-vpc-id=vpc-xxx",
      "amazonec2-security-group=runner-sg"
    ]
```

## 九、Runner缓存配置

```toml
[[runners]]
  [runners.cache]
    Type = "s3"
    Shared = true
    [runners.cache.s3]
      ServerAddress = "s3.amazonaws.com"
      BucketName = "runner-cache"
      BucketLocation = "us-east-1"
      Insecure = false
```

## 十、Runner安全加固

```bash
# 1. 使用非root用户运行
sudo gitlab-runner run --user=gitlab-runner

# 2. 限制Docker权限
# 在config.toml中:
[runners.docker]
  privileged = false
  security_opt = ["no-new-privileges"]
  read_only = true

# 3. 网络隔离
[runners.docker]
  network_mode = "bridge"
  extra_hosts = []

# 4. 使用TLS
[runners.docker]
  tls_cert_path = "/etc/docker/certs.d"

# 5. 定期清理
[runners.docker]
  disable_entrypoint_overwrite = false
  oom_kill_disable = false
  disable_cache = false
```

## 十一、Runner健康监控

```bash
# 检查Runner状态
sudo gitlab-runner verify --delete

# 检查Runner日志
journalctl -u gitlab-runner -f

# Runner API
curl --header "PRIVATE-TOKEN: your-token" \
  "https://gitlab.com/api/v4/runners"
```
