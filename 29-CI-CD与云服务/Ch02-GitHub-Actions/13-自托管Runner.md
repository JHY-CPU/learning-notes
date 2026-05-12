# 自托管Runner

## 一、概念说明

自托管Runner是在自己服务器上运行的GitHub Actions运行器，用于私有网络或特殊环境。

## 二、安装Runner

```bash
# 下载Runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# 配置
./config.sh --url https://github.com/your-org --token YOUR_TOKEN

# 运行
./run.sh

# 或作为服务安装
sudo ./svc.sh install
sudo ./svc.sh start
```

## 三、使用自托管Runner

```yaml
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

## 四、Runner标签

```yaml
jobs:
  build:
    runs-on: [self-hosted, linux, x64]
```

## 五、注意事项

1. **安全风险**：自托管Runner可能有安全风险
2. **自动更新**：开启Runner自动更新
3. **网络访问**：确保Runner能访问GitHub
4. **资源管理**：监控Runner资源使用

## 六、Docker方式运行Runner

```bash
# 使用docker-compose管理Runner
# docker-compose.yml
version: '3.8'
services:
  runner:
    image: myoung34/github-runner:latest
    environment:
      - RUNNER_NAME_PREFIX=self-hosted
      - ACCESS_TOKEN=${GITHUB_TOKEN}
      - RUNNER_WORKDIR=/tmp/runner/work
      - RUNNER_GROUP=default
      - ORG_RUNNER=true
      - ORG_NAME=my-org
      - LABELS=linux,docker,gpu
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /tmp/runner:/tmp/runner
    deploy:
      replicas: 3
```

## 七、Kubernetes Runner（ARC）

```yaml
# Actions Runner Controller (ARC)
# Helm安装
# helm repo add actions-runner-controller https://actions-runner-controller.github.io/actions-runner-controller
# helm install arc actions-runner-controller/actions-runner-controller

apiVersion: actions.summerwind.dev/v1alpha1
kind: RunnerDeployment
metadata:
  name: runner-deployment
spec:
  replicas: 3
  template:
    spec:
      repository: my-org/my-repo
      labels:
        - k8s
        - linux
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"

---
# 自动缩放
apiVersion: actions.summerwind.dev/v1alpha1
kind: HorizontalRunnerAutoscaler
metadata:
  name: runner-autoscaler
spec:
  scaleTargetRef:
    name: runner-deployment
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: TotalNumberOfQueuedAndInProgressWorkflowRuns
      repositoryNames:
        - my-org/my-repo
```

## 八、Runner安全加固

```bash
# 1. 使用非root用户运行
sudo useradd -m -s /bin/bash runner
sudo -u runner ./config.sh --url ... --token ...

# 2. 限制网络访问
# 仅允许访问GitHub API
iptables -A OUTPUT -d github.com -j ACCEPT
iptables -A OUTPUT -d api.github.com -j ACCEPT
iptables -A OUTPUT -j DROP

# 3. 限制文件系统
# 使用SELinux或AppArmor限制Runner进程权限

# 4. 定期清理
# 构建后清理工作空间
./config.sh --ephemeral  # 一次性Runner

# 5. 监控异常
# 监控Runner进程的CPU、内存、网络使用
# 检测异常的命令执行
```

## 九、Runner标签管理

```yaml
# 在工作流中使用标签选择Runner
jobs:
  gpu-job:
    runs-on: [self-hosted, linux, gpu]
    steps:
      - run: nvidia-smi  # GPU任务

  arm-job:
    runs-on: [self-hosted, linux, arm64]
    steps:
      - run: uname -m  # ARM任务

  docker-job:
    runs-on: [self-hosted, linux, docker]
    steps:
      - run: docker build -t myapp .
```
