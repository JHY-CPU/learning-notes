# Deployment详解

## 一、概念说明

Deployment 管理 ReplicaSet 和 Pod，支持滚动更新、回滚、扩缩容。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx:1.25
        ports:
        - containerPort: 80
```

## 二、具体用法

### 2.1 滚动更新

```bash
# 更新镜像
kubectl set image deployment/web-deployment web=nginx:1.26

# 查看更新状态
kubectl rollout status deployment/web-deployment
kubectl rollout history deployment/web-deployment

# 回滚
kubectl rollout undo deployment/web-deployment
kubectl rollout undo deployment/web-deployment --to-revision=2
```

### 2.2 更新策略

```yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # 最多多出的 Pod 数
      maxUnavailable: 0    # 最多不可用的 Pod 数
  minReadySeconds: 10      # Pod 就绪后等待时间
  revisionHistoryLimit: 10 # 保留的 ReplicaSet 数量
```

### 2.3 扩缩容

```bash
# 手动扩缩容
kubectl scale deployment web-deployment --replicas=5

# 自动扩缩容
kubectl autoscale deployment web-deployment --min=2 --max=10 --cpu-percent=70
```

### 2.4 暂停与恢复

```bash
# 暂停更新（用于批量修改）
kubectl rollout pause deployment/web-deployment
kubectl set image deployment/web-deployment web=nginx:1.26
kubectl set env deployment/web-deployment DEBUG=true
kubectl rollout resume deployment/web-deployment
```

### 2.5 金丝雀发布

```yaml
# 新版本 Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-canary
spec:
  replicas: 1            # 少量新版本
  selector:
    matchLabels:
      app: web
      version: canary
  template:
    metadata:
      labels:
        app: web
        version: canary
    spec:
      containers:
      - name: web
        image: nginx:1.26
```

## 三、注意事项与常见陷阱

1. **selector 不可变**：创建后不能修改 selector
2. **资源清理**：回滚后旧 ReplicaSet 会保留（revisionHistoryLimit）
3. **maxSurge/maxUnavailable**：两者不能同时为 0
4. **健康检查**：没有 readiness 探针可能导致流量发到未就绪 Pod
5. **镜像拉取**：更新时确保新镜像已在仓库中
