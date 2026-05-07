# Pod详解

## 一、概念说明

Pod 是 K8s 最小调度单元，包含一个或多个共享网络和存储的容器。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-server
  labels:
    app: web
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    ports:
    - containerPort: 80
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

## 二、具体用法

### 2.1 Pod 资源配置

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1
    resources:
      requests:         # 调度需要的最小资源
        cpu: 100m       # 0.1 核
        memory: 128Mi
      limits:           # 允许使用的最大资源
        cpu: 500m
        memory: 256Mi
```

### 2.2 环境变量

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1
    env:
    - name: DB_HOST
      value: "mysql.default.svc.cluster.local"
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password
    - name: POD_NAME
      valueFrom:
        fieldRef:
          fieldPath: metadata.name
    - name: NODE_NAME
      valueFrom:
        fieldRef:
          fieldPath: spec.nodeName
```

### 2.3 生命周期钩子

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1
    lifecycle:
      postStart:
        exec:
          command: ["/bin/sh", "-c", "echo started > /tmp/started"]
      preStop:
        exec:
          command: ["/bin/sh", "-c", "nginx -s quit; sleep 5"]
    # 或使用 HTTP 钩子
    lifecycle:
      postStart:
        httpGet:
          path: /healthz
          port: 8080
```

### 2.4 重启策略

```yaml
spec:
  restartPolicy: Always       # 默认，总是重启
  # restartPolicy: OnFailure  # 失败时重启
  # restartPolicy: Never      # 从不重启
```

### 2.5 临时容器（Ephemeral）

```bash
# 调试运行中的 Pod
kubectl debug -it <pod-name> --image=busybox --target=app
kubectl debug <pod-name> --copy-to=debug-pod -it --container=app -- /bin/sh
```

## 三、注意事项与常见陷阱

1. **单 Pod 单应用**：尽量一个 Pod 一个主容器
2. **资源设置**：必须设置 requests 和 limits
3. **镜像标签**：避免使用 `latest` 标签
4. **健康检查**：重要服务必须配置探针
5. **优雅关闭**：配置 preStop 钩子和 terminationGracePeriodSeconds
