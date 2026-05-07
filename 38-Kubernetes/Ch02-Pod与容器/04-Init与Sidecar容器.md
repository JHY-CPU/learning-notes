# Init与Sidecar容器

## 一、概念说明

Init 容器在主容器启动前运行，Sidecar 容器与主容器协同工作。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-with-sidecar
spec:
  initContainers:
  - name: init-db-check
    image: busybox
    command: ['sh', '-c', 'until nslookup mysql; do sleep 2; done']
  containers:
  - name: web
    image: nginx:1.25
    volumeMounts:
    - name: logs
      mountPath: /var/log/nginx
  - name: log-shipper
    image: fluent/fluentd:v1.16
    volumeMounts:
    - name: logs
      mountPath: /var/log/nginx
      readOnly: true
  volumes:
  - name: logs
    emptyDir: {}
```

## 二、具体用法

### 2.1 Init 容器模式

```yaml
initContainers:
# 等待依赖服务
- name: wait-for-db
  image: busybox
  command: ['sh', '-c', 'until nc -z mysql 3306; do sleep 1; done']

# 下载配置
- name: fetch-config
  image: curlimages/curl
  command: ['sh', '-c', 'curl -o /config/app.yaml http://config-server/config']

# 权限设置
- name: fix-permissions
  image: busybox
  command: ['sh', '-c', 'chown -R 1000:1000 /data']
  volumeMounts:
  - name: data
    mountPath: /data
```

### 2.2 Sidecar 容器模式

```yaml
containers:
# 主应用容器
- name: app
  image: my-app:v1
  ports:
  - containerPort: 8080
  volumeMounts:
  - name: shared-data
    mountPath: /app/data

# Sidecar: 日志收集
- name: log-collector
  image: fluent/fluentd:v1.16
  volumeMounts:
  - name: shared-data
    mountPath: /app/data
    readOnly: true

# Sidecar: 代理
- name: envoy-proxy
  image: envoyproxy/envoy:v1.28
  ports:
  - containerPort: 9901

volumes:
- name: shared-data
  emptyDir: {}
```

### 2.3 Sidecar 代理模式（Service Mesh）

```yaml
# Istio 注入的 Sidecar
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  annotations:
    sidecar.istio.io/inject: "true"
spec:
  containers:
  - name: app
    image: my-app:v1
  # Envoy sidecar 自动注入
```

### 2.4 Init 失败处理

```bash
# 查看 Init 容器状态
kubectl describe pod my-pod
# 看 Events 和 Init Containers 部分

# 查看 Init 容器日志
kubectl logs my-pod -c init-db-check

# Init 容器失败会反复重试，直到成功
```

### 2.5 容器启动顺序

```yaml
# K8s 1.28+ 支持 Sidecar 容器（restartPolicy: Always）
spec:
  initContainers:
  - name: sidecar
    image: my-sidecar:v1
    restartPolicy: Always  # 使其成为 Sidecar
  containers:
  - name: app
    image: my-app:v1
```

## 三、注意事项与常见陷阱

1. **Init 顺序**：Init 容器按定义顺序执行，必须全部成功
2. **Sidecar 资源**：Sidecar 容器会占用 Pod 资源配额
3. **日志管理**：Sidecar 共享 volume 时注意权限
4. **健康检查**：Sidecar 容器也应配置探针
5. **调试困难**：Sidecar 故障可能影响主容器
