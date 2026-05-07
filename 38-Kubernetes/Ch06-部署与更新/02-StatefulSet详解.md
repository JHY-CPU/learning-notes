# StatefulSet详解

## 一、概念说明

StatefulSet 管理有状态应用，提供稳定网络标识、有序部署和持久存储。

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "web"
  replicas: 3
  selector:
    matchLabels:
      app: web
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
        volumeMounts:
        - name: data
          mountPath: /usr/share/nginx/html
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 1Gi
```

## 二、具体用法

### 2.1 稳定网络标识

```
Pod 名称：web-0, web-1, web-2
DNS 名称：web-0.web.default.svc.cluster.local
         web-1.web.default.svc.cluster.local
         web-2.web.default.svc.cluster.local
```

### 2.2 有序操作

```yaml
spec:
  podManagementPolicy: OrderedReady  # 默认，有序创建
  # podManagementPolicy: Parallel    # 并行创建

  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 0       # 0=全部更新，2=只更新 index>=2
```

### 2.3 Headless Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  clusterIP: None          # 必须使用无头服务
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
```

### 2.4 扩缩容

```bash
# 扩容
kubectl scale statefulset web --replicas=5

# 缩容（按逆序：4,3,2...）
kubectl scale statefulset web --replicas=3

# 查看缩容状态
kubectl get pods -l app=web
```

### 2.5 分区更新

```bash
# 只更新最后一个 Pod
kubectl patch statefulset web -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":2}}}}'

# 逐步更新
kubectl patch statefulset web -p '{"spec":{"updateStrategy":{"rollingUpdate":{"partition":1}}}}'
```

## 三、注意事项与常见陷阱

1. **删除行为**：删除 StatefulSet 不会删除 PVC，需手动清理
2. **扩缩容顺序**：扩容按 0,1,2 顺序，缩容按逆序
3. **卡住状态**：Pod 失败可能导致 StatefulSet 卡住
4. **存储管理**：PVC 与 Pod 同生命周期，注意存储成本
5. **并行模式**：数据库等应用可能需要 Parallel 模式
