# ConfigMap详解

## 一、概念说明

ConfigMap 将配置与容器镜像解耦，支持环境变量、命令行参数、配置文件三种注入方式。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: game-config
data:
  game.properties: |
    enemies=aliens
    lives=3
    enemies.cheat=true
  ui.properties: |
    color.good=purple
    color.bad=yellow
```

## 二、具体用法

### 2.1 热更新

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hot-reload
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: config
      mountPath: /app/config
  volumes:
  - name: config
    configMap:
      name: app-config
# 更新 ConfigMap 后，挂载的文件会自动更新
# 但进程需要 watch 文件变化或重启
```

### 2.2 不可变 ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: immutable-config
immutable: true    # 一旦创建不可修改
data:
  config: "value"
# 优点：减少 API Server 负载，防止意外修改
```

### 2.3 子路径挂载

```yaml
spec:
  containers:
  - name: app
    volumeMounts:
    - name: config
      mountPath: /app/config/app.yaml
      subPath: application.yaml  # 只挂载单个 key
  volumes:
  - name: config
    configMap:
      name: app-config
```

### 2.4 多 ConfigMap 合并

```yaml
spec:
  containers:
  - name: app
    envFrom:
    - configMapRef:
        name: base-config
    - configMapRef:
        name: env-config
    # 后面的会覆盖前面的同名 key
```

### 2.5 配置版本管理

```bash
# 使用标签管理版本
kubectl label configmap app-config version=v2
kubectl get configmap -l version=v2

# 导出备份
kubectl get configmap app-config -o yaml > config-backup.yaml
```

## 三、注意事项与常见陷阱

1. **大小限制**：每个 ConfigMap 最大 1MB
2. **更新延迟**：kubelet 同步周期默认 60 秒，环境变量不会热更新
3. **幂等性**：ConfigMap key 可能被其他 ConfigMap 覆盖
4. **引用计数**：ConfigMap 被 Pod 引用时不能删除
5. **命名空间**：ConfigMap 只能被同命名空间的 Pod 引用
