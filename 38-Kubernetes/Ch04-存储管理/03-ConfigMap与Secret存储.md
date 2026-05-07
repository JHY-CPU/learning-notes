# ConfigMap与Secret存储

## 一、概念说明

ConfigMap 存储非敏感配置，Secret 存储敏感数据（密码、证书）。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  database_url: "postgres://db:5432/myapp"
  log_level: "info"
  application.yaml: |
    server:
      port: 8080
    logging:
      level: ${LOG_LEVEL}
```

## 二、具体用法

### 2.1 创建方式

```bash
# 从字面量
kubectl create configmap app-config \
  --from-literal=DB_HOST=mysql \
  --from-literal=DB_PORT=3306

# 从文件
kubectl create configmap app-config \
  --from-file=application.yaml \
  --from-file=logging.conf

# 从目录
kubectl create configmap app-config --from-file=config/

# 从 env 文件
kubectl create configmap app-config --from-env-file=config.env
```

### 2.2 作为环境变量

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1
    envFrom:
    - configMapRef:
        name: app-config
    env:
    - name: DB_HOST
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_url
```

### 2.3 作为 Volume 挂载

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: config
      mountPath: /app/config
      readOnly: true
  volumes:
  - name: config
    configMap:
      name: app-config
      items:
      - key: application.yaml
        path: app.yaml
      - key: logging.conf
        path: logging.conf
```

### 2.4 Secret 使用

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
stringData:
  username: admin
  password: S3cr3tP@ss!
  ca.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

```bash
# 创建 Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=S3cr3tP@ss!

kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key

kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=pass
```

### 2.5 Secret 作为 Volume

```yaml
spec:
  containers:
  - name: app
    volumeMounts:
    - name: secrets
      mountPath: /app/secrets
      readOnly: true
  volumes:
  - name: secrets
    secret:
      secretName: db-secret
      defaultMode: 0400
```

## 三、注意事项与常见陷阱

1. **Secret 非加密**：Secret 只是 base64 编码，不是加密，etcd 中需加密存储
2. **大小限制**：ConfigMap/Secret 限制 1MB
3. **更新延迟**：挂载的 ConfigMap 更新可能有延迟（kubelet 同步周期）
4. **不可变**：设置 immutable: true 可提升性能
5. **敏感数据**：避免将 Secret 放入环境变量，优先挂载为文件
