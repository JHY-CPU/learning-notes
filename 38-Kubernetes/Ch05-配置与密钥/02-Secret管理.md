# Secret管理

## 一、概念说明

Secret 存储敏感数据，类型包括 Opaque、kubernetes.io/tls、kubernetes.io/dockerconfigjson 等。

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
stringData:           # 明文（会自动 base64 编码）
  username: admin
  password: P@ssw0rd
data:                 # 已 base64 编码
  api-key: c2VjcmV0
```

## 二、具体用法

### 2.1 Secret 类型

```bash
# Opaque：通用类型
kubectl create secret generic my-secret --from-literal=key=value

# TLS 证书
kubectl create secret tls tls-secret --cert=tls.crt --key=tls.key

# Docker 仓库凭据
kubectl create secret docker-registry regcred \
  --docker-server=registry.io \
  --docker-username=user \
  --docker-password=pass

# ServiceAccount Token
kubectl create token my-serviceaccount
```

### 2.2 拉取私有镜像

```yaml
spec:
  imagePullSecrets:
  - name: regcred
  containers:
  - name: app
    image: registry.example.com/my-app:v1
```

### 2.3 etcd 加密

```yaml
# /etc/kubernetes/encryption-config.yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: <base64-encoded-32-byte-key>
    - identity: {}
```

### 2.4 External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: vault-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: app-secret
  data:
  - secretKey: password
    remoteRef:
      key: secret/data/myapp
      property: password
```

### 2.5 Secret 轮换

```bash
# 创建新版本
kubectl create secret generic db-password-v2 \
  --from-literal=password=newpassword

# 更新引用
kubectl patch deployment app \
  -p '{"spec":{"template":{"spec":{"volumes":[{"name":"secrets","secret":{"secretName":"db-password-v2"}}]}}}}'

# 删除旧版本
kubectl delete secret db-password-v1
```

## 三、注意事项与常见陷阱

1. **非加密存储**：默认 base64 编码，必须启用 etcd 加密
2. **RBAC 控制**：严格限制 Secret 的访问权限
3. **日志泄漏**：避免将 Secret 打印到日志
4. **Git 安全**：不要将 Secret 提交到 Git，使用 Sealed Secrets 或 SOPS
5. **轮换策略**：定期轮换密码和证书
