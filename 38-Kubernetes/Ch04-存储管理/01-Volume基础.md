# Volume基础

## 一、概念说明

Volume 为 Pod 提供持久化或临时存储。K8s 支持多种 Volume 类型。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: with-volume
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: data
      mountPath: /app/data
  volumes:
  - name: data
    emptyDir: {}
```

## 二、具体用法

### 2.1 emptyDir 临时存储

```yaml
volumes:
- name: cache
  emptyDir:
    medium: Memory    # 使用内存（tmpfs）
    sizeLimit: 1Gi
```

### 2.2 hostPath 主机路径

```yaml
volumes:
- name: host-data
  hostPath:
    path: /data
    type: DirectoryOrCreate
# 仅适用于单节点开发，生产环境避免使用
```

### 2.3 ConfigMap 挂载

```yaml
volumes:
- name: config
  configMap:
    name: app-config
    items:
    - key: application.yaml
      path: app.yaml
```

### 2.4 Secret 挂载

```yaml
volumes:
- name: secrets
  secret:
    secretName: app-secret
    defaultMode: 0400
```

### 2.5 Projected Volume

```yaml
volumes:
- name: projected
  projected:
    sources:
    - secret:
        name: app-secret
    - configMap:
        name: app-config
    - downwardAPI:
        items:
        - path: "labels"
          fieldRef:
            fieldPath: metadata.labels
```

## 三、注意事项与常见陷阱

1. **emptyDir 非持久**：Pod 删除后数据丢失
2. **hostPath 风险**：生产环境避免使用，安全隐患大
3. **权限问题**：注意 volume 的 defaultMode 和文件权限
4. **子路径**：使用 subPath 挂载单个文件而非整个目录
5. **只读挂载**：使用 readOnly: true 防止容器修改配置
