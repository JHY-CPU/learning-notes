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

## 三、PersistentVolume 与 PVC

### 3.1 PersistentVolume (PV)

PV 是集群级别的存储资源，由管理员预先配置或通过 StorageClass 动态创建：

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  nfs:
    server: nfs-server.example.com
    path: /exports/data
```

### 3.2 PersistentVolumeClaim (PVC)

PVC 是用户对存储的请求，K8s 自动将其绑定到合适的 PV：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd
```

### 3.3 Access Modes 详解

| 模式 | 说明 | 支持的存储 |
|------|------|-----------|
| ReadWriteOnce (RWO) | 单节点读写 | 大多数块存储 |
| ReadOnlyMany (ROX) | 多节点只读 | NFS, CephFS |
| ReadWriteMany (RWX) | 多节点读写 | NFS, CephFS |
| ReadWriteOncePod (RWOP) | 单Pod读写 | CSI 驱动 |

### 3.4 StorageClass 动态供给

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iopsPerGB: "50"
  encrypted: "true"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

## 四、Volume 生命周期与数据持久化

```
用户创建 PVC → K8s 寻找匹配的 PV/动态创建 → PVC绑定到PV → Pod挂载PVC使用
                                                          ↓
Pod删除 → PV数据保留(Retain)/删除(Delete)/回收(Recycle-已废弃)
```

### Reclaim Policy

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| Retain | 保留数据，需管理员手动清理 | 生产数据库 |
| Delete | 自动删除底层存储 | 临时数据 |
| Recycle | (已废弃) 基本清理后可重用 | - |

## 五、注意事项与常见陷阱

1. **emptyDir 非持久**：Pod 删除后数据丢失
2. **hostPath 风险**：生产环境避免使用，安全隐患大
3. **权限问题**：注意 volume 的 defaultMode 和文件权限
4. **子路径**：使用 subPath 挂载单个文件而非整个目录
5. **只读挂载**：使用 readOnly: true 防止容器修改配置
6. **PVC 与 PV 一对一绑定**：绑定后不可更改，需先释放
7. **StorageClass 的 volumeBindingMode**：Immediate 模式可能导致跨可用区问题，推荐 WaitForFirstConsumer
8. **CSI 驱动兼容性**：不同 CSI 驱动支持的 accessModes 不同，需查阅文档确认
