# PV与PVC

## 一、概念说明

PV（PersistentVolume）是集群级存储资源，PVC（PersistentVolumeClaim）是用户对存储的请求。

```
用户 → PVC → PV → 实际存储（NFS/云盘/Ceph）
```

```yaml
# PV 定义
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-data
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  nfs:
    path: /data
    server: nfs-server.example.com
```

## 二、具体用法

### 2.1 PVC 定义

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-claim
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
```

### 2.2 Pod 使用 PVC

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: with-pvc
spec:
  containers:
  - name: app
    image: my-app:v1
    volumeMounts:
    - name: data
      mountPath: /app/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: data-claim
```

### 2.3 StorageClass 动态供应

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iopsPerGB: "50"
reclaimPolicy: Delete
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
```

### 2.4 访问模式

```
ReadWriteOnce (RWO): 单节点读写
ReadOnlyMany (ROX): 多节点只读
ReadWriteMany (RWX): 多节点读写（需文件系统支持）
ReadWriteOncePod (RWOP): 单 Pod 读写（K8s 1.22+）
```

### 2.5 扩展 PVC

```bash
# 需要 StorageClass 支持 allowVolumeExpansion: true
kubectl edit pvc data-claim
# 修改 spec.resources.requests.storage

# 或使用 patch
kubectl patch pvc data-claim -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'
```

## 三、注意事项与常见陷阱

1. **绑定不可逆**：PVC 绑定 PV 后不能解绑（除非删除 PVC）
2. **回收策略**：Retain/Recycle/Delete，生产建议用 Retain
3. **存储类匹配**：PVC 和 PV 的 storageClassName 必须匹配
4. **扩容限制**：并非所有存储类型都支持动态扩容
5. **等待消费者**：volumeBindingMode: WaitForFirstConsumer 更灵活
