# DNS与服务发现

## 一、概念说明

K8s 内置 DNS 服务（CoreDNS），为 Service 和 Pod 提供 DNS 解析。

```
DNS 命名格式：
<Service>.<Namespace>.svc.cluster.local

示例：
web-service.default.svc.cluster.local
api-service.production.svc.cluster.local
```

## 二、具体用法

### 2.1 DNS 查询规则

```bash
# 在 Pod 内查询
nslookup web-service                    # 同命名空间
nslookup web-service.default            # 指定命名空间
nslookup web-service.default.svc        # 完整格式
nslookup web-service.default.svc.cluster.local  # FQDN

# 无头服务返回 Pod IP
nslookup headless-service.default.svc.cluster.local
# 返回多个 A 记录
```

### 2.2 CoreDNS 配置

```yaml
# CoreDNS ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
```

### 2.3 自定义 DNS

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: custom-dns
spec:
  dnsPolicy: "None"          # 使用自定义 DNS
  dnsConfig:
    nameservers:
    - 8.8.8.8
    searches:
    - my-company.com
    options:
    - name: ndots
      value: "5"
  containers:
  - name: app
    image: my-app:v1

---
# 其他 dnsPolicy 选项
# ClusterFirst: 默认，使用集群 DNS
# Default: 使用节点 DNS
# ClusterFirstWithHostNet: 主机网络 + 集群 DNS
```

### 2.4 服务别名

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  type: ExternalName
  externalName: web.example.com
# web.default.svc.cluster.local → web.example.com
```

### 2.5 调试 DNS

```bash
# 测试 DNS 解析
kubectl run dns-test --image=busybox --rm -it --restart=Never -- nslookup kubernetes.default

# 查看 CoreDNS 日志
kubectl logs -n kube-system -l k8s-app=kube-dns

# 检查 CoreDNS 健康
kubectl get pods -n kube-system -l k8s-app=kube-dns

# 查看 DNS 配置
kubectl exec <pod-name> -- cat /etc/resolv.conf
```

## 三、注意事项与常见陷阱

1. **搜索域**：Pod 内 /etc/resolv.conf 有搜索域配置
2. **ndots 配置**：默认 ndots=5，短名称会尝试搜索域拼接
3. **外部 DNS**：集群外部需要 DNS 记录指向 Ingress/LoadBalancer
4. **CoreDNS 资源**：大集群需要增加 CoreDNS 副本数
5. **缓存问题**：DNS 缓存可能导致服务发现延迟
