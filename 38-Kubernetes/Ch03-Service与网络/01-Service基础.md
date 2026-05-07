# Service基础

## 一、概念说明

Service 为 Pod 提供稳定的网络入口，通过标签选择器将流量路由到后端 Pod。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

## 二、具体用法

### 2.1 Service 类型

```yaml
# ClusterIP（默认）：集群内部访问
apiVersion: v1
kind: Service
metadata:
  name: internal-service
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080

---
# NodePort：通过节点端口访问
apiVersion: v1
kind: Service
metadata:
  name: nodeport-service
spec:
  type: NodePort
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080  # 30000-32767

---
# LoadBalancer：云负载均衡器
apiVersion: v1
kind: Service
metadata:
  name: lb-service
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
```

### 2.2 无头服务（Headless）

```yaml
apiVersion: v1
kind: Service
metadata:
  name: headless-service
spec:
  type: ClusterIP
  clusterIP: None  # 无头服务
  selector:
    app: db
  ports:
  - port: 5432
    targetPort: 5432
# DNS 返回所有 Pod IP 而非 Service IP
```

### 2.3 ExternalName

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: db.example.com
# 将服务映射到外部 DNS 名称
```

### 2.4 多端口 Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: multi-port-service
spec:
  selector:
    app: my-app
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: https
    port: 443
    targetPort: 8443
  - name: grpc
    port: 50051
    targetPort: 50051
```

### 2.5 服务发现

```bash
# DNS 查询
nslookup web-service.default.svc.cluster.local

# 环境变量（Pod 自动注入）
# WEB_SERVICE_SERVICE_HOST=10.96.x.x
# WEB_SERVICE_SERVICE_PORT=80

# 在 Pod 中访问
curl http://web-service:80
curl http://web-service.default:80
curl http://web-service.default.svc.cluster.local:80
```

## 三、注意事项与常见陷阱

1. **会话保持**：默认使用轮询，需要会话亲和性设置 `sessionAffinity: ClientIP`
2. **外部流量**：NodePort 可能在未运行 Pod 的节点上访问不到
3. **端口冲突**：避免 NodePort 端口冲突
4. **服务发现**：优先使用 DNS 而非环境变量
5. **负载均衡**：Service 级别的负载均衡是 L4（TCP/UDP）
