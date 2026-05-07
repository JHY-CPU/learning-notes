# Ingress控制器

## 一、概念说明

Ingress 提供 L7（HTTP/HTTPS）负载均衡，支持基于域名和路径的路由。

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 80
      - path: /web
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

## 二、具体用法

### 2.1 安装 NGINX Ingress

```bash
# 使用 Helm 安装
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx

# 或使用 minikube
minikube addons enable ingress
```

### 2.2 TLS 配置

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
spec:
  tls:
  - hosts:
    - example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

```bash
# 创建 TLS Secret
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key
```

### 2.3 高级路由

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: advanced-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: api-v1
            port:
              number: 80
      - path: /v2
        pathType: Prefix
        backend:
          service:
            name: api-v2
            port:
              number: 80
  - host: admin.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-panel
            port:
              number: 80
```

### 2.4 IngressClass

```yaml
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  name: nginx
spec:
  controller: k8s.io/ingress-nginx
```

### 2.5 调试与验证

```bash
# 查看 Ingress
kubectl get ingress
kubectl describe ingress web-ingress

# 查看 Ingress Controller 日志
kubectl logs -n ingress-nginx <controller-pod>

# 测试路由
curl -H "Host: example.com" http://<ingress-ip>/api
```

## 三、注意事项与常见陷阱

1. **控制器依赖**：Ingress 需要安装 Ingress Controller 才能工作
2. **TLS 管理**：证书过期需要手动更新或使用 cert-manager
3. **注解差异**：不同 Ingress Controller 的注解语法不同
4. **路径匹配**：注意 pathType（Exact/Prefix/ImplementationSpecific）
5. **性能调优**：Ingress Controller 可能成为瓶颈，需要监控
