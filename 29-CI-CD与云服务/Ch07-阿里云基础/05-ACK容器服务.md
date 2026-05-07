# ACK容器服务

## 一、概念说明

ACK（Alibaba Cloud Container Service for Kubernetes）是阿里云的托管Kubernetes服务，支持K8s集群的创建、管理和运维。

| 集群类型 | 说明 | 适用场景 |
|----------|------|----------|
| 专业托管版 | 托管Master节点 | 生产环境 |
| Serverless K8s | 无服务器K8s | 事件驱动 |
| 边缘集群 | 边缘节点管理 | IoT/边缘计算 |

## 二、具体用法

### 创建集群

```bash
# 创建ACK托管集群
aliyun cs CreateCluster \
    --name production-k8s \
    --region-id cn-hangzhou \
    --cluster-type ManagedKubernetes \
    --kubernetes-version 1.28.3-aliyun.1 \
    --vpcid vpc-bp1xxxxxxxx \
    --vswitchids '["vsw-bp1xxxxxxxx","vsw-bp1yyyyyyyy"]' \
    --pod-cidr 172.20.0.0/16 \
    --service-cidr 172.21.0.0/20 \
    --worker-instance-type ecs.g7.xlarge \
    --num-of-nodes 3 \
    --os Alinux3 \
    --disable-rollback true \
    --endpoint-public-access false
```

### 部署应用

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: registry.cn-hangzhou.aliyuncs.com/myapp/web:v1.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: web-app-svc
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8080
```

```bash
# 部署应用
kubectl apply -f deployment.yaml

# 查看状态
kubectl get pods -n production
kubectl get svc -n production

# 水平扩缩容
kubectl scale deployment web-app --replicas=5 -n production

# 配置HPA
kubectl autoscale deployment web-app \
    --min=3 --max=10 --cpu-percent=70 \
    -n production
```

### 使用ACR镜像仓库

```bash
# 登录ACR
docker login --username=your_account registry.cn-hangzhou.aliyuncs.com

# 构建并推送
docker build -t registry.cn-hangzhou.aliyuncs.com/myapp/web:v1.0 .
docker push registry.cn-hangzhou.aliyuncs.com/myapp/web:v1.0

# 创建镜像仓库
aliyun cr CreateRepository \
    --RepoName myapp/web \
    --RepoNamespace myapp \
    --RepoType PUBLIC \
    --Summary "Web应用镜像仓库"
```

### Ingress配置

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
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-app-svc
            port:
              number: 80
```

## 三、注意事项与常见陷阱

1. **网络规划**：Pod和Service CIDR不要与VPC CIDR重叠
2. **节点规格**：选择足够大的节点规格避免资源不足
3. **镜像拉取**：配置ACR免密钥拉取，避免公开镜像仓库
4. **存储选择**：使用云盘CSI驱动，根据IO需求选择磁盘类型
5. **日志采集**：配置日志服务采集容器日志
6. **监控告警**：启用Prometheus监控集群和应用指标
7. **安全加固**：启用Pod安全策略，限制特权容器
