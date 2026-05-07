# Helm 包管理

## 一、Helm Chart 结构

```
order-service/
├── Chart.yaml          # Chart 元信息
├── values.yaml         # 默认配置值
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── hpa.yaml
│   └── _helpers.tpl    # 模板函数
└── charts/             # 依赖 Chart
```

## 二、Chart.yaml

```yaml
apiVersion: v2
name: order-service
description: 订单服务 Helm Chart
type: application
version: 1.0.0
appVersion: "1.2.3"
dependencies:
  - name: mysql
    version: "9.x.x"
    repository: "https://charts.bitnami.com"
    condition: mysql.enabled
```

## 三、values.yaml

```yaml
replicaCount: 3

image:
  repository: registry.example.com/order-service
  tag: "v1.2.3"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  host: api.example.com
  path: /api/orders

resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70

env:
  SPRING_PROFILES_ACTIVE: k8s

mysql:
  enabled: false
```

## 四、常用操作

```bash
# 安装
helm install order-service ./order-service -n microservices

# 升级
helm upgrade order-service ./order-service --set image.tag=v1.3.0

# 回滚
helm rollback order-service 1

# 查看历史
helm history order-service

# 模板渲染（调试）
helm template order-service ./order-service
```

## 五、注意事项

1. **values.yaml 管理环境差异**
2. **Chart 版本遵循语义化版本**
3. **使用条件渲染支持可选组件**
4. **Helmfile 管理多环境部署**
5. **Chart 仓库统一管理**
