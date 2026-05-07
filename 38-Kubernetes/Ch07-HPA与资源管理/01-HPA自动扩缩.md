# HPA自动扩缩

## 一、概念说明

HPA（Horizontal Pod Autoscaler）根据指标自动调整 Pod 副本数。

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 二、具体用法

### 2.1 CPU/Memory 指标

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Resource
  resource:
    name: memory
    target:
      type: Utilization
      averageUtilization: 80
```

### 2.2 自定义指标

```yaml
metrics:
- type: Pods
  pods:
    metric:
      name: http_requests_per_second
    target:
      type: AverageValue
      averageValue: "1000"
```

### 2.3 外部指标

```yaml
metrics:
- type: External
  external:
    metric:
      name: sqs_queue_length
      selector:
        matchLabels:
          queue: my-queue
    target:
      type: AverageValue
      averageValue: "30"
```

### 2.4 行为配置

```yaml
behavior:
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
    - type: Percent
      value: 10
      periodSeconds: 60
  scaleUp:
    stabilizationWindowSeconds: 0
    policies:
    - type: Percent
      value: 100
      periodSeconds: 15
    - type: Pods
      value: 4
      periodSeconds: 15
    selectPolicy: Max
```

### 2.5 安装 Metrics Server

```bash
# 安装 Metrics Server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 验证
kubectl top pods
kubectl top nodes

# 查看 HPA 状态
kubectl get hpa
kubectl describe hpa web-hpa
```

## 三、注意事项与常见陷阱

1. **Metrics Server**：HPA 依赖 Metrics Server，必须先安装
2. **资源请求**：HPA 的 Utilization 基于 requests 计算，必须设置 requests
3. **指标延迟**：Metrics 有 15 秒到 2 分钟延迟
4. **抖动防护**：stabilizationWindowSeconds 防止频繁扩缩
5. **冷启动**：扩容后新 Pod 需要时间就绪
