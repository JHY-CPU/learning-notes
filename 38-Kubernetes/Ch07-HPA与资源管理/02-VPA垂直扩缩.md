# VPA垂直扩缩

## 一、概念说明

VPA（Vertical Pod Autoscaler）自动调整容器的 CPU/内存 requests 和 limits。

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: web-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-deployment
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: web
      minAllowed:
        cpu: 100m
        memory: 64Mi
      maxAllowed:
        cpu: 2
        memory: 4Gi
      controlledResources: ["cpu", "memory"]
```

## 二、具体用法

### 2.1 更新模式

```yaml
updatePolicy:
  updateMode: "Off"        # 只推荐，不执行
  updateMode: "Initial"    # 只在创建时应用
  updateMode: "Auto"       # 自动调整（会重启 Pod）
```

### 2.2 安装 VPA

```bash
# 安装 VPA
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# 验证
kubectl get pods -n kube-system | grep vpa
```

### 2.3 资源策略

```yaml
resourcePolicy:
  containerPolicies:
  - containerName: app
    mode: Auto
    minAllowed:
      cpu: 50m
      memory: 32Mi
    maxAllowed:
      cpu: 4
      memory: 8Gi
    controlledResources: ["cpu", "memory"]
  - containerName: sidecar
    mode: "Off"  # 不调整 sidecar
```

### 2.4 查看推荐

```bash
kubectl describe vpa web-vpa
# 看 Recommendation 部分

# 输出示例：
# Container Recommendations:
#   Container Name: web
#   Lower Bound:    cpu: 50m, memory: 50Mi
#   Target:         cpu: 200m, memory: 200Mi
#   Upper Bound:    cpu: 1, memory: 1Gi
```

### 2.5 VPA 与 HPA 配合

```yaml
# VPA 管理资源大小，HPA 管理副本数量
# 注意：不要对同一指标同时使用 VPA 和 HPA
# 可以：VPA 管理 memory，HPA 管理 CPU
```

## 三、注意事项与常见陷阱

1. **Pod 重启**：Auto 模式会重启 Pod 来应用新配置
2. **与 HPA 冲突**：不能对同一指标同时使用 VPA 和 HPA
3. **历史数据**：VPA 需要一段时间的数据才能给出准确推荐
4. **初始推荐**：使用 Off 模式先观察推荐值
5. **生产风险**：Auto 模式可能导致服务中断
