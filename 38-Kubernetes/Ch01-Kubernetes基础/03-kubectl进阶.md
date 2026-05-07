# kubectl进阶

## 一、概念说明

kubectl 是与 K8s 集群交互的命令行工具，支持资源的创建、查看、更新、删除。

```bash
# 语法格式
kubectl [command] [TYPE] [NAME] [flags]

# 示例
kubectl get pods -n default -o wide
kubectl describe pod nginx-pod
kubectl logs -f nginx-pod --tail=100
```

## 二、具体用法

### 2.1 输出格式

```bash
# 默认表格输出
kubectl get pods

# JSON 输出
kubectl get pods -o json

# YAML 输出
kubectl get pods -o yaml

# 自定义列
kubectl get pods -o custom-columns=NAME:.metadata.name,STATUS:.status.phase

# 宽格式（更多信息）
kubectl get pods -o wide

# 仅名称
kubectl get pods -o name
```

### 2.2 调试与诊断

```bash
# 查看 Pod 事件
kubectl describe pod <pod-name>

# 查看日志
kubectl logs <pod-name>                  # 当前日志
kubectl logs <pod-name> --previous       # 上一次容器日志
kubectl logs -f <pod-name>               # 实时跟踪
kubectl logs -l app=web                  # 按标签查日志

# 进入容器
kubectl exec -it <pod-name> -- /bin/sh
kubectl exec -it <pod-name> -c sidecar -- /bin/sh  # 指定容器

# 端口转发
kubectl port-forward <pod-name> 8080:80
kubectl port-forward svc/my-service 8080:80

# 复制文件
kubectl cp <pod-name>:/app/logs ./logs
kubectl cp ./config.yaml <pod-name>:/app/config.yaml
```

### 2.3 批量操作

```bash
# 按标签删除
kubectl delete pods -l app=old-version

# 强制删除
kubectl delete pod <pod-name> --grace-period=0 --force

# 删除所有同类型资源
kubectl delete pods --all -n test-namespace

# 按条件查询
kubectl get pods --field-selector=status.phase=Running
```

### 2.4 资源比较与差异

```bash
# 查看当前配置与文件的差异
kubectl diff -f deployment.yaml

# 查看当前配置
kubectl get deployment nginx -o yaml

# 导出资源定义
kubectl get deployment nginx -o yaml > nginx.yaml

# 补丁更新
kubectl patch deployment nginx -p '{"spec":{"replicas":5}}'
```

### 2.5 集群信息

```bash
# 节点资源使用
kubectl top nodes
kubectl top pods

# 集群事件
kubectl get events --sort-by='.lastTimestamp'

# API 资源列表
kubectl api-resources
kubectl api-versions

# 配置管理
kubectl config get-contexts
kubectl config use-context prod-cluster
kubectl config set-credentials user --token=<token>
```

## 三、注意事项与常见陷阱

1. **命名空间**：默认操作 default 命名空间，用 `-n` 指定
2. **权限问题**：某些操作需要相应 RBAC 权限
3. **日志滚动**：容器重启后日志丢失，需要外部日志收集
4. **上下文切换**：操作前确认当前集群上下文
5. **资源缩写**：pod/po、service/svc、deployment/deploy 等
