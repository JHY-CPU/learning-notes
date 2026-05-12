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

## 五、Helmfile 多环境管理

```yaml
# helmfile.yaml - 多环境统一管理
repositories:
  - name: stable
    url: https://charts.helm.sh/stable
  - name: bitnami
    url: https://charts.bitnami.com

releases:
  - name: order-service
    namespace: microservices
    chart: ./charts/order-service
    values:
      - ./values/common.yaml
    {{ if eq .Environment.Name "dev" }}
      - ./values/dev.yaml
    {{ else if eq .Environment.Name "staging" }}
      - ./values/staging.yaml
    {{ else if eq .Environment.Name "production" }}
      - ./values/production.yaml
    {{ end }}

  - name: user-service
    namespace: microservices
    chart: ./charts/user-service
    values:
      - ./values/common.yaml

environments:
  dev:
    values:
      - env: dev
  staging:
    values:
      - env: staging
  production:
    values:
      - env: production
```

```bash
# Helmfile 命令
helmfile -e dev apply          # 部署到 dev 环境
helmfile -e production diff    # 查看生产环境变更
helmfile -e production apply   # 部署到生产环境
helmfile -e production destroy # 销毁生产环境
```

## 六、Chart 测试

```yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "order-service.fullname" . }}-test-connection"
  annotations:
    "helm.sh/hook": test
spec:
  containers:
    - name: wget
      image: busybox
      command: ['wget']
      args: ['{{ include "order-service.fullname" . }}:{{ .Values.service.port }}/actuator/health']
  restartPolicy: Never
```

```bash
helm test order-service    # 运行 Helm 测试
```

## 七、注意事项

1. **values.yaml 管理环境差异** - 公共配置 + 环境覆盖
2. **Chart 版本遵循语义化版本** - major.minor.patch
3. **使用条件渲染支持可选组件** - `{{- if .Values.mysql.enabled }}`
4. **Helmfile 管理多环境部署** - 一个文件管理所有环境
5. **Chart 仓库统一管理** - 使用 Harbor ChartMuseum
6. **Chart 测试保证质量** - helm test 验证部署成功
