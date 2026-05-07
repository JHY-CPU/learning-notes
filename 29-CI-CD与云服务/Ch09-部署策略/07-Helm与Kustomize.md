# Helm与Kustomize

## 一、概念说明

Helm和Kustomize是Kubernetes的两种配置管理工具。Helm是包管理器，使用模板化方式；Kustomize是原生的配置定制工具，使用覆盖方式。

| 工具 | 方式 | 适合场景 |
|------|------|----------|
| Helm | 模板+参数 | 复杂应用打包分发 |
| Kustomize | 基础+覆盖 | 多环境配置管理 |

## 二、具体用法

### Helm Chart结构

```bash
my-chart/
├── Chart.yaml       # Chart元数据
├── values.yaml      # 默认值
├── values-dev.yaml  # 开发环境值
├── values-prod.yaml # 生产环境值
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── _helpers.tpl
│   └── NOTES.txt
└── charts/          # 依赖Charts
```

### Chart.yaml

```yaml
apiVersion: v2
name: web-app
description: Web应用Helm Chart
version: 1.0.0
appVersion: "2.0"
dependencies:
  - name: redis
    version: "17.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
```

### values.yaml

```yaml
replicaCount: 3

image:
  repository: myregistry/web-app
  tag: "2.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  - name: APP_ENV
    value: production
  - name: DB_HOST
    valueFrom:
      secretKeyRef:
        name: db-secret
        key: host
```

### 模板文件

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "web-app.fullname" . }}
  labels:
    {{- include "web-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "web-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "web-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - containerPort: 8080
        env:
        {{- toYaml .Values.env | nindent 8 }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
```

### Helm操作

```bash
# 安装
helm install web-app ./my-chart -f values-prod.yaml -n production

# 升级
helm upgrade web-app ./my-chart -f values-prod.yaml --set image.tag=2.1

# 回滚
helm rollback web-app 1

# 查看历史
helm history web-app

# 打包
helm package ./my-chart

# 添加仓库
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### Kustomize

```yaml
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
  - ingress.yaml
commonLabels:
  app: web-app
```

```yaml
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
bases:
  - ../../base
replicas:
  - name: web-app
    count: 5
patchesStrategicMerge:
  - deployment-patch.yaml
configMapGenerator:
  - name: app-config
    literals:
      - ENV=production
      - LOG_LEVEL=info
images:
  - name: myregistry/web-app
    newTag: "2.0"
```

```yaml
# overlays/production/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  template:
    spec:
      containers:
      - name: web
        resources:
          requests:
            cpu: 1000m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 2Gi
```

```bash
# Kustomize操作
kustomize build overlays/production | kubectl apply -f -

# kubectl内置支持
kubectl apply -k overlays/production/
```

## 三、注意事项与常见陷阱

1. **Helm版本管理**：Chart版本和App版本要区分管理
2. **values安全**：secrets不要放在values.yaml中，使用外部secrets
3. **模板调试**：使用`helm template`预渲染检查
4. **Kustomize环境差异**：仅覆盖必要的差异配置
5. **依赖管理**：Helm依赖要锁定版本
6. **升级风险**：Helm升级前先用`--dry-run`预览
7. **选择依据**：简单场景用Kustomize，复杂场景用Helm
