# Helm基础

## 一、概念说明

Helm 是 K8s 包管理器，通过 Chart 管理应用的安装、升级、回滚。

```bash
# 安装 Helm
brew install helm    # macOS
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 添加仓库
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# 安装应用
helm install my-nginx bitnami/nginx --namespace web --create-namespace
```

## 二、具体用法

### 2.1 常用命令

```bash
# 搜索
helm search repo nginx
helm search hub nginx          # Helm Hub

# 安装
helm install my-app ./my-chart
helm install my-app bitnami/nginx --version 15.0.0
helm install my-app ./chart -f values.yaml -f prod-values.yaml

# 管理
helm list
helm status my-app
helm history my-app

# 升级
helm upgrade my-app ./my-chart
helm upgrade my-app bitnami/nginx --set replicaCount=3

# 回滚
helm rollback my-app 1

# 卸载
helm uninstall my-app
```

### 2.2 创建 Chart

```bash
helm create my-chart
```

```
my-chart/
├── Chart.yaml          # Chart 元数据
├── values.yaml         # 默认配置值
├── charts/             # 依赖 Charts
├── templates/          # 模板文件
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── _helpers.tpl    # 模板辅助函数
│   └── tests/
└── .helmignore
```

### 2.3 values.yaml

```yaml
# values.yaml
replicaCount: 2

image:
  repository: nginx
  tag: "1.25"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 256Mi

ingress:
  enabled: false
  host: example.com
```

### 2.4 模板语法

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-chart.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        {{- if .Values.resources }}
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
        {{- end }}
```

### 2.5 依赖管理

```yaml
# Chart.yaml
dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
```

```bash
helm dependency update ./my-chart
```

## 三、注意事项与常见陷阱

1. **版本管理**：每次修改 Chart.yaml 的 version 字段
2. **values 覆盖**：命令行 --set 优先级最高
3. **模板调试**：使用 `helm template` 预览生成的 YAML
4. **历史记录**：默认保留 10 条历史，可配置
5. **安全存储**：Helm 3 使用 Secrets 存储 release 信息
