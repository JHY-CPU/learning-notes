# Helm进阶

## 一、概念说明

Helm 进阶包括 Hooks、Tests、Library Charts、CI/CD 集成。

## 二、具体用法

### 2.1 Hooks

```yaml
# templates/pre-install-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ include "my-chart.fullname" . }}-migrate
  annotations:
    "helm.sh/hook": pre-install,pre-upgrade
    "helm.sh/hook-weight": "0"
    "helm.sh/hook-delete-policy": before-hook-creation
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: migrate
        image: my-app:{{ .Values.image.tag }}
        command: ["python", "manage.py", "migrate"]
```

### 2.2 Tests

```yaml
# templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: {{ include "my-chart.fullname" . }}-test
  annotations:
    "helm.sh/hook": test
spec:
  restartPolicy: Never
  containers:
  - name: test
    image: busybox
    command: ['wget']
    args: ['{{ include "my-chart.fullname" . }}:{{ .Values.service.port }}']
```

```bash
helm test my-app
```

### 2.3 Library Charts

```yaml
# Chart.yaml
apiVersion: v2
name: common
version: 0.1.0
type: library    # 库 Chart，不能直接安装
```

```yaml
# templates/_deployment.tpl
{{- define "common.deployment" -}}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount | default 1 }}
  # ...通用配置
{{- end }}
```

### 2.4 条件渲染

```yaml
# templates/service.yaml
{{- if .Values.service.enabled }}
apiVersion: v1
kind: Service
metadata:
  name: {{ include "my-chart.fullname" . }}
spec:
  type: {{ .Values.service.type }}
  ports:
  - port: {{ .Values.service.port }}
{{- end }}
```

### 2.5 CI/CD 集成

```yaml
# GitHub Actions
- name: Deploy to K8s
  run: |
    helm upgrade --install my-app ./chart \
      --namespace production \
      --set image.tag=${{ github.sha }} \
      --set replicaCount=3 \
      --wait --timeout 5m
```

```yaml
# GitLab CI
deploy:
  stage: deploy
  script:
    - helm upgrade --install my-app ./chart -f values-${CI_ENVIRONMENT_NAME}.yaml
  only:
    - main
```

## 三、注意事项与常见陷阱

1. **Hook 执行顺序**：hook-weight 控制执行顺序（数字越小越先执行）
2. **Chart 类型**：library Chart 不能安装，只能被引用
3. **模板调试**：`helm template --debug` 排查模板问题
4. **values 合并**：多个 values 文件按顺序合并
5. **回滚限制**：Hook 创建的资源不在回滚范围内
