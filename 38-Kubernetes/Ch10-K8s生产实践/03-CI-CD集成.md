# CI/CD集成

## 一、概念说明

K8s CI/CD 流水线：代码提交 → 构建镜像 → 推送仓库 → 更新部署。

```yaml
# GitHub Actions 示例
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build & Push
      run: |
        docker build -t $REGISTRY/my-app:${{ github.sha }} .
        docker push $REGISTRY/my-app:${{ github.sha }}
    - name: Deploy
      run: |
        kubectl set image deployment/my-app app=$REGISTRY/my-app:${{ github.sha }}
```

## 二、具体用法

### 2.1 GitOps with ArgoCD

```yaml
# Application 定义
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/user/k8s-manifests.git
    targetRevision: main
    path: apps/my-app
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

### 2.2 Kustomize 管理多环境

```
k8s/
├── base/
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   └── service.yaml
└── overlays/
    ├── dev/
    │   └── kustomization.yaml
    ├── staging/
    │   └── kustomization.yaml
    └── prod/
        └── kustomization.yaml
```

```yaml
# overlays/prod/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
bases:
- ../../base
resources:
- ingress.yaml
patchesStrategicMerge:
- deployment-patch.yaml
configMapGenerator:
- name: app-config
  literals:
  - LOG_LEVEL=warn
```

### 2.3 镜像扫描 CI

```yaml
# GitHub Actions 镜像扫描
- name: Run Trivy
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: my-app:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

### 2.4 Canary 部署（Flagger）

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: web
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
    - name: request-duration
      thresholdRange:
        max: 500
```

### 2.5 蓝绿部署

```bash
# 蓝绿部署脚本
kubectl apply -f green-deployment.yaml
kubectl rollout status deployment/web-green
kubectl patch service web -p '{"spec":{"selector":{"version":"green"}}}'
kubectl delete deployment web-blue
```

## 三、注意事项与常见陷阱

1. **镜像不可变**：使用具体标签或 SHA 摘要，不要用 latest
2. **回滚策略**：确保可以快速回滚到上一个版本
3. **配置分离**：使用 Kustomize 或 Helm 管理多环境配置
4. **安全扫描**：CI 流水线中集成镜像漏洞扫描
5. **渐进发布**：使用 Canary/蓝绿降低发布风险
