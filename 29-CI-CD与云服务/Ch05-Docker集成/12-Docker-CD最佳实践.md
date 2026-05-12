# Docker CD最佳实践

## 一、镜像构建

```yaml
# 最佳实践
1. 使用多阶段构建
2. 利用层缓存
3. 最小化镜像大小
4. 安全扫描集成
```

## 二、镜像推送

```yaml
# 标签策略
tags: |
  type=sha,prefix=
  type=ref,event=branch
  type=semver,pattern={{version}}
```

## 三、部署流程

```yaml
# GitHub Actions部署
deploy:
  needs: build
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Deploy to production
      run: |
        ssh deploy@server << 'EOF'
          docker pull myregistry/myapp:latest
          docker-compose up -d
          docker image prune -f
        EOF
```

## 四、蓝绿部署

```yaml
# 使用Docker Compose实现
services:
  app-blue:
    image: myapp:blue
    
  app-green:
    image: myapp:green
    
  nginx:
    # 切换upstream实现蓝绿切换
```

## 五、Checklist

- [ ] 多阶段构建优化
- [ ] 镜像安全扫描
- [ ] 合理的标签策略
- [ ] 部署前测试
- [ ] 健康检查配置
- [ ] 回滚方案准备
- [ ] 监控告警配置
- [ ] 日志收集配置

## 六、GitOps部署模式

```yaml
# 使用ArgoCD进行GitOps部署
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: myapp
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/myapp-config.git
    targetRevision: main
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

## 七、完整的CD Pipeline

```yaml
name: Continuous Deployment

on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]

jobs:
  deploy-staging:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: |
          kubectl set image deployment/app \
            app=ghcr.io/${{ github.repository }}:${{ github.sha }}
          kubectl rollout status deployment/app --timeout=300s
      - name: Smoke Test
        run: |
          for i in {1..10}; do
            if curl -f https://staging.example.com/health; then
              echo "Smoke test passed"
              exit 0
            fi
            sleep 10
          done
          echo "Smoke test failed"
          exit 1

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      - name: Canary Deploy (10%)
        run: |
          kubectl apply -f k8s/canary.yaml
          sleep 60
      - name: Check Canary Metrics
        run: |
          ERROR_RATE=$(curl -s http://prometheus:9090/api/v1/query?query=rate:http_requests_total:5m{status=~"5.."} | jq '.data.result[0].value[1]')
          if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
            echo "Canary failed, rolling back"
            kubectl delete -f k8s/canary.yaml
            exit 1
          fi
      - name: Full Rollout
        run: |
          kubectl set image deployment/app \
            app=ghcr.io/${{ github.repository }}:${{ github.sha }}
          kubectl rollout status deployment/app
```

## 八、部署后监控

```yaml
# 部署后监控配置
post-deploy-monitor:
  stage: monitor
  script:
    - |
      # 监控15分钟
      for i in {1..15}; do
        echo "=== Minute $i ==="
        # 检查错误率
        ERROR_RATE=$(curl -s prometheus:9090/api/v1/query?query=rate:http_requests_total:5m{status=~"5.."} | jq '.data.result[0].value[1]')
        # 检查延迟
        LATENCY_P99=$(curl -s prometheus:9090/api/v1/query?query=histogram_quantile:0.99:rate:http_request_duration_seconds_bucket:5m | jq '.data.result[0].value[1]')

        echo "Error rate: $ERROR_RATE, Latency P99: $LATENCY_P99"

        if (( $(echo "$ERROR_RATE > 0.05" | bc -l) )); then
          echo "High error rate detected! Triggering rollback..."
          ./rollback.sh
          exit 1
        fi

        sleep 60
      done
      echo "Post-deploy monitoring completed successfully"
```

## 九、容器日志管理

```yaml
# 集中日志收集
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    labels:
      - "logging=promtail"
      - "logging_jobname=app"

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
```
