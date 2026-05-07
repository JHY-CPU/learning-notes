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
