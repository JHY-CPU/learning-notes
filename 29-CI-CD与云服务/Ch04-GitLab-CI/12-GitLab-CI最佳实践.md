# GitLab CI最佳实践

## 一、流水线优化

```yaml
# 1. 使用缓存
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - node_modules/

# 2. 使用制品
artifacts:
  paths:
    - dist/
  expire_in: 1 week

# 3. 并行测试
test:
  parallel:
    matrix:
      - SUITE: [unit, integration, e2e]
```

## 二、模板复用

```yaml
# 提取公共模板
include:
  - local: '/ci/templates/build.yml'
  - local: '/ci/templates/test.yml'
  - local: '/ci/templates/deploy.yml'
```

## 三、安全实践

```yaml
# 1. 使用掩码变量
# 2. 最小权限原则
# 3. 安全扫描集成
# 4. 镜像签名验证
```

## 四、监控与通知

```yaml
# Slack通知
after_script:
  - |
    if [ $CI_JOB_STATUS == "failed" ]; then
      curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"Build failed!"}' \
        $SLACK_WEBHOOK_URL
    fi
```

## 五、Checklist

- [ ] .gitlab-ci.yml语法正确
- [ ] 缓存配置合理
- [ ] 制品保留策略
- [ ] 安全扫描启用
- [ ] 环境变量掩码
- [ ] Runner标签配置
- [ ] MR流水线配置
- [ ] 通知渠道配置
