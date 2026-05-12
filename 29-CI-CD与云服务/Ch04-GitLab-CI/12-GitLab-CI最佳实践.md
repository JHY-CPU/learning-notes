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

## 六、GitLab CI高级配置

```yaml
# 全局默认值
default:
  image: alpine:latest
  retry:
    max: 2
    when:
      - runner_system_failure
      - stuck_or_timeout_failure
  tags:
    - docker

# 中断过时的Pipeline
workflow:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
    - if: $CI_COMMIT_TAG

# 全局中断旧Pipeline
interruptible: true

# 单个Job覆盖
deploy:
  interruptible: false  # 部署Job不被中断
  script: ./deploy.sh
```

## 七、性能优化清单

```yaml
# 1. 浅克隆
variables:
  GIT_DEPTH: 1

# 2. 获取最新标签（用于版本号）
variables:
  GIT_STRATEGY: fetch
  GIT_DEPTH: 0  # tags获取需要完整历史

# 3. 并行DAG执行（使用needs代替stages）
build:frontend:
  needs: []

build:backend:
  needs: []

test:frontend:
  needs: [build:frontend]

test:backend:
  needs: [build:backend]

# 4. 条件执行
lint:
  rules:
    - changes:
        - '**/*.js'
        - '**/*.ts'
      when: always
    - when: manual
      allow_failure: true

# 5. 共享Runner资源
# 合理设置并发数和资源限制
```

## 八、GitLab CI调试技巧

```yaml
# 启用调试模式
variables:
  CI_DEBUG_TRACE: "true"  # 显示所有命令

# 只运行特定Job
# 在GitLab UI中点击"Play"手动触发

# 本地调试（使用gitlab-ci-local）
# npm install -g gitlab-ci-local
# gitlab-ci-local --job test

# 验证配置文件
# 使用gitlab-ci-lint
curl --header "PRIVATE-TOKEN: $TOKEN" \
  --header "Content-Type: application/json" \
  --data '{"content": "<ci配置内容>"}' \
  "https://gitlab.com/api/v4/projects/$PROJECT_ID/ci/lint"
```

## 九、Pipeline效率评分

```bash
# 评估Pipeline效率的指标
# 1. Pipeline总时长: <10分钟
# 2. 缓存命中率: >80%
# 3. 构建失败率: <5%
# 4. 资源利用率: >70%
# 5. 队列等待时间: <30秒

# 在GitLab UI中查看:
# CI/CD > Pipelines > Analytics
# CI/CD > Jobs > 统计数据
```
