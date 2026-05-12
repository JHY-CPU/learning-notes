# Merge Request流水线

## 一、MR触发配置

```yaml
# 仅MR时运行
test:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  script:
    - npm test

# MR和主分支
build:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"
```

## 二、MR流水线设置

```yaml
# Settings > Merge requests
# Pipelines must succeed - 流水线必须成功
# All discussions must be resolved - 讨论必须解决
```

## 三、MR模板

```yaml
# .gitlab/merge_request_templates/default.md
## Description
<!-- 描述变更 -->

## Changes
<!-- 变更列表 -->

## Testing
<!-- 测试说明 -->
```

## 四、MR审批

```yaml
# CODEOWNERS文件
# 指定代码审查人
*.js @frontend-team
*.py @backend-team
Dockerfile @devops-team
```

## 五、最佳实践

1. **流水线速度**：MR流水线应控制在5分钟内完成，过长会导致开发者等待和上下文切换
2. **必要检查**：在Settings中开启"Pipelines must succeed"和"All discussions must be resolved"作为合并门禁
3. **CODEOWNERS配置**：按目录指定代码审查人，确保每次MR都有专业人员审查
4. **合并策略**：使用squash merge保持主分支提交历史简洁

## 六、常见陷阱

1. **流水线卡住**：MR流水线中包含部署Job但没有审批人，导致流水线永久pending，使用`when: manual`配合`allow_failure: false`
2. **重复运行**：同时触发push和MR事件导致流水线运行两次，使用rules精确控制触发条件
3. **目标分支变化**：MR目标分支变更后流水线未重新运行，需手动触发或配置`only: merge_requests`
4. **Draft MR**：Draft状态的MR仍会触发流水线，可通过`rules`检测`$CI_MERGE_REQUEST_TITLE`跳过

## 七、MR流水线性能优化

```yaml
# 只在MR时运行必要的检查
lint:
  script: npm run lint
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# MR时跳过部署
deploy:
  script: ./deploy.sh
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      when: never
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

# 检测变更文件决定是否运行
test:frontend:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      changes:
        - frontend/**/*
        - package.json
      when: always
    - when: never
```

## 八、MR评论集成

```yaml
# MR流水线结果自动评论
mr-comment:
  stage: .post
  image: alpine:latest
  before_script:
    - apk add --no-cache curl jq
  script:
    - |
      COMMENT="## Pipeline Results

      | Check | Status |
      |-------|--------|
      | Lint  | ✅ Passed |
      | Tests | ✅ Passed |
      | Build | ✅ Passed |

      Coverage: 85%"

      curl --request POST \
        --header "PRIVATE-TOKEN: $TOKEN" \
        --header "Content-Type: application/json" \
        --data "$(jq -n --arg body "$COMMENT" '{body: $body}')" \
        "https://gitlab.com/api/v4/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  allow_failure: true
```

## 九、MR流水线完整性检查

```yaml
# 合并前检查清单
checklist:
  stage: .post
  script:
    - |
      echo "=== Merge Request Checklist ==="
      echo "Branch: $CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"
      echo "Target: $CI_MERGE_REQUEST_TARGET_BRANCH_NAME"
      echo "Title: $CI_MERGE_REQUEST_TITLE"

      # 检查提交信息
      COMMITS=$(git log --oneline origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME..HEAD)
      echo "Commits: $COMMITS"

      # 检查变更文件
      CHANGED=$(git diff --name-only origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME..HEAD)
      echo "Changed files: $CHANGED"

      # 验证是否包含测试
      if echo "$CHANGED" | grep -q "^src/"; then
        if ! echo "$CHANGED" | grep -q "test\|spec"; then
          echo "WARNING: Code changes without test updates"
        fi
      fi
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
  allow_failure: true
```

## 十、CODEOWNERS配置详解

```text
# .gitlab/CODEOWNERS

# 默认审查人
* @tech-lead

# 前端代码
/frontend/ @frontend-team
*.css @frontend-team
*.tsx @frontend-team
*.vue @frontend-team

# 后端代码
/backend/ @backend-team
*.go @backend-team

# 基础设施
/infrastructure/ @devops-team
Dockerfile @devops-team
docker-compose*.yml @devops-team
.gitlab-ci.yml @devops-team

# 安全相关
**/security/ @security-team
*.pem @security-team
*.key @security-team

# 文档
/docs/ @tech-writer
*.md @tech-writer
```
