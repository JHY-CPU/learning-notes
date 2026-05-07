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

## 五、注意事项

1. **快速反馈**：MR流水线应快速完成
2. **门禁检查**：CI通过才能合并
3. **代码审查**：配置CODEOWNERS
4. **自动合并**：CI通过后自动合并
