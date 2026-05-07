# Pull Request 与代码审查

## 一、Pull Request 概述

Pull Request（PR）是向项目贡献代码的核心机制。开发者在功能分支完成工作后，请求项目维护者将代码合并到目标分支。

```
PR 流程：

1. Fork/Clone     2. Feature Branch    3. Push & PR
   ┌─────┐           ┌─────┐              ┌─────┐
   │main │──fork──►   │main │──branch──►   │main │
   └─────┘           │     │              │     │
                     │feat │──push──►      │ PR  │
                     └─────┘              └──┬──┘
                                             │
4. Code Review    5. Approve & Merge         │
   ┌─────┐           ┌─────┐              ┌──▼──┐
   │review│──approve──│merge│◄─────────────│  CI  │
   └─────┘           └─────┘              └─────┘
```

## 二、使用 GitHub CLI

### 1. 创建 PR

```bash
# 创建 PR（交互式）
gh pr create

# 带标题和描述创建
gh pr create --title "Add user authentication" --body "## Summary
- Added login/logout endpoints
- Implemented JWT token generation
- Added password hashing

## Test plan
- [x] Unit tests pass
- [x] Manual testing on localhost"

# 指定目标分支和源分支
gh pr create --base main --head feature/auth

# 创建草稿 PR
gh pr create --draft

# 创建并自动填写模板
gh pr create --fill
```

### 2. 查看 PR

```bash
# 列出所有 PR
gh pr list

# 列出特定状态的 PR
gh pr list --state open
gh pr list --state closed
gh pr list --state merged

# 查看特定 PR 详情
gh pr view 123

# 在浏览器中打开 PR
gh pr view 123 --web

# 查看当前分支的 PR
gh pr view
```

### 3. 管理 PR

```bash
# 检出 PR 到本地（用于审查）
gh pr checkout 123

# 检出并创建本地分支
gh pr checkout 123 -b review-pr-123

# 合并 PR
gh pr merge 123

# 合并方式选择
gh pr merge 123 --merge     # 创建合并提交
gh pr merge 123 --squash    # 压缩合并
gh pr merge 123 --rebase    # 变基合并

# 关闭 PR（不合并）
gh pr close 123

# 重新打开 PR
gh pr reopen 123

# 审查 PR
gh pr review 123 --approve
gh pr review 123 --request-changes --body "需要修改密码验证逻辑"
gh pr review 123 --comment --body "LGTM!"
```

### 4. PR Diff 与检查

```bash
# 查看 PR 的差异
gh pr diff 123

# 查看 PR 的状态检查
gh pr checks 123

# 查看 PR 的提交
gh pr view 123 --json commits
```

## 三、PR 模板

### 创建 PR 模板

在仓库中创建 `.github/pull_request_template.md`：

```markdown
## 描述

简要描述这个 PR 的内容。

## 变更类型

- [ ] 新功能 (feature)
- [ ] Bug 修复 (bugfix)
- [ ] 代码重构 (refactor)
- [ ] 文档更新 (docs)
- [ ] 性能优化 (perf)
- [ ] 测试 (test)

## 改动说明

- 改动1：xxx
- 改动2：xxx

## 相关 Issue

Closes #123

## 测试

- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试通过

## 截图

如果是 UI 变更，请提供截图。

## Checklist

- [ ] 代码符合项目规范
- [ ] 已添加必要的测试
- [ ] 文档已更新
- [ ] 无合并冲突
```

### 多个 PR 模板

```
.github/
├── PULL_REQUEST_TEMPLATE/
│   ├── feature.md
│   ├── bugfix.md
│   └── hotfix.md
```

## 四、代码审查最佳实践

### 审查者指南

```
审查要点：
✓ 代码逻辑是否正确
✓ 是否有边界情况未处理
✓ 是否有安全隐患
✓ 代码风格是否一致
✓ 是否有不必要的复杂度
✓ 测试覆盖是否充分
✓ 文档是否需要更新
✓ 性能是否有问题

审查态度：
✓ 针对代码，不对人
✓ 提供具体建议而非模糊意见
✓ 肯定做得好的地方
✓ 使用 "建议：" 而非 "你应该："
```

### 提交者指南

```
提交 PR 前：
✓ 自己先审查一遍
✓ 确保 CI 通过
✓ 写清楚 PR 描述
✓ 分解大的 PR 为小 PR
✓ 响应审查意见及时
✓ 对审查意见保持开放态度
```

## 五、合并策略

### 1. 创建合并提交（Merge Commit）

```bash
gh pr merge --merge
```

```
main:    A ── B ──────── M (合并提交)
              \        /
feature:       C ── D ─┘

优点：保留完整分支历史
缺点：历史图较复杂
```

### 2. 压缩合并（Squash and Merge）

```bash
gh pr merge --squash
```

```
main:    A ── B ── S (压缩提交，包含 C+D)
              \
feature:       C ── D

优点：main 历史干净
缺点：丢失功能分支的详细提交历史
```

### 3. 变基合并（Rebase and Merge）

```bash
gh pr merge --rebase
```

```
main:    A ── B ── C' ── D' (线性历史)
              \
feature:       C ── D

优点：线性历史，保留每个提交
缺点：提交哈希改变
```

### 如何选择

```
大型功能/多提交 → Squash and Merge（保持 main 整洁）
小修改/单提交  → Rebase and Merge（线性历史）
需要完整历史  → Merge Commit（保留分支信息）
```

## 六、Branch Protection Rules

### 常见保护配置

```yaml
# GitHub 网页设置，非配置文件
main 分支保护规则:
  - 要求 Pull Request 审查（至少 1 人批准）
  - 要求状态检查通过
    - CI/build
    - CI/test
    - CI/lint
  - 要求分支为最新
  - 禁止强制推送
  - 限制谁可以推送
    - 仅管理员或特定团队
```

### CODEOWNERS 文件

在 `.github/CODEOWNERS` 中指定代码负责人：

```gitignore
# 默认所有人
* @team-leads

# 前端代码负责人
/src/frontend/ @frontend-team @alice

# 后端代码负责人
/src/backend/ @backend-team @bob

# CI/CD 配置
/.github/ @devops-team

# 数据库迁移
/src/migrations/ @dba-team @charlie
```

## 七、自动化审查

### GitHub Actions 自动审查

```yaml
# .github/workflows/pr-check.yml
name: PR Check

on:
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run lint

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test

  size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: preactjs/compressed-size-action@v2
        with:
          pattern: 'dist/**'
```

### PR 自动标签

```yaml
# .github/labeler.yml
'frontend':
  - src/frontend/**/*

'backend':
  - src/backend/**/*

'docs':
  - '**/*.md'

'test':
  - '**/*.test.*'
```
