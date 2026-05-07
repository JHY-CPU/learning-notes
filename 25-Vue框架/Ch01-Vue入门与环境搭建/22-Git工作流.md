# Git 工作流

## 一、概念说明

Vue 项目使用 Git 进行版本控制。合理的 `.gitignore` 配置、分支策略和提交规范能提升团队协作效率。Vue 项目需要忽略 `node_modules/`、`dist/` 等构建产物。

```vue
<script setup>
// 提交规范示例（Conventional Commits）
// feat: 新功能
// fix: 修复 Bug
// docs: 文档更新
// style: 代码格式（不影响功能）
// refactor: 重构
// test: 测试
// chore: 构建/工具变更
</script>
```

## 二、具体用法

### 2.1 .gitignore 配置

```gitignore
# 依赖
node_modules/
.pnpm-store/

# 构建产物
dist/
.output/

# 环境变量（可能包含密钥）
.env.local
.env.*.local

# 编辑器
.vscode/
.idea/

# 系统
.DS_Store
Thumbs.db

# 日志
*.log
npm-debug.log*
```

### 2.2 分支策略

```
main        → 生产分支（稳定版本）
develop     → 开发分支
feature/*   → 功能分支
bugfix/*    → 修复分支
release/*   → 发布分支
hotfix/*    → 紧急修复
```

```bash
# 功能开发流程
git checkout develop
git checkout -b feature/user-auth
# 开发完成后
git add .
git commit -m "feat: 添加用户认证功能"
git checkout develop
git merge feature/user-auth
```

### 2.3 提交信息规范

```bash
# 格式: <type>(<scope>): <description>
git commit -m "feat(auth): 添加登录页面"
git commit -m "fix(router): 修复路由跳转 404"
git commit -m "docs: 更新 README 安装说明"
git commit -m "style: 统一代码缩进风格"
```

## 三、注意事项与常见陷阱

- **永远不要**提交 `node_modules/` 到 Git
- `.env` 文件包含敏感信息，确保在 `.gitignore` 中
- 提交前运行 `pnpm lint` 确保代码规范
- 合并前先 `git pull --rebase` 避免无意义的 merge commit
- 使用 husky + commitlint 强制提交信息规范
