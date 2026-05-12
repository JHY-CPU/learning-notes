# Git 工作流

## 一、概念说明

Vue 项目使用 Git 进行版本控制。合理的 `.gitignore` 配置、分支策略和提交规范能提升团队协作效率。Vue 项目需要忽略 `node_modules/`、`dist/` 等构建产物。

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

### 2.3 提交信息规范（Conventional Commits）

```bash
# 格式: <type>(<scope>): <description>
git commit -m "feat(auth): 添加登录页面"
git commit -m "fix(router): 修复路由跳转 404"
git commit -m "docs: 更新 README 安装说明"
git commit -m "style: 统一代码缩进风格"
git commit -m "refactor(store): 重构用户状态管理"
git commit -m "test(utils): 添加工具函数单元测试"
git commit -m "chore: 升级 Vite 到 5.x"
```

提交类型说明：

| type | 说明 |
|------|------|
| feat | 新功能 |
| fix | 修复 Bug |
| docs | 文档更新 |
| style | 代码格式（不影响功能） |
| refactor | 重构 |
| perf | 性能优化 |
| test | 测试 |
| chore | 构建/工具变更 |

### 2.4 Husky + Commitlint 强制规范

```bash
# 安装
pnpm add -D husky @commitlint/cli @commitlint/config-conventional

# 配置 commitlint
# commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional']
}

# 初始化 husky
npx husky init

# 添加 commit-msg hook
echo 'npx --no -- commitlint --edit "$1"' > .husky/commit-msg
```

### 2.5 Git 别名提升效率

```bash
# ~/.gitconfig
[alias]
  st = status -sb
  co = checkout
  br = branch
  cm = commit -m
  lg = log --oneline --graph --all
  last = log -1 HEAD --stat
  unstage = reset HEAD --
```

## 三、常见用例

### 3.1 常用 Git 操作流程

```bash
# 日常开发
git pull --rebase origin develop    # 同步最新代码
git checkout -b feature/my-feature  # 创建功能分支
# ... 编写代码 ...
git add -A                          # 暂存所有更改
git commit -m "feat: 实现XX功能"     # 提交
git push -u origin feature/my-feature # 推送到远程
# 创建 Pull Request → Code Review → 合并
```

### 3.2 撤销操作

```bash
# 撤销最后一次提交（保留更改）
git reset --soft HEAD~1

# 撤销暂存的文件
git restore --staged file.js

# 丢弃工作区的更改
git restore file.js

# 修改最后一次提交信息
git commit --amend -m "fix: 修正提交信息"
```

## 四、注意事项与常见陷阱

- **永远不要**提交 `node_modules/` 到 Git
- `.env` 文件包含敏感信息，确保在 `.gitignore` 中
- 提交前运行 `pnpm lint` 确保代码规范
- 合并前先 `git pull --rebase` 避免无意义的 merge commit
- 使用 husky + commitlint 强制提交信息规范
- 大文件使用 Git LFS，避免仓库膨胀
- 不要将构建产物提交到主分支
