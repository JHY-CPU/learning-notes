# GitHub与Pull Request


## 🔄 GitHub 与 Pull Request


Fork 工作流、PR 流程、Code Review、GitHub CLI、GitHub Actions 入门。


## GitHub 协作模型


```
// ========== 两种协作方式 ==========
// 1. 共享仓库 (Shared Repository)
//    协作者有仓库写权限,直接在分支上工作
//    适合: 团队内部项目
//    所有人可 push 到同一个仓库

// 2. Fork + PR (Fork and Pull Request)
//    贡献者 fork 仓库到个人账号
//    在自己 fork 上创建分支,提交,发起 PR
//    仓库维护者审核后合并
//    适合: 开源项目、外部贡献

// ========== Fork 工作流 ==========
// 1. Fork 目标仓库 (GitHub 页面点 Fork)
// 2. 克隆自己的 fork 到本地
git clone https://github.com/myuser/repo.git
cd repo
git remote add upstream https://github.com/original/repo.git

// 3. 创建功能分支
git checkout -b feature/awesome-feature

// 4. 开发、提交
git add -A
git commit -m "feat: add awesome feature"

// 5. 推送到自己的 fork
git push -u origin feature/awesome-feature

// 6. 在 GitHub 上创建 PR (Pull Request)
//    从 myuser:feature/awesome-feature → original:main

// 7. 等待 Code Review,根据反馈修改
// 8. PR 被合并后,同步本地仓库
git checkout main
git pull upstream main
git push origin main
git branch -d feature/awesome-feature
```


## Pull Request 最佳实践


```
// ========== PR 规范 ==========
// 标题: ():
// 内容: What + Why + How

// 好的 PR 模板:
## 描述
[简要描述改动内容和动机]

## 改动类型
- [ ] 新功能 (feat)
- [ ] 修复 (fix)
- [ ] 重构 (refactor)
- [ ] 文档 (docs)
- [ ] 测试 (test)

## 测试
- [ ] 单元测试
- [ ] 手动测试
- [ ] 测试通过

## 相关问题
Closes #123

// ========== PR 原则 ==========
// 1. 小 PR (几百行),不要超大 PR
// 2. 一个 PR 只做一件事
// 3. 清晰的提交历史
// 4. 关联 Issue

// ========== Code Review 要点 ==========
// Reviewer 看什么:
// 1. 逻辑是否正确
// 2. 代码风格一致
// 3. 是否有测试
// 4. 是否有安全漏洞
// 5. 是否有性能问题
// 6. 命名是否合理

// 如何给 Review 意见:
// - 善意的语气
// - 解释为什么
// - 给出建议代码
// - 区分"必须改"和"建议改"
```


## GitHub CLI (gh)


```
// ========== 安装 gh ==========
// macOS: brew install gh
// Linux: apt install gh / dnf install gh
// Windows: winget install GitHub.cli

// 认证:
gh auth login

// ========== 常用 gh 命令 ==========
// PR 操作:
gh pr create                     # 创建 PR
gh pr list                       # 列出 PR
gh pr view 123                   # 查看 PR 详情
gh pr checkout 123               # 检出 PR 到本地
gh pr review 123 --approve       # 审核通过
gh pr merge 123                  # 合并 PR
gh pr close 123                  # 关闭 PR

// Issue:
gh issue create                  # 创建 Issue
gh issue list                    # 列出 Issue
gh issue view 456                # 查看 Issue

// 仓库:
gh repo clone user/repo          # 克隆仓库
gh repo fork                     # Fork 仓库
gh repo view                     # 查看仓库

// 工作流:
gh run list                      # 列出 Actions
gh run watch                     # 实时查看运行
gh run download                  # 下载 artifact

// 创建 PR 并自动填充:
gh pr create \
  --title "feat: add login" \
  --body "Implements user login with JWT" \
  --base main \
  --assignee @me \
  --label feat
```


## GitHub Actions 入门


```
// ========== GitHub Actions ==========
// CI/CD 流水线,自动执行测试/构建/部署

// 目录: .github/workflows/*.yml

// ========== 示例: CI 流水线 ==========
// .github/workflows/ci.yml

name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18, 20]

    steps:
    - uses: actions/checkout@v4

    - uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}

    - run: npm ci
    - run: npm test
    - run: npm run lint

// ========== Actions 市场 ==========
// actions/checkout        - 检出代码
// actions/setup-node      - 设置 Node.js
// actions/setup-python    - 设置 Python
// actions/cache           - 缓存依赖
// actions/upload-artifact - 上传构建产物
// docker/login-action     - Docker 登录
// docker/build-push-action - 构建并推送镜像
```


> **Note:** 💡 PR 工作流是 GitHub 协作的核心——fork → branch → commit → push → PR → review → merge。Good PR = 小改动 + 清晰描述 + 通过 CI。gh CLI 可以完全用命令行管理 PR,不必打开浏览器。


## 练习


<!-- Converted from: 41_GitHub与PullRequest.html -->
