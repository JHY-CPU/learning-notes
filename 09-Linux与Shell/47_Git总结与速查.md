# Git总结与速查


## 📖 Git 总结与速查


Git 核心命令速查、思维导图、常见问题、学习资源。


## Git 思维导图


```
// ========== Git 知识体系 ==========
//
// Git
// ├── 基础
// │   ├── git init / git clone
// │   ├── git add / git commit
// │   ├── git status / git diff
// │   └── git log / git show
// ├── 分支
// │   ├── git branch / git switch
// │   ├── git merge
// │   ├── git rebase / git rebase -i
// │   └── git cherry-pick
// ├── 远程
// │   ├── git remote / git fetch
// │   ├── git pull / git push
// │   └── git pull --rebase
// ├── 撤销
// │   ├── git reset (soft/mixed/hard)
// │   ├── git revert
// │   ├── git restore / git checkout
// │   └── git reflog
// ├── 保存
// │   ├── git stash / stash pop
// │   └── git tag
// ├── 协作
// │   ├── Fork + PR
// │   ├── Gitflow / GitHub Flow
// │   └── Code Review
// └── 进阶
//     ├── git bisect
//     ├── git hooks / husky
//     ├── git submodule
//     └── .gitattributes / .gitignore
```


## 日常操作速查表


```
// ========== 场景命令速查 ==========
// 初始化      git init / git clone
// 暂存        git add  / git add -A
// 提交        git commit -m "msg" / git commit -am "msg"
// 推送        git push -u origin
// 拉取        git pull / git fetch
// 创建分支    git branch  / git switch -c
// 切换分支    git switch
// 合并        git merge
// 查看状态    git status / git status -s
// 查看历史    git log --oneline --graph --all
// 查看差异    git diff / git diff --staged

// ========== 撤销速查 ==========
// 修提交信息   git commit --amend -m "new msg"
// 漏文件      git add  && git commit --amend --no-edit
// 撤 add     git restore --staged
// 撤修改     git restore
// 撤 commit  git reset --soft HEAD~1 (保留修改)
// 撤 commit  git reset --hard HEAD~1 (删除修改) ⚠️
// 撤 推送    git revert  (安全)
// 找丢失     git reflog

// ========== 分支速查 ==========
// 查看        git branch -a
// 创建+切换   git switch -c
// 重命名      git branch -m
// 删除本地    git branch -d
// 删除远程    git push origin --delete
// 基于远程    git switch -c  origin/
// 跟踪        git branch -u origin/
```


## 常见场景速查


```
// ========== 紧急场景 ==========
// 合并冲突了怎么办?
git merge --abort                         # 放弃合并

// rebase 出问题了怎么办?
git rebase --abort                        # 放弃 rebase

// commit 写错了怎么办?
git commit --amend -m "正确信息"          # 修改最近提交

// 提交了敏感信息怎么办?
git reset --soft HEAD~1                   # 撤回提交
# 删除敏感文件
git add -A
git commit -m "fix: remove secrets"
git push --force-with-lease               # 覆盖远程

// 删了分支怎么办?
git reflog                                # 找 hash
git branch                    # 重建分支

// 切分支时工作区有修改怎么办?
git stash push -m "WIP"                   # 暂存
git switch
# 处理完切回来
git stash pop

// ========== 日常场景 ==========
// 更新 fork
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

// 功能分支保持同步
git checkout feature/xxx
git fetch origin
git rebase origin/main

// 合并多个 commit
git rebase -i HEAD~3

// 查看某个文件的修改历史
git log -p -- file.txt
git log --oneline -- file.txt
git blame file.txt

// 切换到上一个分支
git switch -
```


## Git 学习资源


```
// ========== 推荐资源 ==========
// 官方:
// - Git 官方文档: git-scm.com/doc
// - Pro Git 书籍: git-scm.com/book (免费,中文版)

// 可视化:
// - Git 可视化学习: learngitbranching.js.org
// - Git 图解: github.com/onlyliuxin/gitx-intro
// - Git 游戏: gitexercises.fracz.com

// 工具:
// - tig — 终端 Git 浏览器
// - lazygit — 终端 GUI
// - gh — GitHub CLI
// - GitKraken / Sourcetree — GUI 客户端

// GitHub:
// - GitHub Skills: skills.github.com (实操学习)
// - GitHub Docs: docs.github.com

// ========== 一句话总结 ==========
// git add: 把修改放进"候选区"
// git commit: 在"候选区"拍照存档
// git push: 把照片上传到云端
// git pull: 从云端下载别人的照片
// git branch: 在时间线上创建平行宇宙
// git merge: 合并两个平行宇宙
// git rebase: 把自己的修改移到最新的时间点
// git reset: 回到过去 (改写历史)
// git revert: 用逆向操作撤销 (安全)
// git stash: 把半成品放抽屉里
// git tag: 给历史打标记 (v1.0)
// git reflog: 所有操作的操作日志 (后悔药)
```


> **Note:** 💡 Git 学习路径: 先掌握 add/commit/push/pull (2 天) → 分支和合并 (1 周) → rebase 和交互式 rebase (2 周) → 工作流和团队协作 (1 个月)。不用一次学完所有命令,每天用到的就那几个。


## 练习


<!-- Converted from: 47_Git总结与速查.html -->
