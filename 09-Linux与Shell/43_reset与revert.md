# reset与revert


## ↩️ reset 与 revert


git reset (soft/mixed/hard)、git revert、恢复误操作、git reflog 急救。


## git reset — 撤销操作


```
// ========== reset 的三个模式 ==========
// 三个模式影响三个区域:
// --soft:   只移动 HEAD,不影响暂存区和工作区
// --mixed:  移动 HEAD,重置暂存区 (默认)
// --hard:   移动 HEAD,重置暂存区和工作区 ⚠️ 危险

// 示意图:
//              HEAD   Staging   Working
// --soft        ✓      ✗         ✗
// --mixed       ✓      ✓         ✗
// --hard        ✓      ✓         ✓

// ========== --soft ==========
// 撤销 commit,但保留修改在暂存区
// 常用于: 撤销最近的 commit,重新提交

git reset --soft HEAD~1         # 撤销最近 commit,保留内容在暂存区
# 修改文件后重新提交
git commit -m "new message"

// ========== --mixed (默认) ==========
// 撤销 commit 和暂存,但保留工作区修改
// 常用于: 撤销 add 和 commit,保留修改重新来过

git reset HEAD~1                # 撤销最近 commit 和暂存
git reset                       # 撤销所有暂存 (git add 的反向)
git reset HEAD file.txt         # 撤销 file.txt 的暂存

// ========== --hard ⚠️ ==========
// 完全撤销到指定状态,丢失未提交的修改!
// 常用于: 彻底放弃当前实验

git reset --hard HEAD~1         # 回到上一个 commit,丢弃所有修改
git reset --hard HEAD           # 丢弃所有工作区修改
git reset --hard origin/main    # 完全同步远程

// ⚠️ --hard 会丢失工作区和暂存区的所有未提交修改!
// 确认没有未保存的工作再执行

// ========== 恢复到某个 commit ==========
git log --oneline               # 找到目标 commit
git reset --hard abc123         # 恢复到指定 commit
```


## git revert — 安全撤销


```
// ========== revert 概念 ==========
// revert = 创建"反向"commit 来撤销之前的修改
// 不会修改历史 → 安全
// 适用于: 已推送的 commit

// ========== 基本用法 ==========
git revert HEAD                 # 撤销最近的 commit
git revert abc123               # 撤销指定 commit
git revert HEAD~3..HEAD         # 撤销最近 3 个 commit
git revert --no-commit abc123   # 暂存不提交

// revert 会打开编辑器让你填写 commit 信息
// 自动创建: "Revert <原 commit 信息>"

// ========== revert vs reset ==========
//         revert          reset
// 安全性   安全            不安全 (--hard)
// 历史    新增 commit     删除/修改历史
// 适用    已推送的commit  本地未推送
// 效果    生成反向操作    直接回到旧状态

// 推荐:
// 已推送的 commit → revert
// 本地未推送的   → reset
// 公共分支       → revert
// 个人分支       → reset

// ========== 实战场景 ==========
// 场景: 上线后发现 bug
git revert abc123               # 安全地撤销 commit
git push origin main            # 推送撤销 commit

// 场景: 回滚功能,但保留将来重新应用的可能
git revert -m 1 merge-commit    # 撤销 merge commit
# -m 1 表示保留主线分支
```


## git restore — 新式撤销


```
// ========== git restore (Git 2.23+) ==========
// 替代 git checkout 和 git reset 的部分功能
// 语义更清晰

// 撤销工作区修改:
git restore file.txt            # 丢弃工作区修改
git restore .                   # 丢弃所有工作区修改

// 撤销暂存:
git restore --staged file.txt   # 从暂存区移出 (同 git reset HEAD file.txt)
git restore --staged .          # 撤销所有暂存

// 同时撤销工作区和暂存:
git restore --source=HEAD --staged --worktree file.txt

// 从某个 commit 恢复文件:
git restore --source=abc123 file.txt  # 从指定 commit 恢复文件

// ========== restore vs checkout ==========
// 旧: git checkout -- file.txt
// 新: git restore file.txt
//
// 旧: git reset HEAD file.txt
// 新: git restore --staged file.txt
//
// 推荐使用 restore,语义更清晰
```


## git reflog — 急救工具


```
// ========== reflog 概念 ==========
// reflog = reference log
// 记录 HEAD 移动的历史 (本地操作日志)
// 即使 reset --hard 丢失的 commit 也能找回
// reflog 不会同步到远程,只存在本地

// ========== 基本用法 ==========
git reflog                      # 查看 HEAD 移动历史
git reflog --relative-date      # 显示相对时间
git reflog main                 # 查看 main 分支的 reflog

// 输出:
// abc123 (HEAD -> main) HEAD@{0}: commit: feat: add login
// def456 HEAD@{1}: reset: moving to HEAD~1
// 789012 HEAD@{2}: commit: fix: typo
// 345678 HEAD@{3}: rebase: checkout main

// ========== 找回丢失的 commit ==========
// 场景: git reset --hard 后想找回

# 1. 找到丢失的 commit hash
git reflog
# abc123 HEAD@{0}: reset: moving to HEAD~1
# def456 HEAD@{1}: commit: IMPORTANT WORK ← 这个!

# 2. 恢复到那个 commit
git reset --hard def456

// 或者 cherry-pick:
git cherry-pick def456

// ========== 其他用途 ==========
// 查看分支的 reflog:
git reflog feature               # 某分支的操作历史
git reflog --all                 # 所有引用的历史

// 清理 reflog:
git reflog expire --expire=30.days --all  # 清理 30 天前的记录
```


> **Note:** 💡 git reflog 是 Git 的"后悔药":即使 reset --hard 丢了 commit,也能在 reflog 里找回。但 reflog 只存本地,90 天后自动清理。已推送的 commit 用 revert,本地没推送的用 reset。Git 2.23+ 新增 restore 和 switch,推荐新项目使用。


## 练习


<!-- Converted from: 43_reset与revert.html -->
