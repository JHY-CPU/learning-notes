# 合并（Merge）

## 一、合并的基本概念

合并（Merge）是将一个分支的修改整合到另一个分支的操作。Git 中有两种合并方式：

### 1. 快进合并（Fast-Forward）

当目标分支没有新提交时，只需移动指针即可：

```
合并前：
main:    A ── B
              \
feature:       C ── D

合并后（快进）：
main:    A ── B ── C ── D
```

### 2. 三方合并（3-Way Merge）

当两个分支都有新提交时，需要创建一个新的合并提交：

```
合并前：
main:    A ── B ── E ── F
              \
feature:       C ── D

合并后（三方合并）：
main:    A ── B ── E ── F ── G (合并提交)
              \             /
feature:       C ── D ──────
```

## 二、执行合并

### 1. 基本合并

```bash
# 切换到目标分支
git switch main

# 合并功能分支
git merge feature-login

# 如果是快进合并
# Updating abc1234..def5678
# Fast-forward

# 如果是三方合并，会打开编辑器输入合并信息
# Merge made by the 'ort' strategy.
```

### 2. 指定合并信息

```bash
# 自定义合并提交信息
git merge feature-login -m "merge: integrate login feature"
```

### 3. 禁用快进合并

```bash
# 始终创建合并提交（即使可以快进）
git merge --no-ff feature-login

# 原因：保留分支历史的完整信息
# 好处：能看出某次提交属于哪个功能分支
```

### no-ff vs ff 对比

```
--no-ff（推荐用于功能分支）：
main: A ── B ── M (合并提交)
           \  /
feature:     C ── D
# 通过 git log 能看出 C、D 属于 feature 分支

--ff（默认快进）：
main: A ── B ── C ── D
# 看不出 C、D 曾经在 feature 分支上
```

### 4. 合并指定提交

```bash
# 只合并特定的提交（cherry-pick 风格）
git cherry-pick abc1234

# 合并另一个分支的某个提交
git cherry-pick feature~2  # 合并 feature 分支倒数第二个提交
```

### 5. 压缩合并（Squash Merge）

将功能分支的所有提交压缩成一个提交应用到目标分支：

```bash
# 压缩合并（不创建合并提交）
git merge --squash feature-login

# 需要手动提交
git commit -m "feat: add user login (squashed)"
```

```
合并前：
main:    A ── B
              \
feature:       C ── D ── E

--squash 合并后：
main:    A ── B ── F (C+D+E 压缩成一个提交)
              \
feature:       C ── D ── E (feature 分支不变)
```

## 三、中止合并

```bash
# 合并过程中中止（恢复到合并前状态）
git merge --abort

# 如果合并已完成，可以通过 reset 回退
git reset --hard HEAD~1
```

## 四、合并策略

### 可用的合并策略

```bash
# 指定合并策略
git merge -s <strategy> branch-name

# 常用策略：
git merge -s recursive feature    # 默认策略
git merge -s ours feature         # 保留当前分支的内容
git merge -s subtree feature      # 用于子树合并
```

### 策略选项

```bash
# 递归策略选项
git merge -s recursive -X ours feature    # 冲突时偏向当前分支
git merge -s recursive -X theirs feature  # 冲突时偏向被合并分支
git merge -s recursive -X patience feature  # 使用 patience diff 算法
git merge -s recursive -X ignore-space-change feature  # 忽略空格变化
```

## 五、查看合并信息

```bash
# 查看哪些分支已合并到当前分支
git branch --merged

# 查看哪些分支未合并
git branch --no-merged

# 查看合并提交
git log --merges

# 查看两个分支的合并基础
git merge-base main feature

# 图形化查看合并历史
git log --oneline --graph --all
```

## 六、合并的最佳实践

### 1. 合并前准备

```bash
# 确保目标分支是最新的
git switch main
git pull origin main

# 确保功能分支基于最新的 main
git switch feature-login
git rebase main  # 或 git merge main

# 解决冲突后合并
git switch main
git merge feature-login
```

### 2. 选择合并方式

```
场景推荐：

功能分支合入 main → git merge --no-ff feature
  保留分支历史，便于追溯

同步 main 到 feature → git merge main 或 git rebase main
  保持 feature 分支最新

压缩小修改 → git merge --squash feature
  保持 main 历史整洁

发布分支合入 → git merge --no-ff release/1.0
  记录发布点
```

### 3. 合并后的清理

```bash
# 合并完成后删除功能分支
git branch -d feature-login

# 删除远程分支
git push origin --delete feature-login
```
