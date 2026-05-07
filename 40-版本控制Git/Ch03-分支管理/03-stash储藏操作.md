# stash 储藏操作

## 一、什么是 stash

`git stash` 用于临时保存当前工作区和暂存区的修改，将工作区恢复到干净状态。适用于：

- 需要紧急切换分支但当前工作未完成
- 需要拉取远程更新但本地有未提交的修改
- 实验性修改需要临时搁置

```
stash 操作示意：

工作区（有修改） ──git stash──► 储藏栈（保存修改）
                                      │
工作区（干净）   ◄──git stash pop──┘  （恢复修改）
```

## 二、基本储藏操作

### 1. 储藏当前修改

```bash
# 储藏工作区和暂存区的修改
git stash
# 等价于
git stash push

# 带描述信息（推荐）
git stash push -m "WIP: working on login page"

# 储藏时包含未跟踪文件
git stash push -u -m "WIP: including new files"
git stash push --include-untracked -m "WIP: including new files"

# 储藏时也包含被 .gitignore 忽略的文件
git stash push -a -m "WIP: including ignored files"
git stash push --all -m "WIP: including ignored files"
```

### 2. 查看储藏列表

```bash
# 查看所有储藏
git stash list
# stash@{0}: On feature-login: WIP: working on login page
# stash@{1}: On main: WIP: config changes
# stash@{2}: On develop: WIP: test refactor

# 查看储藏的统计信息
git stash list --stat

# 查看储藏的简短信息
git stash list --oneline
```

### 3. 恢复储藏

```bash
# 恢复最近一次储藏并从栈中移除
git stash pop
# 等价于
git stash pop stash@{0}

# 恢复指定储藏（不从栈中移除，可多次应用）
git stash apply
git stash apply stash@{1}

# 恢复到指定分支
git stash apply stash@{0}
```

### 4. 删除储藏

```bash
# 删除最近一次储藏
git stash drop

# 删除指定储藏
git stash drop stash@{2}

# 清空所有储藏（危险！）
git stash clear
```

## 三、高级储藏操作

### 1. 储藏暂存区（保留工作区修改）

```bash
# 只储藏暂存区的内容（保留未暂存的修改）
git stash push --staged -m "staged changes only"
git stash push -S -m "staged changes only"
```

### 2. 部分储藏

```bash
# 交互式选择要储藏的修改
git stash push -p
# 或
git stash push --patch

# 对每个区块选择：y=储藏 n=不储藏 s=分割 e=编辑
```

### 3. 从储藏创建分支

```bash
# 如果储藏应用时有冲突，可以直接创建分支
git stash branch new-branch-name
git stash branch new-branch-name stash@{1}

# 自动基于储藏时所在的提交创建新分支
# 避免冲突的最佳方式
```

### 4. 查看储藏内容

```bash
# 查看最近一次储藏的 diff
git stash show

# 查看完整的 diff
git stash show -p
git stash show --patch

# 查看指定储藏的 diff
git stash show -p stash@{1}

# 查看储藏的统计信息
git stash show --stat
```

## 四、储藏工作流示例

### 场景一：紧急切换分支

```bash
# 当前在 feature 分支有未完成工作
# 需要切换到 main 修复紧急 bug

# 1. 储藏当前工作
git stash push -m "WIP: feature search box"

# 2. 切换到 main
git switch main

# 3. 创建修复分支
git switch -c hotfix/fix-crash

# 4. 修复并提交
git add .
git commit -m "fix: resolve crash on startup"
git switch main
git merge hotfix/fix-crash

# 5. 回到 feature 分支恢复工作
git switch feature/search-box
git stash pop
```

### 场景二：拉取远程更新

```bash
# 本地有未提交的修改，需要拉取远程更新

# 方式1：储藏后拉取
git stash push -m "save before pull"
git pull origin main
git stash pop

# 方式2：使用 autostash（推荐）
git pull --rebase --autostash origin main
# 自动储藏 → 拉取 → 恢复储藏
```

### 场景三：实验性修改

```bash
# 做一些可能要撤销的实验性修改
git stash push -m "experiment: try new algorithm"

# 如果实验成功，恢复
git stash pop

# 如果实验失败，丢弃
git stash drop
```

## 五、stash 的常见问题

### 1. 储藏冲突

```bash
# 应用储藏时可能产生冲突
git stash pop
# Auto-merging file.txt
# CONFLICT (content): Merge conflict in file.txt

# 解决方式与合并冲突相同：
# 1. 编辑冲突文件
# 2. git add resolved-file.txt
# 3. 储藏已被应用（从栈中移除）
```

### 2. 忘记储藏信息

```bash
# 查看所有储藏详情来找到需要的那个
git stash list
for i in $(git stash list | awk -F: '{print $1}'); do
    echo "=== $i ==="
    git stash show --stat "$i"
    echo ""
done
```

### 3. 恢复已删除的储藏

```bash
# 储藏删除后其实还在对象数据库中，可通过 reflog 找到
git fsck --unreachable | grep commit
# 找到对应的哈希后
git stash apply <commit-hash>
```

## 六、stash 最佳实践

```
推荐做法：
✓ 使用描述性信息：git stash push -m "具体描述"
✓ 优先用 git stash branch 创建分支（避免冲突）
✓ 定期清理不需要的储藏：git stash clear
✓ 长期保存的工作用分支而非储藏

避免做法：
✗ 不要在储藏中保存太久（容易忘记内容）
✗ 不要在多个分支间混用储藏（容易混乱）
✗ 不要依赖储藏做备份（用分支更安全）
✗ 不要 git stash clear 除非确定不需要
```
