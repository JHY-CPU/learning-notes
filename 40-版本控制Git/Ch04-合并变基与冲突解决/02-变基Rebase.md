# 变基（Rebase）

## 一、什么是变基

变基（Rebase）是将一系列提交从一个基础移动到另一个基础的操作。与合并不同，变基重写提交历史，产生线性的提交记录。

```
合并 vs 变基：

合并（merge）：
main:    A ── B ── E ── F ── M (合并提交)
              \             /
feature:       C ── D ──────

变基（rebase）：
main:    A ── B ── E ── F
                          \
feature:                   C'─ D' (新的提交，内容相同但哈希不同)
```

## 二、基本变基操作

### 1. 标准变基

```bash
# 将当前分支变基到 main
git switch feature
git rebase main

# 变基到指定分支
git rebase main feature
```

变基过程：

```
变基前：
main:    A ── B ── E ── F
              \
feature:       C ── D

执行 git rebase main（在 feature 分支上）：
1. 找到 main 和 feature 的共同祖先 A
2. 取出 feature 上 A 之后的提交 C、D
3. 逐个应用到 main 的最新提交 F 之后
4. 产生新的提交 C'、D'（内容相同，哈希不同）

变基后：
main:    A ── B ── E ── F
                          \
feature:                   C' ── D'

# 此时可以切回 main 做快进合并
git switch main
git merge feature  # Fast-forward
```

### 2. 变基后合并

```bash
# 完整流程
git switch feature
git rebase main           # 变基
git switch main
git merge feature         # 快进合并（因为是线性历史）
git branch -d feature     # 清理分支
```

## 三、交互式变基

交互式变基是最强大的 Git 功能之一，可以修改提交历史。

### 1. 启动交互式变基

```bash
# 修改最近 3 个提交
git rebase -i HEAD~3

# 变基到指定提交
git rebase -i abc1234

# 变基到 main
git rebase -i main
```

### 2. 交互式变基的指令

编辑器会打开一个待修改的提交列表：

```
pick abc1234 feat: add user model
pick def5678 feat: add user controller
pick 9876543 fix: typo in user model
pick 1234567 feat: add user view
pick fedcba9 test: add user tests

# 可用指令：
# p, pick   = 保留该提交
# r, reword = 修改提交信息
# e, edit   = 暂停并修改该提交
# s, squash = 与前一个提交合并，保留两者的提交信息
# f, fixup  = 与前一个提交合并，丢弃该提交信息
# x, exec   = 执行 shell 命令
# b, break  = 暂停变基
# d, drop   = 删除该提交
```

### 3. 常用操作示例

#### 修改提交信息

```
# 将 reword 缩写为 r
pick abc1234 feat: add user model
r def5678 feat: add user controller  # 将打开编辑器修改信息
pick 9876543 fix: typo
```

#### 合并多个提交（Squash）

```
pick abc1234 feat: add user model
s def5678 feat: add user controller  # 合并到上一个提交
s 9876543 fix: typo                  # 合并到上一个提交
pick 1234567 feat: add user view
# 会打开编辑器让你写合并后的提交信息
```

#### 删除提交

```
pick abc1234 feat: add user model
d def5678 feat: unwanted commit      # 删除这个提交
pick 9876543 feat: add user view
```

#### 重新排列提交

```
pick abc1234 feat: add user model
pick 9876543 fix: typo               # 移到前面
pick def5678 feat: add user controller  # 移到后面
```

### 4. Fixup 提交

```bash
# 在日常开发中发现之前的提交有 bug，创建 fixup 提交
git commit --fixup abc1234
# 自动创建信息为 "fixup! feat: original message" 的提交

# 然后用 autosquash 变基自动合并
git rebase -i --autosquash main
# fixup 提交会自动排到对应的提交后面并标记为 fixup
```

## 四、变基冲突处理

### 1. 解决变基冲突

```bash
# 变基过程中发生冲突
git rebase main
# CONFLICT (content): Merge conflict in file.txt
# error: could not apply abc1234...
# hint: Resolve all conflicts manually, mark them as resolved with
# hint: "git add <path>", then run "git rebase --continue".

# 解决步骤：
# 1. 编辑冲突文件
vim file.txt

# 2. 标记为已解决
git add file.txt

# 3. 继续变基
git rebase --continue

# 4. 如果有更多冲突，重复以上步骤
```

### 2. 跳过有冲突的提交

```bash
# 跳过当前提交（不应用它）
git rebase --skip
```

### 3. 中止变基

```bash
# 完全中止变基，恢复到变基前状态
git rebase --abort
```

## 五、变基 vs 合并

### 对比

| 特性 | Merge | Rebase |
|------|-------|--------|
| 历史记录 | 非线性，有合并提交 | 线性，无合并提交 |
| 提交哈希 | 不变 | 全部改变（新的提交） |
| 冲突解决 | 一次性解决所有冲突 | 每个提交可能都需要解决 |
| 安全性 | 不改变现有历史 | 重写历史 |
| 适用场景 | 公共分支 | 本地/私有分支 |

### 何时用 Merge，何时用 Rebase

```
使用 Merge：
✓ 合并公共/共享分支到 main
✓ 需要保留完整的分支历史
✓ 多人协作的功能分支
✓ 不确定时（更安全）

使用 Rebase：
✓ 更新本地功能分支（同步 main 的最新修改）
✓ 清理本地提交历史（交互式 rebase）
✓ 保持线性的提交历史
✓ 个人分支的开发
```

## 六、黄金法则

> **永远不要对已经推送到远程的提交执行变基！**

```bash
# 安全的操作：变基本地未推送的提交
git rebase main              # 安全 - 这些提交还没有共享

# 危险的操作：变基已推送的提交
git push origin feature
git rebase main              # 危险！
git push --force origin feature  # 会破坏其他人的工作
```

### 原因

```
变基前（远程状态）：
main:    A ── B
              \
feature:       C ── D  (同事基于这个版本继续开发)
                        \
colleague:               E

你执行 rebase 后：
main:    A ── B
              \
feature:       C' ── D'  (C'和D'是新提交，哈希不同)
                        \
colleague:               E  (基于旧的C，与新的C'冲突！)
```

### 如果真的需要变基已推送的提交

```bash
# 使用 force-with-lease（比 --force 安全）
git push --force-with-lease origin feature

# --force-with-lease 会检查远程是否有你不知道的新提交
# 如果有（说明别人在你变基后推送了），推送会被拒绝
```

## 七、实用变基场景

### 场景一：同步最新 main

```bash
# 在 feature 分支上工作，main 有了新提交
git switch feature
git fetch origin
git rebase origin/main  # 将 feature 的提交移到 main 最新提交之后

# 或者使用 pull + rebase
git pull --rebase origin main
```

### 场景二：整理提交历史

```bash
# 在推送前整理混乱的提交
git log --oneline
# abc1234 WIP: trying something
# def5678 fix typo
# 1234567 another typo fix
# fedcba9 feat: add search

# 交互式变基整理
git rebase -i HEAD~4

# 合理整理后：
# pick fedcba9 feat: add search
# squash 1234567 another typo fix
# squash def5678 fix typo
# drop abc1234 WIP: trying something
```

### 场景三：将提交拆分

```bash
# 一个提交包含太多改动，需要拆分
git rebase -i HEAD~1

# 在编辑器中标记为 edit
edit abc1234 feat: big change

# 变基暂停，当前状态就是该提交的内容
git reset HEAD~1          # 撤销该提交但保留修改
git add part1.txt
git commit -m "feat: part 1"
git add part2.txt
git commit -m "feat: part 2"

# 继续变基
git rebase --continue
```
