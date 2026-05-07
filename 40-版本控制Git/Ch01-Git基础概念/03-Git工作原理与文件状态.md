# Git 工作原理与文件状态

## 一、Git 仓库结构

Git 仓库由三个主要部分组成：

```
项目根目录/
├── .git/                 ← Git 仓库数据目录
│   ├── HEAD              ← 指向当前分支
│   ├── config            ← 仓库配置
│   ├── description       ← 仓库描述
│   ├── hooks/            ← 钩子脚本
│   ├── info/             ← 仓库信息
│   │   └── exclude       ← 本地忽略规则
│   ├── objects/          ← 所有数据对象（commits, trees, blobs）
│   │   ├── 0a/
│   │   ├── 1b/
│   │   └── ...
│   ├── refs/             ← 引用（branches, tags）
│   │   ├── heads/        ← 本地分支
│   │   └── tags/         ← 标签
│   └── index             ← 暂存区索引
├── src/                  ← 工作区文件
├── README.md
└── ...
```

## 二、文件的四种状态

Git 中每个文件都处于以下四种状态之一：

```
                        git add          git commit
  ┌──────────┐        ┌────────────┐        ┌──────────┐
  │ Untracked │ ─────► │  Staged    │ ─────► │ Committed│
  │ (未跟踪)  │        │ (已暂存)    │        │ (已提交)  │
  └──────────┘        └────────────┘        └──────────┘
       ▲                    │
       │                    │ 修改文件后
       │                    ▼
       │              ┌────────────┐
       │              │  Modified  │
       │              │ (已修改)    │
       │              └────────────┘
       │                    │
       └────── git rm ──────┘
```

### 状态详解

| 状态 | 说明 | `git status` 显示 |
|------|------|------------------|
| **Untracked** | 新文件，尚未被 Git 跟踪 | 红色 "Untracked files" |
| **Staged** | 文件已加入暂存区，等待提交 | 绿色 "Changes to be committed" |
| **Modified** | 已跟踪的文件被修改，但未暂存 | 红色 "Changes not staged for commit" |
| **Committed** | 文件已安全存储在本地仓库 | 不显示（已是最新的） |

还有一种特殊状态：

| 状态 | 说明 |
|------|------|
| **Deleted** | 文件被删除，需要 `git rm` 或 `git add` 来确认 |
| **Renamed** | 文件被重命名（Git 检测到后显示为 renamed） |
| **Ignored** | 文件被 `.gitignore` 忽略，不显示在 `git status` 中 |

## 三、查看文件状态

```bash
# 查看完整状态（详细）
git status

# 查看简洁状态（每个文件一行）
git status -s
# 输出格式：
# ?? = 未跟踪
# A  = 新增到暂存区
# M  = 已修改（已暂存）
#  M = 已修改（未暂存）
# MM = 已修改（暂存后又修改）
# D  = 已删除（已暂存）
#  D = 已删除（未暂存）
# R  = 重命名
# ?? = 未跟踪
```

### 状态输出示例

```bash
$ git status -s
 M README.md          # 已修改，未暂存
A  src/app.js         # 新增，已暂存
MM src/utils.js       # 已暂存后又修改
D  old-config.json    # 删除，已暂存
?? new-file.txt       # 未跟踪
```

## 四、Git 与文件系统的核心区别

### 1. Git 记录快照，不记录差异

传统版本控制系统（如 SVN）记录文件的变化量（delta）。而 Git 每次提交都记录项目的完整快照。

```
# SVN 方式（记录变化）
v1: file.txt = "Hello"
v2: file.txt = "Hello World"     ← 记录 "World" 被添加
v3: file.txt = "Hello World!"    ← 记录 "!" 被添加

# Git 方式（记录快照）
commit1: file.txt = "Hello"              ← 完整内容
commit2: file.txt = "Hello World"        ← 完整内容
commit3: file.txt = "Hello World!"       ← 完整内容
```

Git 通过优化（如压缩、对象引用）来避免存储冗余数据，所以仓库体积不会成倍增长。

### 2. Git 几乎所有操作都可本地执行

```
不需要网络的操作（本地完成）:
├── git add        ✓ 暂存文件
├── git commit     ✓ 提交更改
├── git branch     ✓ 创建分支
├── git checkout   ✓ 切换分支
├── git log        ✓ 查看历史
├── git diff       ✓ 比较差异
├── git merge      ✓ 合并分支
└── git tag        ✓ 创建标签

需要网络的操作（远程操作）:
├── git clone      ✗
├── git fetch      ✗
├── git pull       ✗
├── git push       ✗
└── git remote     ✗
```

### 3. Git 保证数据完整性

Git 使用 SHA-1 哈希算法对所有内容进行校验。每个对象（提交、树、文件）都有唯一的哈希值。

```bash
# 查看提交的 SHA-1 哈希
git log --oneline
# a1b2c3d (HEAD -> main) 添加用户模块
# e4f5g6h 初始化项目
# 其中 a1b2c3d 就是 SHA-1 哈希的前7位

# 查看完整哈希
git log --format=full
# commit a1b2c3d4e5f6g7h8i9j0...（40位十六进制字符）
```

如果文件被篡改哪怕一个字节，Git 都能检测到。

### 4. Git 只添加数据

Git 中大多数操作只是添加数据。一旦提交快照，就几乎不可丢失。即使执行了删除操作，通常也能恢复。

```bash
# 即使删除了分支，只要知道提交哈希就能恢复
git reflog                    # 查看所有操作记录
git branch recover-branch <commit-hash>  # 恢复分支
```

## 五、`.gitignore` 文件

### 1. 基本规则

```gitignore
# 注释（以 # 开头）

# 忽略特定文件
config.secret
.env

# 忽略特定后缀的文件
*.log
*.tmp
*.swp

# 忽略特定目录
node_modules/
dist/
build/
.DS_Store

# 使用 ! 取消忽略（在已忽略的模式中排除特定文件）
!important.log

# 忽略所有目录下的特定文件
**/*.log

# 只忽略根目录下的文件（相对于 .gitignore 所在位置）
/build/

# 忽略某个目录下所有文件但保留目录本身
build/*
!build/.gitkeep
```

### 2. 常用模板

```gitignore
# ===== Node.js =====
node_modules/
npm-debug.log*
.env
.env.local

# ===== Python =====
__pycache__/
*.pyc
*.pyo
venv/
.venv/

# ===== Java =====
*.class
*.jar
target/
.gradle/

# ===== IDE =====
.idea/
.vscode/
*.swp
*.swo
*~

# ===== OS =====
.DS_Store
Thumbs.db
desktop.ini

# ===== Build Output =====
dist/
build/
out/
```

### 3. 全局 `.gitignore`

```bash
# 创建全局忽略文件
git config --global core.excludesfile ~/.gitignore_global

# 编辑全局忽略文件
# 添加所有项目都需要忽略的文件规则
```

### 4. 忽略已跟踪的文件

如果文件已经被 Git 跟踪，即使加入 `.gitignore` 也不会生效，需要先从跟踪中移除：

```bash
# 停止跟踪但保留本地文件
git rm --cached filename

# 停止跟踪整个目录
git rm --cached -r directory/

# 然后将文件名加入 .gitignore
echo "filename" >> .gitignore
git add .gitignore
git commit -m "chore: stop tracking config file"
```
