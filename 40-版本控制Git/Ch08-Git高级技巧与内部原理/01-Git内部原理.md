# Git 内部原理

## 一、Git 对象模型

Git 是一个内容寻址文件系统，核心是四种对象类型。

### 1. Blob 对象（文件内容）

Blob 存储文件的内容，不包含文件名。

```bash
# 手动创建 blob
echo "Hello Git" | git hash-object -w --stdin
# 输出: a042356e7b4e5d35e4b5e3b1c7a8f9d2e6f4a8c0

# 查看 blob 对象
git cat-file -t a042356    # 类型: blob
git cat-file -p a042356    # 内容: Hello Git
git cat-file -s a042356    # 大小: 10

# 文件内容相同的文件，blob 哈希相同（内容寻址）
echo "Hello Git" > file1.txt
echo "Hello Git" > file2.txt
git hash-object file1.txt  # 两个文件哈希相同
git hash-object file2.txt
```

### 2. Tree 对象（目录结构）

Tree 存储目录结构，包含文件名和指向 blob 的引用。

```bash
# 查看某个提交的 tree
git ls-tree HEAD
# 100644 blob a042356...    README.md
# 040000 tree b152637...    src/
# 100644 blob c263748...    package.json

# 递归查看所有文件
git ls-tree -r HEAD

# 查看 tree 对象
git cat-file -p HEAD^{tree}
# 100644 blob a042356...    README.md
# 040000 tree b152637...    src

# 查看子目录的 tree
git cat-file -p b152637
# 100644 blob d374859...    index.js
# 100644 blob e485960...    utils.js
```

### 3. Commit 对象

Commit 存储一次提交的信息，指向一个 tree 对象。

```bash
# 查看 commit 对象
git cat-file -p HEAD
# tree b152637890abcdef...
# parent c263748fedcba...
# author Alice <alice@example.com> 1704067200 +0800
# committer Alice <alice@example.com> 1704067200 +0800
#
# feat: add user authentication

# commit 对象结构：
# tree    → 本次提交的根目录 tree
# parent  → 父提交（合并提交有多个 parent）
# author  → 代码作者
# committer → 提交者（可能与作者不同）
# 提交信息
```

### 4. Tag 对象（附注标签）

```bash
# 查看 tag 对象
git cat-file -p v1.0.0
# object a042356e7b4e...  # 指向的提交
# type commit
# tag v1.0.0
# tagger Alice <alice@example.com> 1704067200 +0800
#
# Release version 1.0.0
```

### 对象关系图

```
Tag (v1.0.0)
 │
 ▼
Commit ──► Tree (根目录)
 │           │
 │           ├── blob (README.md)
 │           ├── blob (package.json)
 │           └── tree (src/)
 │                ├── blob (index.js)
 │                └── blob (utils.js)
 ▼
Commit (parent)
 │
 ▼
... (更早的提交)
```

## 二、SHA-1 哈希计算

```bash
# Git 对象的哈希计算方式：
# "类型 大小\0内容" 的 SHA-1

# 示例：计算 blob 哈希
content="Hello Git"
size=${#content}
echo -ne "blob ${size}\0${content}" | sha1sum
# 结果: a042356e7b4e5d35e4b5e3b1c7a8f9d2e6f4a8c0

# 与 git hash-object 结果一致
echo "Hello Git" | git hash-object --stdin
```

## 三、对象存储

### 对象存储位置

```bash
# 所有对象存储在 .git/objects/ 下
ls .git/objects/
# 0a/  1b/  2c/  3d/  ...  info/  pack/

# 对象文件路径：前2位作为目录名，后38位作为文件名
# .git/objects/0a/42356e7b4e5d35e4b5e3b1c7a8f9d2e6f4a8c0

# 对象是 zlib 压缩的
```

### Pack 文件

```bash
# 查看 pack 文件
ls .git/objects/pack/
# pack-abc1234567890.idx  pack-abc1234567890.pack

# 查看 pack 内容
git verify-pack -v .git/objects/pack/pack-*.idx

# 手动打包
git gc
git repack -a -d
```

## 四、引用（Refs）

### 1. 引用类型

```bash
# 分支引用
cat .git/refs/heads/main
# a042356e7b4e5d35e4b5e3b1c7a8f9d2e6f4a8c0

# 标签引用
cat .git/refs/tags/v1.0.0

# 远程分支引用
cat .git/refs/remotes/origin/main

# HEAD 引用
cat .git/HEAD
# ref: refs/heads/main

# 符号引用（symbolic-ref）
git symbolic-ref HEAD
# refs/heads/main
```

### 2. 引用规范

```
完整引用路径：

refs/heads/main          → 本地分支
refs/tags/v1.0.0         → 标签
refs/remotes/origin/main → 远程分支
refs/stash               → 储藏

简写：
main      → refs/heads/main
v1.0.0    → refs/tags/v1.0.0
origin/main → refs/remotes/origin/main
```

### 3. packed-refs

```bash
# 当引用数量很多时，Git 会打包引用到 packed-refs 文件
cat .git/packed-refs
# a042356e7b4e5d35e4b5e3b1c7a8f9d2e6f4a8c0 refs/heads/main
# b152637890abcdef1234567890abcdef12345678 refs/heads/feature
# c263748fedcba0987654321fedcba0987654321 refs/tags/v1.0.0
```

## 五、index 文件（暂存区）

```bash
# 查看暂存区内容
git ls-files --stage
# 100644 a042356e7b4e5d35e4b5e3b1c7a8f9d2e6f4a8c0 0    README.md
# 100644 b152637890abcdef1234567890abcdef12345678 0    package.json
# 100644 c263748fedcba0987654321fedcba0987654321 0    src/index.js

# 查看暂存区中特定文件
git ls-files --stage README.md

# index 文件是二进制格式，存储每个文件的：
# - 文件模式（100644, 100755, 120000 等）
# - blob 哈希
# - stage 编号（0=正常，1=base, 2=ours, 3=theirs 在合并时）
# - 文件路径
```

## 六、底层命令与高层命令

```bash
# 高层命令（porcelain）- 用户友好
git add
git commit
git log
git merge

# 底层命令（plumbing）- 直接操作对象
git hash-object      # 创建 blob
git cat-file         # 查看对象
git ls-tree          # 查看 tree
git update-index     # 操作暂存区
git write-tree       # 从暂存区创建 tree
git commit-tree      # 创建 commit
git update-ref       # 更新引用

# 底层命令示例：手动创建提交
# 1. 创建 blob
blob_hash=$(echo "Hello" | git hash-object -w --stdin)

# 2. 更新暂存区
git update-index --add --cacheinfo 100644,$blob_hash,hello.txt

# 3. 创建 tree
tree_hash=$(git write-tree)

# 4. 创建 commit
commit_hash=$(echo "Initial commit" | git commit-tree $tree_hash)

# 5. 更新引用
git update-ref refs/heads/main $commit_hash
```

## 七、.git 目录结构详解

```
.git/
├── HEAD                  当前分支指针
├── config                仓库配置
├── description           仓库描述（用于 GitWeb）
├── index                 暂存区索引（二进制）
├── hooks/                钩子脚本
│   ├── pre-commit        提交前执行
│   ├── commit-msg        检查提交信息
│   ├── pre-push          推送前执行
│   └── ...
├── info/
│   └── exclude           本地忽略规则
├── logs/
│   ├── HEAD              HEAD 变更日志（reflog 数据）
│   └── refs/heads/       各分支的变更日志
├── objects/
│   ├── 0a/               blob/tree/commit 对象
│   ├── 1b/
│   ├── info/             对象信息
│   └── pack/             打包的对象
└── refs/
    ├── heads/            本地分支
    ├── tags/             标签
    └── remotes/          远程跟踪分支
```

## 八、Git 维护命令

```bash
# 垃圾回收（压缩对象，清理松散对象）
git gc

# 打包引用
git pack-refs --all

# 验证对象完整性
git fsck

# 清理不可达对象
git prune

# 重新打包对象
git repack -a -d

# 查看仓库大小
du -sh .git/

# 优化仓库
git gc --aggressive --prune=now
```
