# 版本控制Git基础

## 一、概念说明

Git是**分布式版本控制系统**，用于跟踪代码变更历史、协作开发和管理项目版本。在C++项目中，Git帮助管理源代码、构建配置和文档的变更。

## 二、具体用法

### 2.1 C++项目专用.gitignore

```gitignore
# ===== C++编译产物 =====
*.o
*.obj
*.exe
*.out
*.app
*.dll
*.so
*.dylib
*.a
*.lib

# ===== 构建目录 =====
build/
cmake-build-*/
out/
x64/
x86/

# ===== IDE文件 =====
.vscode/
.idea/
*.suo
*.sdf
*.user
*.ncb
*.opensdf

# ===== CMake =====
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile

# ===== 系统文件 =====
.DS_Store
Thumbs.db
*.swp
*~
```

### 2.2 基本工作流

```bash
# 初始化仓库
git init

# 查看状态
git status

# 添加文件到暂存区
git add main.cpp
git add include/       # 添加整个目录
git add .              # 添加所有更改（谨慎使用）

# 提交
git commit -m "添加主程序文件"

# 查看提交历史
git log
git log --oneline --graph  # 简洁图形化显示

# 查看差异
git diff                  # 工作区与暂存区的差异
git diff --cached         # 暂存区与上次提交的差异
```

### 2.3 分支管理

```bash
# 查看分支
git branch              # 本地分支
git branch -a           # 包含远程分支

# 创建并切换分支
git checkout -b feature/add-sort
# 或使用新语法
git switch -c feature/add-sort

# 合并分支
git checkout main
git merge feature/add-sort

# 删除已合并的分支
git branch -d feature/add-sort

# 解决冲突后
git add conflict_file.cpp
git commit -m "解决合并冲突"
```

### 2.4 .gitattributes处理换行

```gitattributes
# 统一换行符
* text=auto

# C++源文件始终使用LF
*.cpp text eol=lf
*.h   text eol=lf
*.hpp text eol=lf

# Windows批处理文件使用CRLF
*.bat text eol=crlf

# 二进制文件不处理
*.png binary
*.jpg binary
*.exe binary
```

### 2.5 远程仓库操作

```bash
# 添加远程仓库
git remote add origin https://github.com/user/cpp-project.git

# 推送
git push -u origin main     # 首次推送
git push                    # 后续推送

# 拉取
git pull origin main        # 拉取并合并
git fetch origin            # 仅拉取不合并

# 克隆
git clone https://github.com/user/cpp-project.git
```

### 2.6 有用的工作流命令

```bash
# 暂存当前修改
git stash
git stash pop              # 恢复暂存

# 修改最后一次提交
git commit --amend -m "修正提交信息"

# 撤销工作区修改
git checkout -- main.cpp

# 重置到某个提交
git reset --soft HEAD~1    # 保留修改在暂存区
git reset --hard HEAD~1    # 丢弃所有修改（危险！）

# 樱桃拣选（选择性合并提交）
git cherry-pick <commit-hash>
```

## 三、注意事项与常见陷阱

1. **不要提交编译产物**：`.o`、`.exe`、`build/`等应在`.gitignore`中排除
2. **不要提交大文件**：二进制文件、视频等大文件使用Git LFS管理
3. **提交信息要清晰**：使用祈使语气，如"修复内存泄漏"而非"修复了"
4. **频繁提交**：小步提交比大块提交更容易追踪和回滚
5. **不要force push到共享分支**：`git push --force`会覆盖他人工作，仅在个人分支使用
