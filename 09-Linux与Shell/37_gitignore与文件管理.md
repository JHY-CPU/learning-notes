# gitignore与文件管理


## 🚫 .gitignore 与文件管理


.gitignore 模式、git rm/mv、清理文件、模板推荐。


## .gitignore 基础


```
// ========== .gitignore 是什么 ==========
// 告诉 Git 哪些文件不需要跟踪
// 通常用于: 编译产物、依赖目录、环境文件、IDE 配置

// .gitignore 放在仓库根目录
// 也可以放在子目录 (只对该目录生效)
// 支持 glob 模式匹配

// ========== 模式语法 ==========
# 注释        ← 以 # 开头

*.log         # 忽略所有 .log 文件
/build        # 忽略 build 目录 (从根目录)
build/        # 忽略所有 build 目录
node_modules/ # 忽略 node_modules 目录

# 取反 (!)
*.log         # 忽略所有 .log
!important.log # 但不忽略 important.log

# 递归 vs 非递归
*.log         # 递归忽略所有 .log (任何层级)
/*.log         # 只在根目录忽略 .log

# 通配符
*.tmp         # 所有 .tmp 文件
doc/*.pdf     # doc 目录下所有 .pdf
**/temp       # 任何层级的 temp 目录/文件

# 空目录
# Git 不跟踪空目录,所以 .gitignore 也不能跟踪空目录
# 习惯在空目录放 .gitkeep 文件
```


## 常见 .gitignore 模板


```
// ========== Node.js ==========
node_modules/
dist/
.env
.env.local
*.log
npm-debug.log*
.DS_Store
coverage/
.nyc_output/

// ========== Python ==========
__pycache__/
*.py[cod]
*.so
.env
venv/
.venv/
dist/
build/
*.egg-info/
.eggs/
*.egg
.tox/
.mypy_cache/
.pytest_cache/
.coverage

// ========== Java ==========
target/
*.class
*.jar
*.war
*.ear
.idea/
*.iml
*.ipr
*.iws
.settings/
.project
.classpath
.gradle/
build/

// ========== Go ==========
*.exe
*.test
*.out
vendor/

// ========== Rust ==========
target/
Cargo.lock  # 库项目忽略,可执行项目保留

// ========== IDE & 系统 ==========
# IDE
.idea/
*.iml
.vscode/
*.swp
*.swo
*~
.project
.classpath
.settings/

# 系统
.DS_Store
Thumbs.db
Desktop.ini

# 编辑器临时文件
*.swp
*.swo
*~

// ========== 安全 (永远不要提交!) ==========
.env
.env.*
*.pem
*.key
credentials.json
service-account.json
config/secrets.yml
**/secrets/**
```


## 文件管理命令


```
// ========== git rm — 删除文件 ==========
// 从 Git 追踪中移除文件

git rm file.txt                     # 删除文件并暂存删除
git rm --cached file.txt            # 从 Git 移除但保留文件 (不再跟踪)
git rm -r dir/                      # 递归删除目录

// ========== git mv — 移动/重命名 ==========
git mv old.txt new.txt              # 重命名

// 等价于:
mv old.txt new.txt
git rm old.txt
git add new.txt

// ========== 更新 .gitignore 后的处理 ==========
// 如果文件已被跟踪,加入 .gitignore 不会自动取消跟踪
// 需要手动移除:

# 1. 加入 .gitignore
echo "*.env" >> .gitignore

# 2. 取消跟踪已存在的文件
git rm --cached .env

# 3. 提交
git commit -m "chore: stop tracking .env files"

// ========== 清理 untracked 文件 ==========
git clean -n                        # 预览要删除的文件 (dry-run)
git clean -f                        # 强制删除 untracked 文件
git clean -fd                       # 包括目录
git clean -fx                       # 包括 .gitignore 中的文件

// ========== 全局 .gitignore ==========
// 对于所有仓库都通用的忽略规则
git config --global core.excludesFile ~/.gitignore_global

// ~/.gitignore_global:
.DS_Store
Thumbs.db
*.swp
*~
```


> **Note:** 💡 .gitignore 的最佳实践: 提交前设置好,避免不小心提交敏感文件。如果文件已经被跟踪,加入 .gitignore 不会生效,需要用 git rm --cached 取消跟踪。GitHub 的 gitignore 模板仓库 (github.com/github/gitignore) 提供了各语言的参考模板。


## 文件属性与权限


```
// ========== Git 文件状态 ==========
// Untracked:   未跟踪 (新文件)
// Tracked:
//   Unmodified: 未修改
//   Modified:   已修改
//   Staged:     已暂存

// 文件状态转换:
// Untracked → git add → Staged → commit → Unmodified → 修改 → Modified → git add → Staged

// ========== git ls-files ==========
git ls-files                        # 列出所有被跟踪的文件
git ls-files -s                     # 显示暂存区状态 (hash/模式/序号)
git ls-files --others               # untracked 文件
git ls-files --ignored              # 被 .gitignore 忽略的文件

// ========== 文件权限 ==========
// Git 只跟踪两种权限:
// 100644 → 普通文件 (rw-r--r--)
// 100755 → 可执行文件 (rwxr-xr-x)

git add --chmod=+x script.sh        # 添加为可执行
git update-index --chmod=+x script.sh  # 修改已跟踪文件为可执行

// 查看权限:
git ls-files -s script.sh
// 100755 7d30d50... 0 script.sh  ← 可执行
// 100644 7d30d50... 0 script.sh  ← 普通文件
```


## 练习


<!-- Converted from: 37_gitignore与文件管理.html -->
