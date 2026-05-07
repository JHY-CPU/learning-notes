# Git 安装与初始配置

## 一、安装 Git

### Windows

1. 从 [git-scm.com](https://git-scm.com/download/win) 下载安装包
2. 运行安装程序，推荐选择以下配置：
   - 默认编辑器：选择 VS Code 或 Vim
   - PATH 环境：选择 "Git from the command line and also from 3rd-party software"
   - 行尾符号转换：选择 "Checkout Windows-style, commit Unix-style line endings"
3. 安装完成后打开终端验证：

```bash
git --version
# 输出: git version 2.43.0.windows.1
```

### macOS

```bash
# 方式一：通过 Homebrew 安装（推荐）
brew install git

# 方式二：安装 Xcode Command Line Tools（自带 Git）
xcode-select --install

# 验证安装
git --version
```

### Linux

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install git

# CentOS / Fedora / RHEL
sudo dnf install git

# Arch Linux
sudo pacman -S git

# 验证安装
git --version
```

## 二、初次配置

### 1. 设置用户身份

安装 Git 后第一件事就是设置用户名和邮箱，这会附加到每次提交中。

```bash
# 设置全局用户名（所有仓库生效）
git config --global user.name "你的名字"

# 设置全局邮箱
git config --global user.email "your_email@example.com"

# 为特定仓库设置不同的身份（在仓库目录内执行）
git config user.name "项目专用名"
git config user.email "project@example.com"
```

### 2. 配置级别

Git 配置有三个级别，优先级从高到低：

| 级别 | 文件位置 | 命令参数 | 作用范围 |
|------|---------|---------|---------|
| 系统级 | `/etc/gitconfig` | `--system` | 所有用户 |
| 全局级 | `~/.gitconfig` | `--global` | 当前用户 |
| 仓库级 | `.git/config` | （无参数） | 当前仓库 |

```bash
# 查看系统级配置
git config --system --list

# 查看全局级配置
git config --global --list

# 查看当前仓库配置
git config --list

# 查看所有配置及其来源
git config --list --show-origin
```

### 3. 设置默认编辑器

```bash
# 设置 VS Code 为默认编辑器
git config --global core.editor "code --wait"

# 设置 Vim
git config --global core.editor "vim"

# 设置 Nano
git config --global core.editor "nano"

# 设置 Notepad++（Windows）
git config --global core.editor "'C:/Program Files/Notepad++/notepad++.exe' -multiInst -notabbar -nosession -noPlugin"
```

### 4. 设置默认分支名

```bash
# 将默认分支名从 master 改为 main（Git 2.28+）
git config --global init.defaultBranch main
```

## 三、常用配置选项

### 1. 别名（Alias）

为常用命令设置简写，提高效率：

```bash
# 常用别名设置
git config --global alias.st status        # git st → git status
git config --global alias.co checkout      # git co → git checkout
git config --global alias.br branch        # git br → git branch
git config --global alias.ci commit        # git ci → git commit
git config --global alias.unstage 'reset HEAD --'  # git unstage
git config --global alias.last 'log -1 HEAD'       # git last 查看最后一次提交
git config --global alias.lg "log --oneline --graph --decorate --all"  # 美化日志
git config --global alias.amend "commit --amend --no-edit"  # 修改上次提交

# 高级别名 - 彩色日志图
git config --global alias.tree "log --oneline --graph --all --decorate"
```

设置后配置文件中会添加：

```ini
[alias]
    st = status
    co = checkout
    br = branch
    ci = commit
    lg = log --oneline --graph --decorate --all
```

### 2. 颜色输出

```bash
git config --global color.ui auto
```

### 3. 换行符处理

```bash
# Windows：检出时转换为 CRLF，提交时转换为 LF
git config --global core.autocrlf true

# macOS/Linux：检出时不转换，提交时转换为 LF
git config --global core.autocrlf input

# 不做任何转换（适合二进制文件项目）
git config --global core.autocrlf false
```

### 4. 忽略文件权限变化

```bash
# Windows 上建议开启（文件系统可能无意修改权限）
git config --global core.fileMode false
```

### 5. 凭证存储

```bash
# 缓存凭证（默认15分钟）
git config --global credential.helper cache

# 永久存储凭证（明文保存，注意安全）
git config --global credential.helper store

# 使用操作系统密钥链（推荐）
# Windows
git config --global credential.helper manager
# macOS
git config --global credential.helper osxkeychain
```

## 四、查看配置

```bash
# 查看所有配置
git config --list

# 查看特定配置项
git config user.name
git config user.email

# 查看配置及其来源文件
git config --list --show-origin

# 查看 Git 内置帮助
git help config
# 或
git config --help
```

## 五、SSH 密钥配置

### 1. 生成 SSH 密钥

```bash
# 生成 SSH 密钥对（推荐 Ed25519）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 如果不支持 Ed25519，使用 RSA
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 按提示选择密钥保存位置（默认 ~/.ssh/id_ed25519）
# 设置密码短语（可选，建议设置）
```

### 2. 添加 SSH 密钥到 ssh-agent

```bash
# 启动 ssh-agent
eval "$(ssh-agent -s)"

# 添加私钥
ssh-add ~/.ssh/id_ed25519
```

### 3. 添加公钥到 GitHub/GitLab

```bash
# 复制公钥内容
# Windows
clip < ~/.ssh/id_ed25519.pub
# macOS
pbcopy < ~/.ssh/id_ed25519.pub
# Linux
cat ~/.ssh/id_ed25519.pub
```

然后在 GitHub → Settings → SSH and GPG keys → New SSH key 中粘贴公钥。

### 4. 测试连接

```bash
ssh -T git@github.com
# 成功输出: Hi username! You've successfully authenticated...
```

## 六、获取帮助

```bash
# 查看所有 Git 命令
git help

# 查看特定命令的帮助（HTML 格式）
git help commit
git help config

# 终端内简洁帮助
git commit -h
git commit --help
```
