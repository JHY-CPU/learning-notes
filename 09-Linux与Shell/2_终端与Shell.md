# 终端与Shell


## 🖥️ 终端与 Shell


终端模拟器、Shell 类型 (Bash/Zsh/Fish)、提示符、快捷键、Shell 配置文件。


## 终端 vs Shell


终端 (Terminal) 是窗口,Shell 是命令解释器——终端提供界面,Shell 执行命令。


```
// ========== 终端模拟器 ==========
// Windows: Windows Terminal (推荐), PowerShell
// macOS:   Terminal.app, iTerm2 (推荐)
// Linux:   GNOME Terminal, Konsole, Alacritty, Kitty
//
// ========== 常见 Shell ==========
// Bash:      默认 Shell, 几乎所有 Linux 都有
// Zsh:       功能更强, 插件丰富 (Oh My Zsh)
// Fish:      开箱即用, 智能提示, 语法高亮
// sh:        最基础 (POSIX 标准)
// PowerShell: Windows 的 Shell (也支持 Linux)
//
// 当前 Shell 查看:
//   $ echo $SHELL      # 输出: /bin/bash 或 /bin/zsh
//   $ ps -p $$         # 查看当前进程

// ========== Shell 提示符 ==========
// 默认提示符格式:
//   user@hostname:directory$
//
//   alice@server:~$ _
//   ──┬── ──┬── ─┬─ ┬
//    用户  主机  目录 提示符 ($=普通用户, #=root)
//
// 提示符由 PS1 环境变量定义:
//   $ echo $PS1
//   \[\\e[32m\]\\u@\\h:\\w\\$\[\\e[0m\]
//
// 常见转义:
//   \u 用户名 | \h 主机名 | \w 当前目录
//   \d 日期   | \t 时间   | \$ 提示符 (#/$)
//   \\e[32m 绿色 | \\e[0m 重置颜色
```


## Bash 基础


```
// ========== 基本操作 ==========
// Tab 补全:       按 Tab 自动补全命令/文件/路径
// 历史搜索:       Ctrl+R 搜索历史命令
// 清屏:           Ctrl+L 或 clear
// 取消当前命令:   Ctrl+C
// 暂停命令:       Ctrl+Z (然后 bg/fg)
// 退出 Shell:     exit 或 Ctrl+D
// 查看历史:       history
// 重复上次命令:   !!
// 重复第 N 条:    !123
// 上次参数:       !$

// ========== 光标快捷键 (Emacs 模式) ==========
// Ctrl+A   行首
// Ctrl+E   行尾
// Ctrl+←/→ 单词跳转 (Alt+B/F)
// Ctrl+U   删除到行首
// Ctrl+K   删除到行尾
// Ctrl+W   删除上一个单词
// Ctrl+Y   粘贴删除的内容
// Ctrl+T   交换两个字符

// ========== 通配符 (Globbing) ==========
// *        匹配任意字符 (0 或多个)
// ?        匹配单个字符
// [abc]    匹配 a/b/c 中一个
// [!abc]   匹配非 a/b/c 的字符
// {a,b,c}  花括号展开
//
// 示例:
//   ls *.txt          所有 .txt 文件
//   ls file?.txt      file1.txt, file2.txt...
//   ls [a-z]*.txt     a-z 开头的 txt
//   mkdir {2023,2024}  创建 2023 和 2024 目录
//   echo {1..10}      输出 1 2 3...10

// ========== 别名 ==========
// $ alias ll='ls -lh'
// $ alias la='ls -A'
// $ alias gs='git status'
//
// 查看所有别名:
//   $ alias
// 永久保存: 写入 ~/.bashrc 或 ~/.zshrc
```


## Shell 配置文件


```
// ========== Bash 配置文件加载顺序 ==========
//
// 登录 Shell (ssh/终端登录):
//   /etc/profile → ~/.bash_profile → ~/.bashrc
//
// 非登录 Shell (打开终端):
//   /etc/bash.bashrc → ~/.bashrc
//
// ========== 各配置文件用途 ==========
// /etc/profile:     系统级全局配置 (对所有用户)
// /etc/bash.bashrc: 系统级 bash 配置
// ~/.bash_profile:  用户登录配置 (环境变量)
// ~/.bashrc:        用户交互式 Shell 配置 (别名/函数)
// ~/.bash_logout:   退出 Shell 时执行
//
// Zsh 对应:
//   .zshrc (主要), .zprofile, .zshenv

// ========== 常见配置项 ==========
// # 设置 PATH
// export PATH=$HOME/bin:$PATH
//
// # 别名
// alias ll='ls -lh'
// alias gs='git status'
// alias dc='docker compose'
//
// # 提示符
// export PS1='\u@\h:\w\$ '
//
// # 历史设置
// export HISTSIZE=10000
// export HISTFILESIZE=20000
// export HISTCONTROL=ignoreboth:erasedups
//
// # 默认编辑器
// export EDITOR=vim
// export VISUAL=vim

// ========== 让配置生效 ==========
// $ source ~/.bashrc    # 重新加载
// $ . ~/.bashrc         # source 的缩写
// $ exec bash           # 重新启动 Shell
```


> **Note:** ⚡ 善用别名和 Shell 配置能极大提升效率。推荐: alias dc='docker compose', alias gs='git status', 加上 Ctrl+R 历史搜索,日常操作效率翻倍。


<!-- Converted from: 2_终端与Shell.html -->
