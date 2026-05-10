# 导航命令pwd-ls-cd


## 🗺️ 导航命令 pwd/ls/cd


核心导航命令详解、ls 选项、cd 技巧、目录栈、pushd/popd。


## 三个核心导航命令


```
// ========== pwd (Print Working Directory) ==========
// 显示当前所在目录的完整路径
//
// $ pwd
// /home/alice/projects
//
// ========== ls (List Directory Contents) ==========
// 列出目录内容
//
// $ ls                    # 简洁列表
// $ ls -l                 # 详细列表 (权限/大小/时间)
// $ ls -a                 # 显示所有文件 (包括 . 开头隐藏文件)
// $ ls -la                # -l + -a 组合
// $ ls -lh                # 人性化文件大小 (K/M/G)
// $ ls -lt                # 按时间排序 (最新在前)
// $ ls -ltr               # 按时间排序 (最旧在前)
// $ ls -lS                # 按大小排序
// $ ls -d */              # 只显示目录
// $ ls -R                 # 递归显示子目录
//
// $ ls -la /etc           # 列出指定目录

// ========== cd (Change Directory) ==========
// $ cd /etc               # 进入 /etc 目录
// $ cd ..                 # 上一级目录
// $ cd ../..              # 上两级目录
// $ cd ~                  # 回家目录 (等价于 cd)
// $ cd -                  # 回到上一个目录
// $ cd /                  # 到根目录
// $ cd ~/projects         # 到用户目录下的 projects

// ========== 特殊目录引用 ==========
// .    当前目录
// ..   父目录
// ~    当前用户的家目录
// ~alice 用户 alice 的家目录
// -    上一个目录
```


## ls -l 输出详解


```
// ========== ls -l 各字段 ==========
// $ ls -l /etc/hosts
// -rw-r--r-- 1 root root 256 Jan 15 10:30 /etc/hosts
// ┬┴┬┴┬┴┬┴ ┬ ┬──┬ ┬──┬ ┬────┬──────┬───────────
// │ │ │ │  │  │    │    │     │       └── 文件名
// │ │ │ │  │  │    │    │     └── 最后修改时间
// │ │ │ │  │  │    │    └── 文件大小 (字节)
// │ │ │ │  │  │    └── 组 (group)
// │ │ │ │  │  └── 所有者 (owner)
// │ │ │ │  └── 硬链接数
// │ │ │ └── 其他用户权限 (r--)
// │ │ └──── 组权限 (r--)
// │ └────── 所有者权限 (rw-)
// └──────── 文件类型 (-)

// ========== 文件类型字符 ==========
// -  普通文件
// d  目录
// l  符号链接
// c  字符设备
// b  块设备
// s  本地套接字
// p  命名管道

// ========== 权限字符 ==========
// r  可读 (4)
// w  可写 (2)
// x  可执行 (1)
// -  无此权限
//
// 每组三个字符: rwx (所有者) rwx (组) rwx (其他)
```


## cd 进阶与目录栈


```
// ========== CDPATH ==========
// 设置 CDPATH 环境变量,cd 会自动在这些目录中搜索
//
// $ export CDPATH=~/projects:/usr/local
// $ cd my-project   # 如果 ~/projects/my-project 存在,直接跳转

// ========== 目录栈 (Directory Stack) ==========
// 使用 pushd/popd 管理多个目录的切换
//
// $ pushd /tmp            # 保存当前目录并跳转到 /tmp
// $ pushd /etc            # 保存 /tmp 并跳转到 /etc
// $ dirs                  # 显示目录栈
//    /etc /tmp ~/projects
// $ popd                  # 回到 /tmp
// $ popd                  # 回到 ~/projects

// ========== 实用技巧 ==========
// 快速创建并进入目录:
//   $ mkdir -p new/project && cd $_
//
// 返回上一级:
//   $ cd ..
//
// 回到上上级:
//   $ cd ../..
//
// 打开文件管理器:
//   $ nautilus .           # GNOME
//   $ nemo .               # Cinnamon
//   $ thunar .             # XFCE
//   $ open .               # macOS
//
// 在 Finder 中打开 (macOS):
//   $ open .
// 在资源管理器中打开 (WSL):
//   $ explorer.exe .
```


> **Note:** 💡 cd - 是最常用的技巧之一——在两个目录间快速切换。比如你在 ~/projects/backend 编辑代码,cd 到 /etc/nginx 查看配置,输入 cd - 就能立即回到代码目录。


## 模拟练习


<!-- Converted from: 3_导航命令pwd-ls-cd.html -->
