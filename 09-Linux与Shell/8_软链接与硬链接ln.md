# 软链接与硬链接ln


## 🔗 软链接与硬链接 (ln)


ln 命令、软链接 vs 硬链接、inode 概念、链接计数、实际应用场景。


## 硬链接 (Hard Link)


硬链接是文件系统中多个文件名指向同一个 inode (数据块) 的机制。


```
// ========== 硬链接原理 ==========
// Linux 中,文件 = 文件名 + inode (元数据+数据块)
// 硬链接 = 多个文件名指向同一个 inode
//
// $ echo "Hello" > original.txt
// $ ln original.txt hardlink.txt
// $ ls -li                 # -i 显示 inode 号
// 123456 -rw-r--r-- 2 alice alice 6 Jan 15 10:00 original.txt
// 123456 -rw-r--r-- 2 alice alice 6 Jan 15 10:00 hardlink.txt
//   ↑ 相同 inode!              ↑ 链接数 = 2
//
// ========== 硬链接特性 ==========
// 1. 所有硬链接完全平等 (没有"原文件"概念)
// 2. 删除任一硬链接,数据仍在 (引用计数 -1)
// 3. 引用计数为 0 时,数据才被删除
// 4. 不能跨文件系统 (不同分区不行)
// 5. 不能对目录创建硬链接 (除 . 和 ..)
// 6. 不占用额外磁盘空间 (只占一个目录条目)

// ========== 硬链接使用场景 ==========
// 备份/快照:
//   $ ln data.txt backup/data-2024-01-15.txt
//   多个时间点的"快照"共享数据,节省空间
//
// 版本管理:
//   不同版本引用相同的数据文件
//   直到某个版本修改内容时才复制 (写时复制)
```


## 软链接 (Symbolic Link / Symlink)


软链接是一个特殊的文件,内容是指向另一个文件或目录的路径指针。


```
// ========== 软链接原理 ==========
// 软链接 = 存储目标路径的特殊文件
// 类似 Windows 的"快捷方式"
//
// $ echo "Hello" > original.txt
// $ ln -s original.txt symlink.txt
// $ ls -li
// 123456 -rw-r--r-- 1 alice alice 6 Jan 15 10:00 original.txt
// 123457 lrwxrwxrwx 1 alice alice 12 Jan 15 10:01 symlink.txt -> original.txt
//   ↑ 不同 inode        ↑ 类型 l          ↑ 指向目标
//
// ========== 软链接特性 ==========
// 1. 有自己的 inode,是独立文件
// 2. 内容 = 目标文件的路径 (文本)
// 3. 删除原文件 → 软链接"断开" (dangling link)
// 4. 可以跨文件系统
// 5. 可以链接目录
// 6. 占用极少量磁盘 (仅存储路径字符串)
// 7. 权限始终为 lrwxrwxrwx,实际权限看目标文件

// ========== 符号链接类型 ==========
// 绝对路径:
//   $ ln -s /home/alice/docs/manual.pdf manual.pdf
//   在别的目录也能访问
//
// 相对路径:
//   $ ln -s ../data/config.yml config.yml
//   移动目录结构时不会断开
```


## 对比与应用


```
// ========== 硬链接 vs 软链接 ==========
//
// 特性         硬链接              软链接
// ─────────────────────────────────────────
// inode        相同                不同
// 类型         - (普通文件)         l (符号链接)
// 跨文件系统    ❌                  ✅
// 目录          ❌                  ✅
// 删除原文件   仍可访问            断开 (No such file)
// 磁盘占用     极小 (目录条目)      路径字符串大小
// 性能         直接访问            额外路径解析
// 管理方式     自动                需要手动维护

// ========== 实际应用 ==========
//
// 版本切换:
//   $ ln -sf /usr/local/node-v18/ /usr/local/node
//   $ ln -sf /usr/local/node-v20/ /usr/local/node
//   更改软链接即可切换 Node 版本
//
// 日志轮转:
//   app.log → app-2024-01-15.log (软链接)
//   每天生成新日志文件,更新软链接
//
// 共享库版本:
//   libc.so.6 → libc-2.35.so
//   多个版本号指向具体版本
//
// Web 部署:
//   $ ln -sfn /var/www/releases/v123/ /var/www/current
//   部署新版本时只需改链接,零停机

// ========== 链接管理 ==========
// 查看链接目标:
//   $ readlink symlink.txt       # 显示目标
//   $ stat symlink.txt           # 详细信息
//
// 更新链接:
//   $ ln -sf new_target link     # -f 覆盖已存在的链接
//
// 查找损坏链接:
//   $ find . -type l ! -exec test -e {} \; -print
//
// 修复链接:
//   删除旧的,重新创建
```


> **Note:** 💡 ln -sf 是部署中最常用的技巧——新版本代码准备好后,只需将 current 软链接指向新版本目录,实现原子切换,零停机部署。


## inode 详解


```
// ========== inode 概念 ==========
// inode (索引节点) 是文件系统中存储文件元数据的结构
//
// inode 包含:
//   - 文件类型和权限
//   - 所有者 (UID/GID)
//   - 文件大小
//   - 时间戳 (atime/ctime/mtime)
//   - 数据块指针
//   - 链接计数 (有多少文件名指向此 inode)
//
// 不包含文件名! (文件名在目录条目中)
//
// 查看 inode 信息:
//   $ stat file.txt
//   $ ls -li file.txt

// ========== 查看链接数 ==========
// $ ls -la
// drwxr-xr-x 3 alice alice 4096 Jan 15 12:00 .
// drwxr-xr-x 5 alice alice 4096 Jan 15 10:00 ..
// -rw-r--r-- 2 alice alice  123 Jan 15 11:00 file.txt
//             ↑ 链接数
//
// 目录的链接数 = 2 + 子目录数
// 新建空目录的链接数 = 2 (目录本身 + .)

// ========== 查找硬链接 ==========
// 查找相同 inode 的文件:
//   $ find /path -xdev -type f -inum 123456
//   (需要知道 inode 号, 用 ls -i 查看)

// ========== 删除软链接注意 ==========
// ❌ rm symlink/    (反斜杠防止补全,会删目标目录内容)
// ✅ rm symlink     (正确)
// ✅ unlink symlink (正确,ln 的逆操作)
```


<!-- Converted from: 8_软链接与硬链接ln.html -->
