# 查看文件cat-less-head-tail


## 👁️ 查看文件 cat/less/head/tail


文件查看命令详解、分页浏览、实时监控、文件合并、二进制查看。


## cat — 连接与显示文件


```
// ========== cat (Concatenate) ==========
// 最基础的文件查看命令
//
// $ cat file.txt                   # 显示文件内容
// $ cat -n file.txt                # 显示行号
// $ cat -b file.txt                # 非空行显示行号
// $ cat -s file.txt                # 压缩连续空行为一行
// $ cat -A file.txt                # 显示所有字符 (包括制表符^I 行尾$)
//
// 合并文件:
//   $ cat file1.txt file2.txt > merged.txt
//   $ cat *.log > all-logs.log
//
// 创建文件:
//   $ cat > file.txt               # 从键盘输入创建文件 (Ctrl+D 结束)
//   $ cat << EOF > file.txt
//   > Hello World
//   > EOF

// ========== less — 分页查看 (推荐) ==========
// less 是 more 的增强版,可以上下翻页
//
// $ less file.txt                  # 分页查看
// $ less -N file.txt               # 显示行号
// $ less +G file.txt               # 从末尾开始看
// $ less +/search file.txt         # 打开时搜索
//
// less 内部快捷键:
//   空格/PageDown  下一页
//   b/PageUp       上一页
//   /pattern       向下搜索
//   ?pattern       向上搜索
//   n/N            下一个/上一个匹配
//   g/G            首页/末页
//   q              退出
//   h              帮助

// ========== head / tail ==========
// 查看文件的头部/尾部 (默认 10 行)
//
// $ head file.txt                  # 前 10 行
// $ head -n 20 file.txt            # 前 20 行
// $ head -c 100 file.txt           # 前 100 字节
//
// $ tail file.txt                  # 后 10 行
// $ tail -n 20 file.txt            # 后 20 行
// $ tail -f file.txt               # 实时追加强大! (follow)
// $ tail -F file.txt               # 文件轮转后重新跟随
// $ tail -f -n 100 app.log         # 显示最后 100 行并实时监控
```


## tail -f 实战


tail -f 是调试和监控最常用的命令，实时查看日志增长。


```
// ========== tail -f 实战 ==========
// 实时监控 Nginx 访问日志:
//   $ tail -f /var/log/nginx/access.log
//
// 过滤错误:
//   $ tail -f /var/log/app.log | grep ERROR
//
// 同时监控多个日志:
//   $ tail -f /var/log/nginx/access.log /var/log/nginx/error.log
//
// 使用 multitail (需安装):
//   $ multitail /var/log/syslog /var/log/auth.log

// ========== 其他查看工具 ==========
// more — 基础分页 (只能向下翻):
//   $ more file.txt
//
// nl — 带行号显示:
//   $ nl file.txt
//
// od — 八进制/十六进制查看 (二进制文件):
//   $ od -c file.bin           # 显示字符
//   $ od -x file.bin           # 十六进制
//
// xxd — 十六进制转储:
//   $ xxd file.bin
//
// strings — 提取二进制文件中的可读字符串:
//   $ strings /usr/bin/node | head -20

// ========== 查看大型文件 ==========
// 大文件不要用 cat,用 less:
//   $ less hugefile.log
//
// 随机采样查看:
//   $ shuf -n 100 hugefile.log  # 随机取 100 行
//
// 只看特定范围:
//   $ sed -n '1000,2000p' hugefile.log  # 1000-2000 行
//
// 分割文件:
//   $ split -l 10000 huge.log chunk_    # 每 10000 行一个文件
```


> **Note:** 🔍 tail -f /var/log/app.log 是后端开发的必备技能——部署后实时查看日志,问题一目了然。配合 grep 可以立即过滤出关键信息。


## 文件比较与校验


```
// ========== diff (文件比较) ==========
// $ diff file1.txt file2.txt              # 比较两个文件
// $ diff -u file1.txt file2.txt           # 统一格式 (生成补丁)
// $ diff -r dir1/ dir2/                   # 递归比较目录
// $ diff -q file1.txt file2.txt           # 只报告是否相同
//
// 示例输出:
//   --- file1.txt  2024-01-01 12:00:00
//   +++ file2.txt  2024-01-02 12:00:00
//   @@ -1,3 +1,4 @@
//    line1
//   -line2         ← 删除的行
//   +new line      ← 添加的行
//    line3

// ========== 文件校验 ==========
// $ md5sum file.txt              # MD5 哈希 (快速但不够安全)
// $ sha256sum file.txt           # SHA-256 (推荐)
// $ sha1sum file.txt             # SHA-1
//
// 验证下载文件:
//   $ sha256sum downloaded-file.iso
//   $ echo "已知哈希值" | sha256sum -c
//
// 检查目录完整性:
//   $ find . -type f -exec sha256sum {} \; > checksums.sha256
//   $ sha256sum -c checksums.sha256

// ========== wc (Word Count) ==========
// $ wc file.txt                  # 行数 单词数 字节数
// $ wc -l file.txt               # 只显示行数
// $ wc -w file.txt               # 只显示单词数
// $ wc -c file.txt               # 只显示字节数
// $ wc *.txt                     # 统计多个文件
//
// 实用:
//   $ wc -l /var/log/syslog      # 日志总行数
//   $ ps aux | wc -l             # 进程数
//   $ ls -1 | wc -l              # 文件数
```


<!-- Converted from: 6_查看文件cat-less-head-tail.html -->
