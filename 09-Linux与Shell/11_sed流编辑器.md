# sed流编辑器


## ✂️ sed 流编辑器


sed 替换、行删除、插入/追加、地址范围、正则替换标志、in-place 编辑。


## sed 基础


sed (Stream Editor) 是流编辑器, 非交互式地处理文本, 逐行读取、处理、输出。


```
// ========== sed 基本语法 ==========
// sed [选项] '命令' 文件
//
// 常用选项:
//   -n    不自动输出 (配合 p 命令)
//   -i    直接修改文件 (in-place)
//   -i.bak 备份后修改
//   -e    多个命令
//   -f    从文件读取命令

// ========== s — 替换命令 ==========
// 格式: s/pattern/replacement/flags
//
// $ sed 's/old/new/' file.txt        # 每行第一个匹配
// $ sed 's/old/new/g' file.txt       # 全部替换 (g=global)
// $ sed 's/old/new/2' file.txt       # 每行第二个匹配
// $ sed 's/old/new/gi' file.txt      # 全部+忽略大小写
// $ sed 's/old/new/gw output.txt'    # 替换并写入文件
//
// 分隔符: 不一定要用 /, 可用任何字符
//   $ sed 's|/usr/local|/opt|g' file.txt  # 路径替换
//   $ sed 's_old_new_g' file.txt

// ========== 地址范围 ==========
// $ sed '3s/old/new/' file.txt       # 只替换第 3 行
// $ sed '1,5s/old/new/' file.txt     # 1-5 行
// $ sed '5,$s/old/new/' file.txt     # 5 行到末尾
// $ sed '/pattern/s/old/new/' file   # 匹配行的替换
// $ sed '/start/,/end/s/old/new/'    # 范围行替换

// ========== 删除与打印 ==========
// $ sed '3d' file.txt                # 删除第 3 行
// $ sed '1,5d' file.txt              # 删除 1-5 行
// $ sed '/^$/d' file.txt             # 删除空行
// $ sed '/#/d' file.txt              # 删除注释行
// $ sed -n '5p' file.txt             # 只打印第 5 行
// $ sed -n '10,20p' file.txt         # 打印 10-20 行
```


## sed 实战场景


```
// ========== 配置文件修改 ==========
// $ sed -i 's/Listen 80/Listen 8080/' /etc/nginx/nginx.conf
// $ sed -i.bak 's/debug/info/' config.yml  # 备份并修改

// ========== 批量重命名 ==========
// 将 "foo" 改为 "bar" (直接改文件):
//   $ sed -i 's/foo/bar/g' *.txt
//
// 文件内容中的路径替换:
//   $ sed -i 's|/old/path|/new/path|g' config/*.yml

// ========== 文本格式化 ==========
// 删除行首空白:
//   $ sed 's/^[ \t]*//' file.txt
//
// 删除行尾空白:
//   $ sed 's/[ \t]*$//' file.txt
//
// 压缩连续空行为一行:
//   $ sed '/^$/d' file.txt
//
// 每行前加行号:
//   $ sed = file.txt | sed 'N;s/\n/ /'

// ========== 日志处理 ==========
// 提取某时间段日志:
//   $ sed -n '/2024-01-15 10:00:00/,/2024-01-15 11:00:00/p' app.log
//
// 脱敏 (隐藏密码):
//   $ sed 's/password=[^&]*/password=***/g' request.log

// ========== 多命令 ==========
// $ sed -e 's/foo/bar/g' -e 's/baz/qux/g' file.txt
// $ sed 's/foo/bar/g; s/baz/qux/g' file.txt
// $ sed -f commands.sed file.txt     # 从文件读命令
```


## sed 高级模式


```
// ========== 保持空间 (Hold Space) ==========
// sed 有两个缓冲区: 模式空间 (处理中) 和 保持空间 (暂存)
// 高级操作需要两者配合
//
// h   复制模式空间到保持空间
// H   追加模式空间到保持空间
// g   复制保持空间到模式空间
// G   追加保持空间到模式空间
// x   交换模式空间和保持空间

// ========== 高级示例 ==========
// 倒序文件行:
//   $ sed '1!G;h;$!d' file.txt
//
// 在匹配行后插入:
//   $ sed '/pattern/a\new line' file.txt   # after
//   $ sed '/pattern/i\new line' file.txt   # before

// ========== 分组捕获 ==========
// sed 使用 \(\) 进行分组捕获, \1 \2 引用
//
// 将 "firstName lastName" 改为 "lastName, firstName":
//   $ sed 's/\([a-zA-Z]*\) \([a-zA-Z]*\)/\2, \1/' file.txt

// ========== 字符转换 ==========
// y 命令: 字符映射 (类似于 tr)
//
// $ sed 'y/abc/ABC/' file.txt        # a→A, b→B, c→C
// $ sed 'y/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/'

// ========== 注意 ==========
// macOS sed 与 GNU sed 有差异:
//   $ sed -i '' 's/old/new/g' file.txt     # macOS (需要空参数)
//   $ sed -i 's/old/new/g' file.txt        # Linux
//   建议: 用 perl 替代跨平台 sed
//   $ perl -i -pe 's/old/new/g' file.txt
```


> **Note:** ⚡ sed -i 's/old/new/g' file.txt 是最常用的 sed 命令。但注意: macOS 和 Linux 的 sed 有语法差异,跨平台脚本请用 perl -i -pe 替代。


## 练习


<!-- Converted from: 11_sed流编辑器.html -->
