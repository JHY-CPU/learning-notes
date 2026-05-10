# grep文本搜索


## 🔍 grep 文本搜索


grep 基础用法、正则表达式、递归搜索、上下文行控制、扩展正则。


## grep 基础


grep 是 Linux 最强大的文本搜索工具，在一个或多个文件中搜索匹配模式的行。


```
// ========== 基本用法 ==========
// $ grep pattern file.txt          # 在文件中搜索
// $ grep "hello" file.txt          # 搜索包含 hello 的行
// $ grep -i "hello" file.txt       # 忽略大小写
// $ grep -v "hello" file.txt       # 反向匹配 (不包含 hello)
// $ grep -n "hello" file.txt       # 显示行号
// $ grep -c "hello" file.txt       # 只计数匹配行数
// $ grep -w "hello" file.txt       # 精确匹配单词 (不是 helloworld)
// $ grep -x "hello" file.txt       # 精确匹配整行

// ========== 递归搜索 ==========
// $ grep -r "pattern" /etc/        # 递归搜索目录
// $ grep -R "pattern" ~/projects/  # 递归 (跟随符号链接)
// $ grep -rl "pattern" /etc/       # 只显示文件名
// $ grep -rn "TODO" src/           # 搜索 TODO 注释 (显示行号)
// $ grep -r --include="*.js" "function" /path/
//   # 只搜索 .js 文件
// $ grep -r --exclude="*.min.js" "function" /path/
//   # 排除 .min.js 文件
// $ grep -r --exclude-dir=node_modules "require(" /path/
//   # 排除目录

// ========== 上下文控制 ==========
// $ grep -B 5 "error" log.txt      # 匹配行前 5 行 (Before)
// $ grep -A 5 "error" log.txt      # 匹配行后 5 行 (After)
// $ grep -C 5 "error" log.txt      # 前后各 5 行 (Context)
// $ grep -10 "error" log.txt       # 前后各 10 行 (快捷方式)
```


## 正则表达式


```
// ========== 基础正则 (BRE) ==========
// grep 默认使用基础正则
//
// .            任意单个字符
// *            前一个字符重复 0 次或多次
// ^            行首
// $            行尾
// []           字符集合
// [^]          否定字符集合
// \            转义
//
// 示例:
//   grep "^ERROR" log.txt     # 以 ERROR 开头的行
//   grep "success$" log.txt   # 以 success 结尾的行
//   grep "^$" file.txt        # 空行
//   grep "[0-9]" file.txt     # 包含数字的行
//   grep "http[s]*://" urls   # http:// 或 https://

// ========== 扩展正则 (ERE) ==========
// grep -E 或 egrep 使用扩展正则
//
// +        前一个字符 1 次或多次
// ?        前一个字符 0 次或 1 次
// |        或
// ()       分组
// {n,m}    重复 n 到 m 次
//
// 示例:
//   grep -E "color|colour" file.txt     # 匹配 color 或 colour
//   grep -E "^[A-Z].*\.$" file.txt      # 大写开头句号结尾
//   grep -E "[0-9]{3}-[0-9]{4}" file.txt # 匹配 123-4567
//   grep -E "https?" urls               # http 或 https

// ========== Perl 兼容正则 (PCRE) ==========
// grep -P (需要 GNU grep) 支持更强大的语法
//
//   \d     数字 (等价 [0-9])
//   \w     单词字符
//   \s     空白字符
//   \b     单词边界
//   (?=)   正向预查
//   (?!)   负向预查
//
// 示例:
//   grep -P "\b\d{3}\b" file.txt    # 独立的 3 位数
//   grep -P "(?<=ERROR: ).*" log.txt # ERROR: 之后的内容
```


## grep 实用场景


```
// ========== 日志分析 ==========
// 查找错误:
//   $ grep "ERROR\|FATAL" app.log | tail -50
//
// 统计各类型错误:
//   $ grep -o "ERROR_[A-Z_]*" app.log | sort | uniq -c | sort -nr
//
// 查看某时间段日志:
//   $ grep "2024-01-15 10:" app.log
//
// IP 地址统计:
//   $ grep -oP "\d+\.\d+\.\d+\.\d+" access.log | sort | uniq -c | sort -nr | head -10

// ========== 代码搜索 ==========
// 查找函数定义:
//   $ grep -rn "^func " *.go
//
// 查找 TODO/FIXME:
//   $ grep -rn "TODO\|FIXME\|HACK" src/ --exclude-dir=node_modules
//
// 查找导入语句:
//   $ grep -rn "^import" src/ --include="*.py"

// ========== 文件查找 ==========
// 包含特定内容的文件:
//   $ grep -rl "password" ~/config/
//
// 空文件:
//   $ grep -l "^$" *.txt

// ========== 其他实用变体 ==========
// fgrep (grep -F): 固定字符串, 不解释正则, 更快
//   $ fgrep "a.b" file.txt    # 搜索字面量 "a.b" 而不是正则
//
// zgrep / zcat: 压缩文件中搜索
//   $ zgrep "error" *.gz
//   $ zcat access.log.gz | grep "ERROR"
```


> **Note:** 🔍 grep -rn "TODO" src/ --exclude-dir=node_modules 是我最常用的代码搜索命令之一。强大的正则支持使 grep 成为开发者的瑞士军刀。


## 练习


<!-- Converted from: 10_grep文本搜索.html -->
