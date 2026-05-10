# awk文本处理


## 📊 awk 文本处理


awk 列处理、模式匹配、内置变量、格式化输出、awk 脚本编程。


## awk 基础


awk 是强大的文本分析工具, 将文本按行和列处理, 尤其适合结构化的数据 (如日志、CSV)。


```
// ========== awk 基本语法 ==========
// awk 'pattern { action }' file
//
// $ awk '{ print $1 }' file.txt      # 打印第一列
// $ awk '{ print $1, $3 }' file.txt  # 打印第一和第三列
// $ awk '{ print NR, $0 }' file.txt  # 打印行号和整行
// $ awk '{ print $NF }' file.txt     # 打印最后一列

// ========== 内置变量 ==========
// $0    整行内容
// $1,$2...  第 N 列
// NR    当前行号 (Number of Record)
// NF    当前行字段数 (Number of Fields)
// FS    字段分隔符 (默认空白, 可设置 -F)
// OFS   输出字段分隔符 (默认空格)
// RS    行分隔符 (默认换行)
// FILENAME 当前文件名

// ========== 基本示例 ==========
// 日志文件: access.log
//   192.168.1.1 GET /index.html 200 1234
//   10.0.0.2 POST /api/login 401 56
//   192.168.1.1 PUT /api/users 200 78
//
// $ awk '{ print $1, $4 }' access.log
//   192.168.1.1 200
//   10.0.0.2 401
//   192.168.1.1 200
//
// 指定分隔符 (CSV):
//   $ awk -F',' '{ print $1, $3 }' data.csv
```


## 模式匹配与条件


```
// ========== 模式匹配 ==========
// $ awk '/ERROR/ { print }' app.log     # 匹配包含 ERROR 的行
// $ awk '$1 ~ /192.168/ { print }' log  # 第一列匹配
// $ awk '$1 !~ /192.168/ { print }' log # 第一列不匹配

// ========== 条件过滤 ==========
// $ awk '$4 > 200 { print }' access.log     # 状态码 > 200
// $ awk '$NF > 1000 { print $1, $NF }' log  # 最后列 > 1000
// $ awk 'NR > 1 && NR < 10 { print }' file  # 2-9 行
// $ awk '$4 == 404 { count++ } END { print count }' log
//   # 统计 404 次数

// ========== BEGIN 和 END 块 ==========
// BEGIN: 在处理文件前执行
// END:   在处理文件后执行
//
// $ awk 'BEGIN { print "开始分析" }
//        { print $0 }
//        END { print "完成" }' file.txt
//
// $ awk 'BEGIN { sum=0 }
//        { sum += $NF }
//        END { print "总和:", sum }' numbers.txt
```


## awk 实战


```
// ========== 日志分析 ==========
// 统计 IP 访问次数:
//   $ awk '{ ips[$1]++ } END { for(ip in ips) print ip, ips[ip] }' access.log | sort -k2 -nr
//
// 统计各状态码数量:
//   $ awk '{ status[$4]++ } END { for(s in status) print s, status[s] }' access.log
//
// 统计总请求大小:
//   $ awk '{ total += $NF } END { print "总流量:", total/1024/1024, "MB" }' access.log

// ========== 数据处理 ==========
// 计算平均值:
//   $ awk '{ sum += $1; count++ } END { print "平均:", sum/count }' numbers.txt
//
// 找出最大/最小值:
//   $ awk 'NR==1 { max=$1; min=$1 } { if($1>max) max=$1; if($11 { print $3, $11 }' | sort -rn | head -5
//
// 查看内存使用率最高的进程:
//   $ ps aux | awk 'NR>1 { print $4, $11 }' | sort -rn | head -5
```


## awk 脚本编程


```
// ========== 变量与数组 ==========
// $ awk '{
//     total += $NF
//     lines[NR] = $0
//   }
//   END {
//     print "总行数:", NR
//     print "总计:", total
//   }' file.txt
//
// 关联数组 (类似字典):
//   $ awk '{ count[$1]++ } END { for(k in count) print k, count[k] }' file

// ========== 控制语句 ==========
// if-else:
//   $ awk '{ if($NF > 1000) print $1, "大"; else print $1, "小" }' file
//
// 循环:
//   $ awk '{ for(i=1; i<=NF; i++) sum += $i; print sum }' file

// ========== 内置函数 ==========
// length():     字符串长度
// substr():     子串
// index():      查找位置
// split():      分割字符串
// tolower():    转小写
// toupper():    转大写
// int():        取整
// rand():       随机数
//
// 示例:
//   $ awk '{ print length($0), $0 }' file.txt  # 显示每行长度

// ========== 一行命令合集 ==========
// 打印 80 个字符以上的行:
//   $ awk 'length($0) > 80' file.txt
//
// 删除行首空白:
//   $ awk '{ sub(/^[ \t]+/, ""); print }' file.txt
//
// 合并两行为一行:
//   $ awk '{ if(NR%2==0) print prev","$0; else prev=$0 }' file.txt
```


> **Note:** 📊 awk 的核心价值在于列处理——一句 awk '{ print $1, $NF }' access.log 就能提取 IP 和状态码。配合 sort 和 uniq 可以完成大部分日志分析需求。


<!-- Converted from: 12_awk文本处理.html -->
