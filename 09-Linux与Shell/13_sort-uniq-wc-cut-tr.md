# sort-uniq-wc-cut-tr


## 🔧 sort/uniq/wc/cut/tr


文本排序、去重统计、计数、列切割、字符转换工具详解。


## sort — 排序


```
// ========== sort 基础 ==========
// $ sort file.txt                    # 按字母升序
// $ sort -r file.txt                 # 降序
// $ sort -n file.txt                 # 按数字排序
// $ sort -h sizes.txt                # 人类可读排序 (1K < 1M < 1G)
// $ sort -u file.txt                 # 排序并去重 (同 sort|uniq)
// $ sort -k2 file.txt                # 按第 2 列排序
// $ sort -t',' -k3 -n data.csv       # CSV 按第 3 列数字排序
//
// ========== sort 进阶 ==========
// $ sort -k2,2 file.txt              # 严格按第 2 列 (非第 2 列起)
// $ sort -k2 -k3 file.txt            # 先按第 2 列,再按第 3 列
// $ sort -n -k5 -t' ' access.log     # 按第 5 列数字排序
// $ sort -V versions.txt             # 版本号排序 (v1.2 < v1.10)
// $ sort -c file.txt                 # 检查文件是否已排序
// $ sort -m sorted1.txt sorted2.txt  # 合并已排序文件

// ========== uniq — 去重与统计 ==========
// 注意: uniq 只能去除 连续重复 的行! 必须先 sort
//
// $ uniq file.txt                    # 去除连续重复
// $ uniq -c file.txt                 # 统计重复次数
// $ uniq -d file.txt                 # 只显示重复行
// $ uniq -u file.txt                 # 只显示不重复行
// $ sort file.txt | uniq             # 经典组合: 全局去重
// $ sort file.txt | uniq -c | sort -nr  # 按出现频率排序
```


## wc / cut / tr


```
// ========== wc (Word Count) ==========
// $ wc file.txt                      # 行数 单词数 字节数
// $ wc -l file.txt                   # 行数
// $ wc -w file.txt                   # 单词数
// $ wc -c file.txt                   # 字节数
// $ wc -m file.txt                   # 字符数 (含多字节)
// $ wc -L file.txt                   # 最长行的长度
//
// $ wc -l *.txt                      # 多个文件统计
// $ cat *.log | wc -l               # 总行数
// $ find . -name "*.js" | xargs wc -l # 代码总行数

// ========== cut — 列切割 ==========
// $ cut -d',' -f1,3 data.csv         # 按逗号切,取第 1,3 列
// $ cut -d' ' -f1 access.log         # 取第一列 (IP)
// $ cut -c1-10 file.txt              # 取每行前 10 字符
// $ cut -f1-3 -d: /etc/passwd        # 取前 3 个冒号分隔字段
// $ cut -c3- file.txt                # 从第 3 字符到末尾

// ========== tr — 字符转换 ==========
// $ tr 'a-z' 'A-Z' < file.txt        # 转大写
// $ tr ',' '\t' < data.csv            # 逗号→制表符
// $ tr -d '\r' < win.txt > unix.txt   # 删除 Windows 换行符 (CR)
// $ tr -s ' ' < file.txt              # 压缩多个空格为一个
// $ tr -d ' ' < file.txt              # 删除所有空格
// $ tr '[:upper:]' '[:lower:]'        # 字符类方式转小写

// ========== 经典组合 ==========
// 出现最多的 IP (Top 5):
//   $ cut -d' ' -f1 access.log | sort | uniq -c | sort -nr | head -5
//
// 统计 HTTP 方法:
//   $ cut -d' ' -f2 access.log | sort | uniq -c | sort -nr
//
// 处理 /etc/passwd:
//   $ cut -d: -f1,3 /etc/passwd | sort -t: -k2 -n
```


## 文本处理实战


```
// ========== 完整日志分析管道 ==========
// $ cat access.log |
//   cut -d' ' -f1,4,7 |      # 提取 IP, 状态码, 路径
//   sort |                    # 排序
//   uniq -c |                 # 统计
//   sort -rn |                # 按次数降序
//   head -20                  # Top 20
//
// ========== 磁盘使用 Top 10 ==========
// $ du -sh /* 2>/dev/null | sort -rh | head -10

// ========== 查找大文件 ==========
// $ find / -type f -size +100M 2>/dev/null | sort -r | head -10

// ========== 文件类型统计 ==========
// $ find . -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn
//
// ========== 文本去重统计 ==========
// $ sort words.txt | uniq -c | sort -nr | head -20

// ========== 将脚本转换为 Unix 格式 ==========
// $ tr -d '\r' < script.sh > script_unix.sh
// $ sed -i 's/\r$//' script.sh     # 同效果

// ========== 统计代码行数 ==========
// $ find . -name "*.py" -o -name "*.js" | xargs wc -l | tail -1
// $ find . -name "*.go" -exec wc -l {} \; | awk '{ total += $1 } END { print total }'
```


> **Note:** 🔧 组合是最强大的——用 cut 提取列, sort 排序, uniq -c 统计, sort -nr 倒序, head 取前 N。这个管道组合能解决 80% 的文本分析需求。


## 文本处理工具对比


```
// ========== 工具选择指南 ==========
//
// 任务                    工具
// ─────────────────────────────────
// 列提取                  cut / awk
// 文件计数                wc -l
// 按行排序                sort
// 字符替换                tr / sed
// 正则替换                sed
// 去重统计                sort | uniq -c
// 上下文搜索              grep -C
// 复杂列处理              awk
// 多文件替换              sed -i / perl -i
// JSON 处理               jq
// XML 处理                xmlstarlet / xmllint
// CSV 处理                csvkit / awk
// YAML 处理               yq (需安装)

// ========== 命令速查表 ==========
// 查找:      grep / rg / ag
// 替换:      sed / perl
// 分析:      awk
// 去重:      sort | uniq
// 提取:      cut
// 转换:      tr
// 统计:      wc
// 选择:      head / tail
// 分页:      less
// 比较:      diff
// 查看:      cat / bat
```


<!-- Converted from: 13_sort-uniq-wc-cut-tr.html -->
