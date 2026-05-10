# Bash脚本基础


## 📜 Bash 脚本基础


Shebang、变量、参数、数组、退出码、脚本执行方式、基础语法。


## 脚本结构


```
// ========== Shebang ==========
// 脚本第一行,指定解释器
// #!/bin/bash           # Bash
// #!/usr/bin/env bash   # 更可移植 (从 PATH 找)
// #!/bin/sh             # POSIX sh (更轻量)
// #!/usr/bin/env python3 # Python 脚本
//
// ========== 创建和运行脚本 ==========
// 1. 创建文件:
//   $ cat > myscript.sh << 'EOF'
//   #!/bin/bash
//   echo "Hello, World!"
//   EOF
//
// 2. 添加执行权限:
//   $ chmod +x myscript.sh
//
// 3. 运行:
//   $ ./myscript.sh        # 当前目录
//   $ bash myscript.sh      # 作为参数传给 bash
//   $ /path/to/myscript.sh  # 绝对路径

// ========== 变量 ==========
// #!/bin/bash
// # 定义变量 (等号两边不能有空格!)
// name="Alice"
// age=25
//
// # 使用变量 ($ 或 ${})
// echo $name              # Alice
// echo "My name is $name" # My name is Alice
// echo "${name}_suffix"   # Alice_suffix (花括号明确边界)
//
// # 只读变量
// readonly API_KEY="abc123"
//
// # 删除变量
// unset name
```


## 脚本参数与特殊变量


```
// ========== 命令行参数 ==========
// $ ./script.sh arg1 arg2 arg3
//
// $0     脚本名 (./script.sh)
// $1     第一个参数 (arg1)
// $2     第二个参数 (arg2)
// $#     参数个数 (3)
// $@     所有参数 (列表)
// $*     所有参数 (单个字符串)
// $?     上一条命令的退出码
// $$     当前脚本的 PID
// $!     上一个后台命令的 PID

// ========== 参数处理示例 ==========
// #!/bin/bash
// echo "脚本名: $0"
// echo "参数个数: $#"
// echo "所有参数: $@"
//
// if [ $# -lt 2 ]; then
//     echo "用法: $0  "
//     exit 1
// fi
//
// name=$1
// age=$2
// echo "姓名: $name, 年龄: $age"

// ========== shift — 移动参数 ==========
// #!/bin/bash
// while [ $# -gt 0 ]; do
//     echo "处理: $1"
//     shift   # $2→$1, $3→$2, 参数数减 1
// done

// ========== 数组 ==========
// #!/bin/bash
// # 定义数组
// fruits=("apple" "banana" "orange")
//
// # 访问元素
// echo ${fruits[0]}        # apple
// echo ${fruits[@]}         # 所有元素
// echo ${#fruits[@]}        # 数组长度
//
// # 添加元素
// fruits+=("grape")
//
// # 遍历
// for fruit in "${fruits[@]}"; do
//     echo $fruit
// done
```


## 退出码与条件测试


```
// ========== 退出码 (Exit Code) ==========
// 0   成功
// 1   一般错误
// 2   误用 shell 内建
// 126 命令不可执行
// 127 命令未找到
// 128 无效退出参数
// 130 Ctrl+C 终止
// 137 SIGKILL
// 255 退出码超出范围
//
// # 自定义退出
// exit 0    # 成功
// exit 1    # 失败
//
// # 检查上一条命令
// if command; then
//     echo "成功"
// else
//     echo "失败, 退出码: $?"
// fi

// ========== test / [ ] 条件测试 ==========
// # 文件测试
// [ -f "file.txt" ]    # 是否为普通文件
// [ -d "dir" ]         # 是否为目录
// [ -e "path" ]        # 是否存在
// [ -s "file" ]        # 文件非空
// [ -r "file" ]        # 是否可读
// [ -w "file" ]        # 是否可写
// [ -x "file" ]        # 是否可执行
// [ -L "link" ]        # 是否为符号链接
//
// # 字符串比较
// [ "$str1" = "$str2" ]    # 相等 (POSIX)
// [ "$str1" == "$str2" ]   # 相等 (Bash)
// [ "$str1" != "$str2" ]   # 不等
// [ -z "$str" ]            # 空字符串
// [ -n "$str" ]            # 非空字符串
//
// # 数字比较
// [ "$a" -eq "$b" ]    # 等于
// [ "$a" -ne "$b" ]    # 不等于
// [ "$a" -gt "$b" ]    # 大于
// [ "$a" -ge "$b" ]    # 大于等于
// [ "$a" -lt "$b" ]    # 小于
// [ "$a" -le "$b" ]    # 小于等于
//
// # 逻辑运算
// [ ! expr ]           # 非
// [ expr1 -a expr2 ]   # 与 (and)
// [ expr1 -o expr2 ]   # 或 (or)
//
// # 推荐: 用 [[ ]] (Bash 扩展)
// [[ -f "$file" && "$str" == "hello" ]]  # 更强大安全
```


## 脚本模板


```
// ========== 标准脚本模板 ==========
// #!/usr/bin/env bash
// #
// # scriptname.sh - 描述脚本功能
// #
// # 用法: ./scriptname.sh [选项] <参数>
// #
//
// set -euo pipefail   # 严格模式
// # -e: 遇到错误退出
// # -u: 使用未定义变量时报错
// # -o pipefail: 管道中任一命令失败则失败
//
// # 用法函数
// usage() {
//     echo "用法: $0 [选项] <参数>"
//     echo "选项:"
//     echo "  -h    显示帮助"
//     echo "  -v    详细模式"
//     exit 1
// }
//
// # 默认值
// verbose=false
//
// # 解析选项
// while getopts "hv" opt; do
//     case $opt in
//         h) usage ;;
//         v) verbose=true ;;
//         *) usage ;;
//     esac
// done
// shift $((OPTIND-1))
//
// # 主逻辑
// main() {
//     local input="$1"
//
//     if $verbose; then
//         echo "处理: $input"
//     fi
//
//     # 核心逻辑
//     echo "完成"
// }
//
// # 入口
// if [ $# -lt 1 ]; then
//     usage
// fi
// main "$@"
```


> **Note:** 📜 在脚本开头加 set -euo pipefail 是 Bash 最佳实践——它让脚本在出错时立即停止,避免"静默失败"导致的灾难。


<!-- Converted from: 15_Bash脚本基础.html -->
