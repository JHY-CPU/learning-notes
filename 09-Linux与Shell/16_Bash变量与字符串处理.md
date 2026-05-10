# Bash变量与字符串处理


## 📝 Bash 变量与字符串处理


变量定义与使用、字符串操作、参数扩展、变量作用域。


## 变量基础


```
// ========== 变量定义与使用 ==========
// 定义 (等号两边无空格!)
name="Alice"
age=25
// 注意: Bash 变量默认都是字符串,算术需用 $(( ))

// 使用变量
echo $name           # Alice
echo "Hi, $name"     # Hi, Alice (双引号内可展开)
echo 'Hi, $name'     # Hi, $name (单引号不展开)
echo ${name}         # Alice (花括号明确边界)

// 未定义变量
echo $nonexist       # 空字符串 (不会报错,除非 set -u)

// ========== 变量命名规则 ==========
// 1. 字母/数字/下划线,不能数字开头
// 2. 区分大小写: NAME != name
// 3. 避免使用环境变量名 (PATH/HOME/UID等)
valid_name="ok"
_name="ok"
name1="ok"
// 1name="no"        # 错误!

// ========== 变量的类型 ==========
// Bash 变量无类型,全部存为字符串
var=42              # 字符串 "42"
echo $((var + 1))   # 43 (算术展开中视为数字)

// declare/typeset 可指定属性
declare -i num=10   # 整数属性
declare -r CONST=42 # 只读
declare -a arr      # 数组
declare -A map      # 关联数组 (Bash 4+)
declare -l lower    # 自动转小写
declare -u upper    # 自动转大写
declare -x ENV_VAR  # 导出为环境变量

// ========== 删除变量 ==========
unset name          # 删除变量
unset -v name       # 明确删除变量
```


## 参数扩展 (Parameter Expansion)


```
// ========== 默认值处理 ==========
// ${var:-default}  变量未定义或为空 → 默认值
// ${var-default}   变量未定义 → 默认值 (空串不算未定义)
${name:-"world"}    # name为空 → "world"
${name-"world"}     # name="" → "" (空串不算未定义)

// ${var:=default}  变量未定义或为空 → 赋值并返回
// 常用于脚本中设置默认值
: ${OUTPUT_DIR:="/tmp"}  # : 是空命令,利用副作用

// ${var:?error}    变量未定义或为空 → 报错退出
${DATABASE_URL:?"缺少 DATABASE_URL 环境变量"}

// ${var:+alt}      变量已定义且非空 → 替代值
${DEBUG:+ "-v"}     # DEBUG 非空时返回 "-v"

// ========== 字符串长度 ==========
str="hello"
echo ${#str}           # 5

// ========== 子串截取 ==========
str="hello world"
echo ${str:0:5}        # hello (从0取5个)
echo ${str:6}          # world (从6到末尾)
echo ${str: -5}        # world (从末尾取5个,注意空格)
echo ${str:(-5)}       # world (同)

// ========== 子串删除 ==========
// # 从头删最短匹配, ## 从头删最长匹配
// % 从尾删最短匹配, %% 从尾删最长匹配
url="https://example.com/path/file.txt"
echo ${url#*/}         # /example.com/path/file.txt  (最短前缀)
echo ${url##*/}        # file.txt                    (最长前缀)
echo ${url%.*}         # https://example.com/path/file (最短后缀)
echo ${url%%.*}        # https://example             (最长后缀)

// ========== 替换 ==========
str="foo bar foo"
echo ${str/foo/baz}    # baz bar foo  (第一个)
echo ${str//foo/baz}   # baz bar baz  (全部)
echo ${str/#foo/baz}   # baz bar foo  (开头匹配)
echo ${str/%foo/baz}   # foo bar baz  (结尾匹配)

// ========== 大小写转换 ==========
str="Hello World"
echo ${str,,}          # hello world  (全小写, Bash 4+)
echo ${str,}           # hello World  (首字母小写)
echo ${str^^}          # HELLO WORLD  (全大写)
echo ${str^}           # Hello World  (首字母大写)
```


## 字符串操作


```
// ========== 连接字符串 ==========
first="Hello"
second=" World"
result="$first$second"    # Hello World (直接拼接)
result="${first} ${second}" # Hello World

// ========== 字符串包含检查 ==========
str="Hello World"
if [[ "$str" == *"World"* ]]; then
    echo "包含 World"
fi

if [[ "$str" =~ ^Hello ]]; then
    echo "以 Hello 开头"  # 正则匹配
fi

// ========== 分割字符串 ==========
// 方法 1: IFS 分割
IFS=',' read -ra parts <<< "a,b,c"
// parts=([0]="a" [1]="b" [2]="c")

// 方法 2: 替换分隔符为换行,配合 readarray/mapfile
data="a:b:c"
readarray -d: -t items <<< "$data"

// ========== printf 格式化 ==========
printf "%-10s %5d\n" "Alice" 25   # 左对齐 右对齐
printf "%x\n" 255                  # ff (十六进制)
printf "%o\n" 255                  # 377 (八进制)
printf "%.2f\n" 3.14159           # 3.14

// ========== Here-doc 与 Here-string ==========
// Here-document
cat << 'EOF'
多行文本
不会展开 $变量
EOF

// Here-string
cat <<< "hello"

// 缩进 here-doc (<<- 允许 tab 缩进)
cat <<- 'EOF'

使用 tab 缩进

EOF
```


> **Note:** 💡 参数扩展是 Bash 最强大的功能之一,熟练使用 ${var:-default}、${var#pattern}、${var%pattern} 可以写出更短更安全的脚本。用 declare -i/-a/-A 声明变量类型是良好的 Bash 实践。


## 特殊变量与作用域


```
// ========== 特殊变量 ==========
// $0      脚本名
// $1-$9   位置参数
// $#      参数个数
// $@      所有参数 (数组)
// $*      所有参数 (字符串)
// $?      最后命令退出码
// $$      当前 Shell PID
// $!      最后后台进程 PID
// $-      当前 Shell 选项 (himBHs)
// $_      最后命令的最后一个参数

// ========== 变量作用域 ==========
// 全局变量: 默认
// 局部变量: local 关键字 (函数内)
// 环境变量: export 导出

myfunc() {
    local local_var="仅在函数内可见"
    global_var="全局可见"
    echo $local_var
}

// ========== indirection 间接引用 ==========
var_name="color"
color="red"
echo ${!var_name}      # red (间接引用)

// ========== eval 小心使用 ==========
// eval 可执行字符串作为命令,但存在安全风险
cmd="echo hello"
eval $cmd              # hello
// 危险: eval "echo $user_input"  # 命令注入!

// ========== readonly 只读 ==========
readonly PI=3.14159
readonly -a SIZES=(small medium large)
// readonly -p  # 列出所有只读变量
```


## 练习


<!-- Converted from: 16_Bash变量与字符串处理.html -->
