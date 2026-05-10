# Bash函数与参数传递


## 🔧 Bash 函数与参数传递


函数定义、参数传递、返回值、局部变量、作用域、最佳实践。


## 函数基础


```
// ========== 函数定义 ==========
// 两种语法:
function greet {
    echo "Hello, $1!"
}

greet2() {
    echo "Hi, $1!"
}

// ========== 函数调用 ==========
greet "Alice"          # Hello, Alice!
greet2 "Bob"           # Hi, Bob!

// 定义必须在使用之前 (Bash 是解释执行)
// 先定义后调用

// ========== 函数参数 ==========
// 函数内使用 $1 $2 ... $n 访问参数
// $0 仍然是脚本名,不是函数名!

print_info() {
    local name=$1
    local age=$2
    local city=${3:-"未知"}  # 第3个参数带默认值

    echo "姓名: $name, 年龄: $age, 城市: $city"
}

print_info "Alice" 25 "Beijing"
# 姓名: Alice, 年龄: 25, 城市: Beijing

print_info "Bob" 30
# 姓名: Bob, 年龄: 30, 城市: 未知  (默认值)

// ========== 内部特殊变量 ==========
// $#    参数个数
// $@    所有参数
// $*    所有参数 (单字符串)
// $FUNCNAME  当前函数名

myfunc() {
    echo "函数名: ${FUNCNAME[0]}"
    echo "参数个数: $#"
    for arg in "$@"; do
        echo "  - $arg"
    done
}
```


## 返回值


```
// ========== return ==========
// return 只返回整数 (0-255),用于表示成功/失败
// 0 = 成功, 非 0 = 失败

is_even() {
    local num=$1
    return $(( num % 2 ))  # 偶数为 0,奇数为 1
}

is_even 4 && echo "偶数" || echo "奇数"

// ========== 返回字符串 ==========
// 方法 1: echo (命令替换捕获)
get_user() {
    echo "Alice"    # 通过 stdout "返回"
}

name=$(get_user)   # name=Alice

// 方法 2: 全局变量 (不推荐)
result=""
compute() {
    result=$(( $1 + $2 ))
}
compute 3 4
echo $result       # 7

// 方法 3: 引用传递 (利用 declare -n, Bash 4.3+)
set_var() {
    local -n ref=$1
    ref="new value"
}
myvar="old"
set_var myvar
echo $myvar        # new value

// 方法 4: 输出到文件
write_output() {
    echo "name=Alice" > /tmp/out
    echo "age=25" >> /tmp/out
}

// ========== 返回值风格指南 ==========
// return → 状态码 (成功/失败)
// echo → 数据输出 (字符串/数字)
// 全局变量 → 多个返回值 (谨慎使用)

// 多值返回: echo 多行, read 读取
get_coords() {
    echo "10"
    echo "20"
}

read x y < <(get_coords)
echo "x=$x, y=$y"   # x=10, y=20
```


## 局部变量与作用域


```
// ========== local 关键字 ==========
// local 声明的变量只在函数内可见

global_var="全局"

demo_scope() {
    local local_var="局部"
    inner_var="隐式全局"     # 没加 local → 全局!

    echo "函数内: $local_var"
    echo "函数内: $global_var"
}

demo_scope
# 函数内: 局部
# 函数内: 全局

echo "外部: $local_var"    # 空 (局部不可见)
echo "外部: $inner_var"    # 隐式全局可见!

// 结论: 函数内所有变量都加 local 关键字!

// ========== 变量遮蔽 ==========
name="global"
test_shadow() {
    local name="local"   # 遮蔽全局 name
    echo $name           # local
}
test_shadow
echo $name               # global (不受影响)

// ========== 递归函数 ==========
// Bash 支持递归,注意深度限制
factorial() {
    local n=$1
    if [ $n -le 1 ]; then
        echo 1
    else
        local prev=$(factorial $((n - 1)))
        echo $((n * prev))
    fi
}
echo $(factorial 5)      # 120
```


## 函数高级技巧


```
// ========== 函数作为命令 ==========
// 函数可以和普通命令一样使用: 管道、重定向、后台

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_info "服务启动" >> app.log   # 重定向
log_info "处理请求" | tee -a log  # 管道

// ========== 函数库 ==========
# 在 lib/utils.sh 中定义函数库:
# is_root() { [ "$EUID" -eq 0 ]; }
# is_installed() { command -v "$1" >/dev/null 2>&1; }

# 在脚本中引用:
# source lib/utils.sh
# . lib/utils.sh  # 同效果

if is_root; then
    echo "以 root 运行"
fi

// ========== 错误处理辅助 ==========
die() {
    echo "错误: $*" >&2
    exit 1
}

warn() {
    echo "警告: $*" >&2
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "需要命令: $1"
}

// 使用:
// require_cmd "docker"
// require_cmd "jq"

// ========== getopts 参数解析 ==========
// 用于解析命令行选项

usage() {
    echo "用法: $0 [-v] [-o output] "
    exit 1
}

verbose=false
output=""

while getopts "vo:h" opt; do
    case $opt in
        v) verbose=true ;;
        o) output="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

echo "详细模式: $verbose"
echo "输出文件: $output"
echo "输入参数: $@"
```


> **Note:** 💡 Bash 函数设计原则: (1) 所有变量必须加 local; (2) 用 echo 返回数据,return 返回状态码; (3) 函数名用动词+下划线命名法; (4) 逻辑函数 + 辅助函数分离。这能让 Bash 脚本像真正的编程语言一样组织。


## 练习


<!-- Converted from: 19_Bash函数与参数传递.html -->
