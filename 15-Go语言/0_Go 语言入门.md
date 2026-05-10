# Go 语言入门


## 🚀 Go 语言入门


Go 语言特点、安装配置、go mod 模块管理、变量声明、基本类型、零值、常量 iota、fmt 格式化输出。


## Go 语言概述


```
// ========== Go 语言 ==========
// Go 是 Google 开发的编译型静态语言
// 设计目标: 简洁/高效/天生并发

// ========== 核心特点 ==========
// 1. 编译型: 编译为机器码, 运行快
// 2. 静态类型: 类型安全
// 3. 并发原生: goroutine + channel
// 4. 垃圾回收: 自动内存管理
// 5. 简洁语法: 无继承/泛型(1.18+)/异常
// 6. 快速编译: 编译速度快
// 7. 跨平台: Windows/Linux/macOS
// 8. 工具链丰富: go fmt/build/test/mod

// ========== 第一个程序 ==========
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

// 运行:
// go run main.go
// 编译:
// go build -o myapp main.go
// ./myapp

// ========== Go Module ==========
// go mod init example.com/myapp   // 初始化模块
// go get github.com/gin-gonic/gin // 添加依赖
// go mod tidy                     // 整理依赖
// go mod vendor                   // 复制到 vendor

// go.mod 文件:
// module example.com/myapp
//
// go 1.21
//
// require (
//     github.com/gin-gonic/gin v1.9.1
// )

// ========== GOPATH vs Modules ==========
// 旧: GOPATH 模式 (所有项目在 $GOPATH/src 下)
// 新: Go Modules (1.16+ 默认), 项目可在任意位置

// 环境变量:
// go env GOOS GOARCH           // 查看目标平台
// go env -w GOOS=linux         // 设置跨平台编译
// go env -w GOARCH=arm64
```


## 变量与基本类型


```
// ========== 变量声明 ==========

package main

import "fmt"

func main() {
    // ========== 声明方式 ==========
    // 方式 1: var 声明
    var name string = "Alice"
    var age int = 25

    // 方式 2: 类型推断
    var city = "Beijing"        // 推断为 string

    // 方式 3: 短声明 (函数内)
    country := "China"          // 最常用
    count := 42

    // 方式 4: 批量声明
    var (
        x = 1
        y = 2
        z = 3
    )

    // ========== 基本类型 ==========
    // 布尔: bool (true/false)
    var isActive bool = true

    // 字符串: string (UTF-8)
    var msg string = "Hello"

    // 整数:
    var a int     = -42    // 平台相关 (32/64 位)
    var b int8    = 127    // -128 ~ 127
    var c int16   = 32767
    var d int32   = 2147483647
    var e int64   = 9223372036854775807
    var f uint    = 42     // 无符号
    var g uint8   = 255    // byte 别名
    var h uint16  = 65535
    var i uint32  = 4294967295
    var j uint64  = 18446744073709551615

    // 浮点:
    var pi float32 = 3.14159
    var e2 float64 = 2.718281828459045

    // 复数:
    var cpx complex128 = 1 + 2i

    // 字节与符文 (Unicode):
    var b byte = 'A'          // uint8 别名
    var r rune = '中'         // int32 别名, Unicode

    // ========== 零值 (默认值) ==========
    // Go 声明后自动赋零值, 没有未初始化变量
    var zeroInt int        // 0
    var zeroFloat float64  // 0.0
    var zeroBool bool      // false
    var zeroStr string     // ""
    var zeroPtr *int       // nil
    var zeroSlice []int    // nil
    var zeroMap map[int]int// nil

    // ========== 类型转换 ==========
    // Go 没有隐式类型转换
    var m int = 42
    var n float64 = float64(m)  // 显式转换
    var o int32 = int32(n)

    // 字符串与数字:
    // strconv.Itoa(42)           // int → string
    // strconv.Atoi("42")         // string → int
    // strconv.ParseFloat("3.14", 64)
    // fmt.Sprintf("%d", 42)      // int → string (格式化)
}
```


## 常量与 iota


```
// ========== 常量 ==========
// 使用 const 声明, 编译时确定
// 可用作: 数字/字符串/布尔

const Pi = 3.14159
const AppName = "MyApp"

// 批量声明:
const (
    StatusOK    = 200
    StatusNotFound = 404
)

// 类型常量:
const MaxSize int = 1024

// ========== iota ==========
// 常量计数器, 每行自动 +1

const (
    Sunday = iota     // 0
    Monday            // 1
    Tuesday           // 2
    Wednesday         // 3
    Thursday          // 4
    Friday            // 5
    Saturday          // 6
)

// 跳过值:
const (
    A = iota    // 0
    _           // 1 (跳过)
    B           // 2
    C           // 3
)

// 位掩码:
const (
    Read   = 1 << iota  // 1 (1)
    Write               // 2 (10)
    Execute             // 4 (100)
)

// 自定义类型 + iota:
type ByteSize float64

const (
    _           = iota
    KB ByteSize = 1 << (10 * iota)  // 1 << 10
    MB                                // 1 << 20
    GB                                // 1 << 30
    TB                                // 1 << 40
)

// ========== fmt 格式化 ==========
// %v   值的默认格式
// %+v  带字段名的结构体
// %#v  完整 Go 语法
// %T   类型
// %d   十进制整数
// %s   字符串
// %f   浮点 (%.2f 两位小数)
// %t   布尔
// %p   指针

name := "Alice"
age := 30
pi := 3.14159

fmt.Printf("姓名: %s, 年龄: %d\n", name, age)
fmt.Printf("PI: %.2f\n", pi)          // 3.14
fmt.Printf("类型: %T\n", pi)          // float64

str := fmt.Sprintf("Hello, %s!", name)  // 格式化为字符串
fmt.Println(str)                        // Hello, Alice!

// %v 通用:
fmt.Printf("any=%v", 42)               // 42
fmt.Printf("any=%v", "hello")          // hello

// 结构体:
type Point struct{ X, Y int }
p := Point{1, 2}
fmt.Printf("%v\n", p)    // {1 2}
fmt.Printf("%+v\n", p)   // {X:1 Y:2}
fmt.Printf("%#v\n", p)   // main.Point{X:1, Y:2}
```


## 包与导入


```
// ========== 包 (package) ==========
// Go 程序由包组成
// main 包是可执行程序的入口
// 每个 Go 文件属于一个包

// main.go
package main

import (
    "fmt"
    "math/rand"
    "time"

    "example.com/myapp/mypackage"  // 自定义包
)

func main() {
    fmt.Println("随机数:", rand.Intn(100))
    mypackage.Hello()
}

// mypackage/hello.go
package mypackage

import "fmt"

// 导出的函数: 首字母大写
func Hello() {
    fmt.Println("Hello from mypackage")
}

// 未导出的函数: 首字母小写
func internal() {
    // 仅包内可访问
}

// ========== 导入别名 ==========
import (
    "fmt"
    myfmt "example.com/myfmt"     // 别名
    _ "example.com/init"          // 匿名导入 (只执行 init)
    . "fmt"                       // 点导入 (直接使用函数名)
)

// ========== init 函数 ==========
// 每个包可以有多个 init()
// 在 main() 之前自动执行
// 用于初始化操作

package config

import "os"

var Port string

func init() {
    Port = os.Getenv("PORT")
    if Port == "" {
        Port = "8080"
    }
}

// init 执行顺序:
// 1. 导入的包 → 2. 当前包的 const/var → 3. init() → 4. main()

// ========== 空白标识符 _ ==========
// 忽略不需要的值

result, _ := strconv.Atoi("42")  // 忽略 error
_ = someFunction()               // 忽略返回值

// ========== go build 常用命令 ==========
// go build ./...           # 构建所有包
// go build -o app main.go  # 指定输出
// go install               # 安装到 $GOPATH/bin
// go run main.go           # 编译并运行
// go clean                 # 清理构建产物
// go fmt ./...             # 格式化所有代码
// go vet ./...             # 静态检查
```


> **Note:** 💡 Go 要点: go mod init 模块管理; var/:= 变量声明; 基本类型有零值 (非 nil); 显式类型转换; const + iota 枚举; fmt.Printf 格式化; 首字母大写=导出; init 函数自动执行; _ 空白标识符忽略值。


## 练习


<!-- Converted from: 0_Go 语言入门.html -->
