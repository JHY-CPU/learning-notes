# Go 函数


## 🔧 Go 函数


函数声明、多返回值、命名返回值、可变参数、函数类型、闭包、递归、defer 清理、泛型函数 (1.18+)。


## 函数基础


```
// ========== Go 函数 ==========
// 使用 func 关键字声明
// 参数类型在后 (Go 风格)

package main

import (
    "errors"
    "fmt"
)

// ========== 基本函数 ==========
func sayHello(name string) {
    fmt.Printf("Hello, %s!\n", name)
}

func add(a int, b int) int {          // 完整参数
    return a + b
}

func subtract(a, b int) int {        // 同类型简写
    return a - b
}

// ========== 多返回值 ==========
// Go 函数可以返回多个值
// 常用于返回结果 + 错误

func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, errors.New("除数不能为零")
    }
    return a / b, nil
}

// 使用:
// result, err := divide(10, 2)
// if err != nil {
//     log.Fatal(err)
// }

// ========== 命名返回值 ==========
// 提前声明返回值变量名
// 函数体内直接赋值, 最后 return 裸返回

func getCoordinates() (x, y int) {
    x = 10          // 直接赋值命名返回值
    y = 20
    return          // 裸返回 (return x, y)
}

func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return
}

// ========== 可变参数 ==========
// ... 表示可变数量参数, 在函数内是切片

func sum(numbers ...int) int {
    total := 0
    for _, n := range numbers {
        total += n
    }
    return total
}

// sum(1, 2, 3)      → 6
// sum(10, 20)       → 30
// sum()             → 0

func join(sep string, strs ...string) string {
    return strings.Join(strs, sep)
}
// join(", ", "a", "b", "c") → "a, b, c"

// 展开切片:
nums := []int{1, 2, 3, 4, 5}
total := sum(nums...)           // 展开切片为参数

// ========== 函数调用 ==========
func main() {
    sayHello("Alice")

    result := add(3, 4)
    fmt.Println("3 + 4 =", result)   // 7

    q, r := divide(10, 3)
    fmt.Println(q, r)                // 3 nil

    // 忽略某个返回值
    q2, _ := divide(10, 2)

    x, y := getCoordinates()
    fmt.Println(x, y)                // 10 20

    fmt.Println(sum(1, 2, 3, 4))     // 10
}
```


## 函数类型与闭包


```
// ========== 函数类型 ==========
// 函数是一等公民: 可赋值给变量/作为参数/作为返回值

// 定义函数类型
type MathFunc func(int, int) int

// 函数作为参数 (回调)
func compute(a, b int, fn MathFunc) int {
    return fn(a, b)
}

func main() {
    // 函数赋值给变量
    add := func(a, b int) int {
        return a + b
    }
    fmt.Println(add(3, 4))    // 7

    // 匿名函数, 立即执行
    result := func(x, y int) int {
        return x*x + y*y
    }(3, 4)
    fmt.Println(result)       // 25

    // 函数作为参数
    fmt.Println(compute(10, 5, add))            // 15
    fmt.Println(compute(10, 5, func(a, b int) int {
        return a * b
    }))                                         // 50
}

// ========== 闭包 (Closure) ==========
// 函数捕获外部变量

// 计数器生成器
func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

c1 := counter()
fmt.Println(c1())  // 1
fmt.Println(c1())  // 2
fmt.Println(c1())  // 3

c2 := counter()    // 新闭包, 独立 count
fmt.Println(c2())  // 1

// 闭包: 累加器
func adder(base int) func(int) int {
    return func(n int) int {
        base += n
        return base
    }
}

add5 := adder(5)
fmt.Println(add5(3))   // 8
fmt.Println(add5(2))   // 10
fmt.Println(add5(10))  // 20

// ========== 闭包陷阱 ==========
// 循环中捕获循环变量的问题

// ❌ 错误: 所有 goroutine 捕获同一个变量
// for i := 0; i < 3; i++ {
//     go func() {
//         fmt.Println(i)  // 可能都输出 3
//     }()
// }

// ✅ 正确: 传参复制
// for i := 0; i < 3; i++ {
//     go func(n int) {
//         fmt.Println(n)  // 0, 1, 2
//     }(i)
// }
```


## 递归与泛型


```
// ========== 递归 ==========

// 阶乘
func factorial(n int) int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n-1)
}

// 斐波那契
func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

// 二分查找 (递归)
func binarySearch(arr []int, target, low, high int) int {
    if low > high {
        return -1
    }
    mid := low + (high-low)/2
    if arr[mid] == target {
        return mid
    } else if arr[mid] < target {
        return binarySearch(arr, target, mid+1, high)
    } else {
        return binarySearch(arr, target, low, mid-1)
    }
}

// ========== 泛型 (Go 1.18+) ==========
// 使用类型参数 [T any] 或 [T constraints.Ordered]

// 泛型函数: 任意类型
func PrintSlice[T any](items []T) {
    for _, item := range items {
        fmt.Println(item)
    }
}

// 泛型函数: 可比较类型
func Index[T comparable](s []T, x T) int {
    for i, v := range s {
        if v == x {
            return i
        }
    }
    return -1
}

// 泛型函数: 约束接口
type Number interface {
    int | int64 | float64
}

func Sum[T Number](numbers []T) T {
    var total T
    for _, n := range numbers {
        total += n
    }
    return total
}

// 使用:
// PrintSlice([]string{"a", "b", "c"})
// PrintSlice([]int{1, 2, 3})
// idx := Index([]int{10, 20, 30}, 20)  // 1
// total := Sum([]float64{1.5, 2.5, 3.0}) // 7.0

// ========== 泛型类型 ==========
type Stack[T any] struct {
    items []T
}

func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    item := s.items[len(s.items)-1]
    s.items = s.items[:len(s.items)-1]
    return item, true
}

// 使用:
// stack := Stack[string]{}
// stack.Push("hello")
// stack.Push("world")
// val, ok := stack.Pop()  // "world", true

// ========== 函数选项模式 (Functional Options) ==========
type Server struct {
    host    string
    port    int
    timeout time.Duration
    tls     bool
}

type Option func(*Server)

func WithHost(host string) Option {
    return func(s *Server) {
        s.host = host
    }
}

func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(t time.Duration) Option {
    return func(s *Server) {
        s.timeout = t
    }
}

func NewServer(opts ...Option) *Server {
    s := &Server{
        host:    "localhost",
        port:    8080,
        timeout: 30 * time.Second,
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// 使用:
// srv := NewServer(WithPort(9090), WithTimeout(time.Minute))
```


> **Note:** 💡 函数要点: func 声明; 多返回值 (result, err); 命名返回值裸 return; 可变参数 ...type; 函数是一等公民; 闭包捕获外部变量; 循环中复制参数避免陷阱; 泛型 [T any] 1.18+; 函数选项模式灵活配置。


## 练习


<!-- Converted from: 2_Go 函数.html -->
