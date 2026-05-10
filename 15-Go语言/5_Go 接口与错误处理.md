# Go 接口与错误处理


## 🔌 Go 接口与错误处理


接口隐式满足、空接口 any、类型断言、error 接口、自定义错误、fmt.Errorf 包装错误、errors.Is/As、panic/recover 最佳实践。


## 接口基础


```
// ========== 接口 ==========
// Go 接口是方法集合
// 隐式实现: 类型实现了所有方法就自动满足接口
// 不需要 implements 关键字

package main

import "fmt"

// ========== 定义接口 ==========
type Speaker interface {
    Speak() string
}

type Mover interface {
    Move(distance float64)
}

// ========== 隐式实现 ==========
type Dog struct {
    Name string
}

// Dog 实现了 Speaker 接口 (不写 implements)
func (d Dog) Speak() string {
    return "Woof! I'm " + d.Name
}

type Cat struct {
    Name string
}

func (c Cat) Speak() string {
    return "Meow! I'm " + c.Name
}

// 一个类型实现多个接口
func (d Dog) Move(dist float64) {
    fmt.Printf("%s moved %.1f meters\n", d.Name, dist)
}

// ========== 使用接口 ==========
func printSpeak(s Speaker) {
    fmt.Println(s.Speak())
}

func main() {
    dog := Dog{Name: "Buddy"}
    cat := Cat{Name: "Kitty"}

    // 多态: 同一个接口不同实现
    printSpeak(dog)  // Woof! I'm Buddy
    printSpeak(cat)  // Meow! I'm Kitty

    // 接口变量
    var s Speaker
    s = dog
    fmt.Println(s.Speak())

    s = cat
    fmt.Println(s.Speak())

    // 类型断言
    dog2, ok := s.(Dog)
    if ok {
        fmt.Println("It's a dog:", dog2.Name)
    }
}

// ========== io.Writer 接口示例 ==========
// 标准库接口: type Writer interface { Write(p []byte) (n int, err error) }

type MyWriter struct{}

func (w MyWriter) Write(p []byte) (n int, err error) {
    fmt.Println("写入:", string(p))
    return len(p), nil
}

// 使用:
// var w io.Writer = MyWriter{}
// w.Write([]byte("hello"))

// ========== 空接口 interface{} / any ==========
// Go 1.18+ 使用 any (alias of interface{})

func printAny(v any) {
    fmt.Printf("值: %v, 类型: %T\n", v, v)
}

func main() {
    printAny(42)           // 值: 42, 类型: int
    printAny("hello")      // 值: hello, 类型: string
    printAny(3.14)         // 值: 3.14, 类型: float64
    printAny(Dog{})        // 值: {}, 类型: main.Dog
}
```


## 类型断言与 switch


```
// ========== 类型断言 ==========
// 从接口值提取具体类型

func assert(x interface{}) {
    // 方式 1: 安全断言 (推荐)
    value, ok := x.(int)
    if ok {
        fmt.Println("整数:", value)
    } else {
        fmt.Println("不是整数")
    }

    // 方式 2: 不安全 (失败 panic)
    // value := x.(int)  // 如果 x 不是 int, panic!
}

// ========== 类型 switch ==========
func checkType(v interface{}) {
    switch val := v.(type) {
    case int:
        fmt.Println("整数:", val)
    case string:
        fmt.Println("字符串:", val)
    case bool:
        fmt.Println("布尔:", val)
    case []int:
        fmt.Println("整数切片:", len(val))
    case nil:
        fmt.Println("nil")
    default:
        fmt.Printf("未知类型: %T\n", val)
    }
}

func main() {
    checkType(42)              // 整数: 42
    checkType("hello")         // 字符串: hello
    checkType(true)            // 布尔: true
    checkType(nil)             // nil
}

// ========== 接口组合 ==========
// 接口可以嵌入接口

type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

type Closer interface {
    Close() error
}

// 组合接口
type ReadWriter interface {
    Reader
    Writer
}

type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

// ========== 接口值为 nil 的陷阱 ==========

// ❌ 常见错误: 接口持有 nil 指针, 但接口本身不为 nil
func getWriter() io.Writer {
    var buf *bytes.Buffer
    return buf    // 返回 *bytes.Buffer(nil) 作为 io.Writer
}

// w := getWriter()
// w.Write([]byte("hello"))  // panic!

// 接口值为 nil 的条件: 类型和值都是 nil
// 上述例子: 类型 = *bytes.Buffer, 值 = nil, 所以接口 != nil

// ✅ 正确:
func getWriterCorrect() io.Writer {
    return nil  // 或返回具体 nil 值
}
```


## 错误处理


```
// ========== Go 错误处理 ==========
// Go 没有异常, 使用 error 类型返回值
// error 是内置接口:
// type error interface {
//     Error() string
// }

// ========== 基本模式 ==========
func divide(a, b int) (int, error) {
    if b == 0 {
        return 0, fmt.Errorf("除数不能为零")
    }
    return a / b, nil
}

func main() {
    result, err := divide(10, 0)
    if err != nil {
        fmt.Println("错误:", err)  // 除数不能为零
        return
    }
    fmt.Println(result)
}

// ========== 自定义错误 ==========
// 实现 error 接口即可

// 方式 1: struct 实现
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("验证失败: %s - %s", e.Field, e.Message)
}

// 方式 2: 哨兵错误 (sentinel error)
var ErrNotFound = errors.New("资源不存在")
var ErrForbidden = errors.New("无权限")

// 方式 3: 带错误码
type AppError struct {
    Code    int
    Message string
    Err     error  // 内部错误
}

func (e *AppError) Error() string {
    return fmt.Sprintf("[%d] %s", e.Code, e.Message)
}

func (e *AppError) Unwrap() error {
    return e.Err  // 支持 errors.Is/As 链
}

// ========== 错误包装 ==========
// Go 1.13+ fmt.Errorf %w 包装错误

func readConfig(path string) error {
    file, err := os.Open(path)
    if err != nil {
        return fmt.Errorf("打开配置文件失败: %w", err)
    }
    defer file.Close()
    // ...
    return nil
}

// errors.Is: 检查错误链中是否包含特定错误
func main() {
    err := readConfig("/not/exist")
    if errors.Is(err, os.ErrNotExist) {
        fmt.Println("文件不存在!")
    }
}

// errors.As: 检查错误链中是否包含特定类型
var appErr *AppError
if errors.As(err, &appErr) {
    fmt.Printf("错误码: %d\n", appErr.Code)
}

// ========== 错误处理最佳实践 ==========
// 1. 总是检查错误
// 2. 如果不需要错误, 用 _
// result, _ := doSomething()

// 3. 早 return 缩进少
// ❌ if err == nil { ... } else { ... }
// ✅ if err != nil { return err }

// 4. 错误信息大写/标点
// errors.New("database connection failed")

// 5. %w 包装错误 (而不是 %v)
// 6. 日志 + 返回错误 (不要同时日志和返回)
// 7. 忽略错误要注释说明原因
```


## 实践示例


```
// ========== 真实错误处理示例 ==========

package main

import (
    "errors"
    "fmt"
    "net/http"
)

// ========== 自定义业务错误 ==========
type BizError struct {
    Code    int
    Message string
}

var (
    ErrUserNotFound   = &BizError{1001, "用户不存在"}
    ErrPasswordWrong  = &BizError{1002, "密码错误"}
    ErrUserDisabled   = &BizError{1003, "用户已禁用"}
    ErrInsufficientBal = &BizError{2001, "余额不足"}
)

func (e *BizError) Error() string {
    return fmt.Sprintf("Biz-%d: %s", e.Code, e.Message)
}

// ========== service 层 ==========
type UserService struct{}

func (s *UserService) Login(username, password string) (*User, error) {
    if username == "" {
        return nil, &BizError{1000, "用户名不能为空"}
    }

    // 模拟数据库查询
    if username != "admin" {
        return nil, ErrUserNotFound
    }

    if password != "123456" {
        // 包装额外上下文
        return nil, fmt.Errorf("用户 %s %w", username, ErrPasswordWrong)
    }

    return &User{Username: username}, nil
}

// ========== handler 层 ==========
func loginHandler(w http.ResponseWriter, r *http.Request) {
    username := r.FormValue("username")
    password := r.FormValue("password")

    svc := &UserService{}
    user, err := svc.Login(username, password)

    if err != nil {
        // 错误类型分类
        var bizErr *BizError
        if errors.As(err, &bizErr) {
            // 业务错误: 返回对应 HTTP 状态码
            switch bizErr.Code {
            case 1001, 1002:
                http.Error(w, bizErr.Message, http.StatusUnauthorized)
            case 1003:
                http.Error(w, bizErr.Message, http.StatusForbidden)
            default:
                http.Error(w, bizErr.Message, http.StatusBadRequest)
            }
        } else {
            // 系统错误: 500
            log.Printf("系统错误: %v", err)
            http.Error(w, "内部错误", http.StatusInternalServerError)
        }
        return
    }

    fmt.Fprintf(w, "欢迎 %s", user.Username)
}

// ========== 函数选项模式 + 错误 ==========
type Option func(*Config) error

func WithHost(host string) Option {
    return func(c *Config) error {
        if host == "" {
            return errors.New("host 不能为空")
        }
        c.Host = host
        return nil
    }
}

func NewConfig(opts ...Option) (*Config, error) {
    c := &Config{Host: "localhost", Port: 8080}
    for _, opt := range opts {
        if err := opt(c); err != nil {
            return nil, fmt.Errorf("配置错误: %w", err)
        }
    }
    return c, nil
}

// 使用:
// cfg, err := NewConfig(WithHost("example.com"), WithPort(9090))
```


> **Note:** 💡 接口与错误要点: 接口隐式满足 (duck typing); any 空接口; 类型断言 val, ok := x.(T); 类型 switch v.(type); error 是内置接口; 自定义 error; errors.Is/As 链式检查; %w 包装错误; 避免接口值持有 nil 指针; 错误分类处理。


## 练习


<!-- Converted from: 5_Go 接口与错误处理.html -->
