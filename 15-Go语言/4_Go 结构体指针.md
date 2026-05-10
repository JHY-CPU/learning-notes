# Go 结构体与指针


## 📐 Go 结构体与指针


struct 定义与初始化、指针 &/*/new、结构体标签 tag、方法接收者 (值/指针)、类型嵌入 (继承模拟)、JSON 序列化。


## 结构体


```
// ========== 结构体 ==========
// Go 没有类, 使用 struct 定义类型
// 是值类型 (赋值/传参复制)

package main

import (
    "encoding/json"
    "fmt"
)

// 定义结构体
type User struct {
    ID        int64
    Username  string
    Email     string
    Age       int
    CreatedAt time.Time
}

func main() {
    // ========== 初始化 ==========
    // 方式 1: 字面量 (推荐, 按字段名)
    u1 := User{
        ID:       1,
        Username: "Alice",
        Email:    "alice@example.com",
        Age:      30,
    }

    // 方式 2: 字面量 (按顺序, 不推荐)
    u2 := User{2, "Bob", "bob@example.com", 25, time.Now()}

    // 方式 3: new (返回指针)
    u3 := new(User)           // *User, 字段为零值
    u3.Username = "Charlie"

    // 方式 4: var 声明 (值类型)
    var u4 User               // 字段为零值
    u4.Username = "David"

    // 方式 5: & 取地址
    u5 := &User{ID: 5, Username: "Eve"}  // *User

    // ========== 访问字段 ==========
    fmt.Println(u1.Username)   // Alice
    fmt.Println(u3.Username)   // Charlie

    u1.Age = 31               // 修改字段

    // ========== 结构体是值类型 ==========
    user1 := User{Username: "Alice"}
    user2 := user1            // 复制整个结构体
    user2.Username = "Bob"
    fmt.Println(user1.Username) // Alice (不受影响)
}

// ========== 结构体标签 (Tag) ==========
// 反射时读取, 用于 JSON/ORM/验证

type Product struct {
    ID       int64   `json:"id"`
    Name     string  `json:"name"`
    Price    float64 `json:"price,omitempty"`
    Stock    int     `json:"-"`                  // 忽略
    Category string  `json:"category,omitempty" validate:"required"`
}

func tagDemo() {
    p := Product{ID: 1, Name: "手机", Price: 2999.0}
    jsonData, _ := json.Marshal(p)
    fmt.Println(string(jsonData))
    // {"id":1,"name":"手机","price":2999}

    // 获取 tag 信息 (反射)
    t := reflect.TypeOf(Product{})
    field, _ := t.FieldByName("Name")
    fmt.Println(field.Tag.Get("json"))  // "name"
}
```


## 指针


```
// ========== 指针 ==========
// & 取地址, * 解引用
// 零值: nil

func main() {
    // ========== 基本操作 ==========
    x := 42
    p := &x               // p 是指向 x 的指针 (类型 *int)

    fmt.Println(p)        // 内存地址 (0x...)
    fmt.Println(*p)       // 42 (解引用)

    *p = 100              // 通过指针修改 x
    fmt.Println(x)        // 100

    // ========== 指针类型 ==========
    var p1 *int           // nil 指针
    var p2 *string
    var p3 *User          // 结构体指针

    // ========== new 函数 ==========
    // 分配零值内存, 返回指针
    ptr := new(int)       // *int, 指向 0
    fmt.Println(*ptr)     // 0
    *ptr = 10
    fmt.Println(*ptr)     // 10

    // new vs &:
    u1 := new(User)           // *User, 零值
    u2 := &User{}             // *User, 零值 (推荐)
    u3 := &User{Username: "A"} // *User

    // ========== 指针作为参数 ==========
    // 传指针可修改原值
    func increment(n *int) {
        *n++
    }

    count := 10
    increment(&count)         // count = 11

    // ========== nil 指针检查 ==========
    var p *int
    if p != nil {
        fmt.Println(*p)     // 安全
    }
    // fmt.Println(*p)      // panic: nil pointer

    // ========== 指针与结构体 ==========
    // 结构体指针直接用 . 访问字段 (自动解引用)
    user := &User{Username: "Alice"}
    user.Username = "Bob"   // 等价于 (*user).Username = "Bob"
}// ========== 何时用指针 ==========
// 需要修改原值: 用指针
// 大结构体传参: 用指针 (避免复制)
// 可 nil 语义: 用指针
// 小对象/不可变: 用值 (缓存友好, 栈分配)

// 常见:
type Config struct {
    Host string
    Port int
}

// 指针接收: 可修改, 避免复制
func (c *Config) SetHost(host string) {
    c.Host = host
}

// 函数返回指针可表示 nil
func findUser(id int) *User {   // 可能返回 nil
    if id <= 0 {
        return nil
    }
    return &User{ID: int64(id)}
}
```


## 方法


```
// ========== 方法 ==========
// 在函数名前加接收者 (receiver)
// 接收者可以是值或指针

type Rectangle struct {
    Width  float64
    Height float64
}

// 值接收者 (不修改原对象)
func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

// 值接收者: 方法内修改不影响原对象
func (r Rectangle) Scale(factor float64) {
    r.Width *= factor    // 无效! 只修改了副本
    r.Height *= factor
}

// 指针接收者 (可修改原对象)
func (r *Rectangle) ScalePtr(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

// 指针接收者: 避免大对象复制
func (r *Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

// 使用方法:
func main() {
    r := Rectangle{Width: 10, Height: 5}
    fmt.Println(r.Area())       // 50
    fmt.Println(r.Perimeter())  // 30

    r.Scale(2)                  // 无效 (值接收者)
    fmt.Println(r.Width)        // 10

    r.ScalePtr(2)               // 有效 (指针接收者)
    fmt.Println(r.Width)        // 20
}

// ========== 值/指针接收者选择 ==========
// 值接收者:
// - 方法内不修改接收者
// - 小对象 (如 Point)
// - 不可变语义

// 指针接收者:
// - 方法内修改接收者 (setter)
// - 大结构体 (避免复制)
// - 包含 sync.Mutex 等不可复制字段
// - 一致性: 如果某方法用指针, 全部用指针

// ========== 任意类型添加方法 ==========
// 只能为当前包的类型定义方法

type MyInt int

func (m MyInt) IsPositive() bool {
    return m > 0
}

func (m *MyInt) Double() {
    *m *= 2
}

// 使用:
// var n MyInt = 42
// fmt.Println(n.IsPositive())  // true
// n.Double()                    // n = 84

// ========== 方法链式调用 ==========
type Builder struct {
    result string
}

func (b *Builder) Add(s string) *Builder {
    b.result += s
    return b
}

func (b *Builder) Build() string {
    return b.result
}

// 使用:
// b := &Builder{}
// result := b.Add("Hello").Add(", ").Add("World!").Build()
// fmt.Println(result)  // "Hello, World!"
```


## 类型嵌入


```
// ========== 类型嵌入 (Embedding) ==========
// Go 用嵌入实现复用 (类似继承, 但不是继承)
// 嵌入的类型自动获得其方法和字段

// 基础类型
type Animal struct {
    Name string
}

func (a Animal) Speak() string {
    return "..."
}

// 嵌入 Animal
type Dog struct {
    Animal              // 嵌入 (匿名), 没有字段名
    Breed string
}

// 重写方法
func (d Dog) Speak() string {
    return "Woof!"
}

func embeddingDemo() {
    dog := Dog{
        Animal: Animal{Name: "Buddy"},
        Breed:  "Golden Retriever",
    }

    // 可以访问 Animal 的字段
    fmt.Println(dog.Name)        // Buddy (提升字段)
    fmt.Println(dog.Animal.Name) // Buddy

    // 方法提升
    fmt.Println(dog.Speak())     // Woof! (自己的)
}

// ========== 接口嵌入 ==========
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// 嵌入两个接口
type ReadWriter interface {
    Reader
    Writer
}

// ========== 结构体嵌入实现装饰器 ==========
type LoggedUser struct {
    User
    logger *log.Logger
}

func (lu *LoggedUser) Login() {
    lu.logger.Printf("用户 %s 登录", lu.Username)
    // 调用原始 User 的方法...
}

// ========== 注意事项 ==========
// 1. 嵌入不是继承, 没有多态
// 2. 同名方法覆盖
// 3. 嵌入指针类型也可以
// 4. 嵌入冲突会导致编译错误
```


> **Note:** 💡 结构体指针要点: struct 值类型; &/* 指针; new 分配零值指针; 标签 JSON/ORM; 方法接收者值 vs 指针 (修改/大对象/一致性); 嵌入提升字段/方法; 任意类型可加方法; 指针参数可修改原值; nil 指针必须检查。


## 练习


<!-- Converted from: 4_Go 结构体指针.html -->
