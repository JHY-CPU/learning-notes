# Go nil 与零值全面理解


## ⬜ Go nil 与零值全面理解


各类型零值、nil 在不同类型中的含义、nil slice vs empty slice、nil map 安全操作、nil channel 特性、nil interface 与类型信息、零值可用 vs 不可用类型。


## 各类型零值


```
// ========== Go 零值 (Zero Value) ==========
// Go 变量声明后自动初始化为零值, 没有未初始化变量

func zeroValues() {
    var b bool        // false
    var i int         // 0
    var f float64     // 0.0
    var s string      // ""
    var p *int        // nil (指针)
    var sl []int      // nil (切片)
    var m map[int]int // nil (map)
    var ch chan int   // nil (channel)
    var fn func()     // nil (函数)
    var iface interface{} // nil (接口)
    var st struct{}   // struct{}{} (结构体零值)

    fmt.Printf("bool:    %v\n", b)
    fmt.Printf("int:     %v\n", i)
    fmt.Printf("string:  %q\n", s)
    fmt.Printf("pointer: %v\n", p)
    fmt.Printf("slice:   %#v\n", sl)   // []int(nil)
    fmt.Printf("map:     %#v\n", m)    // map[int]int(nil)
    fmt.Printf("chan:    %v\n", ch)
    fmt.Printf("func:    %v\n", fn)
    fmt.Printf("iface:   %#v\n", iface) // <nil>
}

// ========== 零值可用类型 ==========
// 某些类型的零值可直接使用 (无需初始化)

// 1. sync.Mutex — 零值可用, 未锁定状态
type SafeCounter struct {
    mu    sync.Mutex  // 零值即可用!
    count int
}

func (c *SafeCounter) Inc() {
    c.mu.Lock()
    c.count++
    c.mu.Unlock()
}

// 2. bytes.Buffer — 零值可用
func zeroBuffer() {
    var buf bytes.Buffer  // 零值可用, 空缓冲区
    buf.WriteString("hello")
    fmt.Println(buf.String())  // "hello"
}

// 3. testing.T — 由框架构造, 用户不需初始化

// 4. 切片 append — nil 切片可直接 append
func nilAppend() {
    var s []int          // nil slice
    s = append(s, 1)     // 安全! nil 切片可 append
    s = append(s, 2, 3)  // 自动分配底层数组
    fmt.Println(s)       // [1, 2, 3]
}

// 5. 结构体零值 (所有字段为各自零值)
type Point struct {
    X, Y float64
}
var origin Point  // {0, 0}, 直接可用

// ========== 零值不可用 ==========
// map, channel 需要 make 初始化
func zeroNotUsable() {
    var m map[string]int

    // 读操作安全
    v, ok := m["key"]  // 不 panic, v = 0, ok = false

    // 写操作 panic!
    // m["key"] = 1  // panic: assignment to entry in nil map
    // delete(m, "key")  // 安全! nil map delete 是空操作

    var ch chan int
    // <-ch     // 永久阻塞!
    // ch <- 1  // 永久阻塞!
    close(ch)   // panic: close of nil channel

    var f func()
    // f()      // panic: runtime error (nil function)
}
```


## nil 在不同类型中的含义


```
// ========== nil 的语义 ==========
// nil 没有单一类型, 在不同类型中有不同表示

func nilMeaning() {
    // nil 指针: 表示没有指向有效内存
    var p *Config
    if p != nil {
        p.Host = "localhost"  // 解引用前检查
    }

    // nil 切片: 表示没有底层数组
    var s []int
    fmt.Println(len(s))  // 0
    fmt.Println(cap(s))  // 0
    // 语义: 空集合, 但 JSON 序列化为 null

    // nil map: 表示没有哈希表
    var m map[string]int
    // 语义: 只读安全的空映射, 不能写

    // nil channel: 表示不存在通道
    var ch chan int
    // 语义: select 中 nil channel 永远不会被选中!

    // nil 接口: 没有类型信息和值
    var i interface{}
    // 语义: 空接口值

    // nil 函数: 未设置函数
    var fn func(string) error
    // 语义: 调用导致 panic
}

// ========== nil 切片 vs 空切片 ==========
// 核心区别: 底层数组指针不同

func nilVsEmpty() {
    var s1 []int           // nil slice
    s2 := []int{}          // empty slice
    s3 := make([]int, 0)   // empty slice

    fmt.Println(s1 == nil)  // true  — 没有底层数组
    fmt.Println(s2 == nil)  // false — 有底层数组 (空)
    fmt.Println(s3 == nil)  // false

    // 底层结构:
    // s1: {Data: 0, Len: 0, Cap: 0}
    // s2: {Data: 0x..., Len: 0, Cap: 0}  (指向空数组)

    // JSON 序列化差异:
    json1, _ := json.Marshal(s1)  // "null"
    json2, _ := json.Marshal(s2)  // "[]"

    // 通常推荐: 返回 nil 切片表示空结果 (更简洁)
    func getItems() []int {
        return nil  // 调用方 range 安全, JSON 返回 null
    }
}

// ========== nil 接口 ==========
// 接口值 = (type, value) 对
// 只有 type 和 value 都为 nil 时接口才为 nil

func nilInterface() {
    var i interface{}       // nil 接口: (nil, nil)
    fmt.Println(i == nil)   // true

    var p *int = nil        // nil 指针
    i = p                   // 接口: (*int, nil)
    fmt.Println(i == nil)   // false! 因为 type 不为 nil

    // 判断接口内指针是否为 nil:
    func isNilPtr(i interface{}) bool {
        v := reflect.ValueOf(i)
        return v.IsNil()  // 需确保传入的是指针类型
    }
    fmt.Println(isNilPtr(i))  // true

    // 理解: 空接口 == nil 检查应结合 reflect
    // 或设计上避免将 nil 指针赋给接口
}

// ========== nil channel 的妙用 ==========
// nil channel 在 select 中永远不会被选中
// 用于动态开关 channel

func nilChannelTrick() {
    ch1 := make(chan int)
    var ch2 chan int  // nil channel (关闭)

    select {
    case v := <-ch1:
        fmt.Println("ch1:", v)
    case v := <-ch2:
        // 永远不会被执行 (ch2 为 nil)
        fmt.Println("ch2:", v)
    default:
        fmt.Println("no message")
    }

    // 动态开关 channel 模式:
    ch := make(chan int, 1)
    ch <- 1

    var activeChan chan int  // 初始 nil

    for i := 0; i < 3; i++ {
        if i == 1 {
            activeChan = ch  // 动态启用
        }
        select {
        case v := <-activeChan:
            fmt.Println("收到:", v)
            activeChan = nil  // 消费后禁用
        default:
            fmt.Println("无数据")
        }
    }
}
```


## nil 安全与模式


```
// ========== nil 接收者 ==========
// Go 允许在 nil 指针上调用方法! (只要方法处理 nil)

// 常见模式: 链表 nil 安全实现
type Node struct {
    Value int
    Next  *Node
}

// nil 接收者也是安全的!
func (n *Node) Sum() int {
    if n == nil {
        return 0  // nil 节点的总和 = 0
    }
    return n.Value + n.Next.Sum()
}

func (n *Node) PrintAll() {
    if n == nil {
        return
    }
    fmt.Println(n.Value)
    n.Next.PrintAll()
}

// nil 接收者模式: 链式操作
type IntOption struct {
    val    int
    hasVal bool
}

func (o *IntOption) Get() (int, bool) {
    if o == nil {
        return 0, false
    }
    return o.val, o.hasVal
}

// ========== nil 安全设计原则 ==========
// 1. 函数返回: 明确 nil 的含义

// 两种风格:
// 风格 A: 空结果返回 nil 切片
func FindUsers(active bool) []User {
    if !active {
        return nil  // 调用方 range 安全
    }
    // ... 查询
}

// 风格 B: 空结果返回空切片 (JSON 为 [])
func FindUsersJSON(active bool) []User {
    if !active {
        return []User{}  // JSON: []
    }
}

// 2. nil 值检查函数
func requireNonNil(name string, v interface{}) {
    if v == nil {
        panic(fmt.Sprintf("%s 不能为 nil", name))
    }
}

// 3. 可选值用指针表示 (nil = 未设置)
type UserUpdate struct {
    Name *string `json:"name,omitempty"`  // nil = 不更新
    Age  *int    `json:"age,omitempty"`
}

// 4. 资源清理时处理 nil
type Resource struct {
    conn *sql.Conn
}

func (r *Resource) Close() error {
    if r == nil || r.conn == nil {
        return nil
    }
    return r.conn.Close()
}

// ========== nil vs 空值选择指南 ==========
// nil:     "不存在", "未设置", "空集合"
// 零值:    "存在但为零"
// 空切片:  "存在但为空集合" (JSON [])
// nil 切片: "空集合" (JSON null)

// JSON 反序列化:
// "field": null     → nil 指针字段
// "field": 0       → 零值 int
// "field": ""      → 空字符串
// 字段不存在       → 零值 (非指针) 或 nil (指针)
```


> **Note:** 💡 nil 与零值要点: 每种类型有明确零值 (int=0, string="", 指针=nil); nil 切片可 append 但不能索引赋值; nil map 可读不可写; nil channel 在 select 中永远阻塞; nil 接口 = (nil, nil), 赋 nil 指针后接口 ≠ nil; reflect.ValueOf(i).IsNil() 检查接口内指针; nil 接收者可调用方法并处理; 空切片 vs nil 切片 JSON 输出不同 ([] vs null); bytes.Buffer/sync.Mutex 零值可用。


## 练习


<!-- Converted from: 15_Go nil 与零值全面理解.html -->
