# Goroutine 与 Channel

## go 关键字

Go 使用 `go` 关键字启动 goroutine，这是一种轻量级协程，由 Go 运行时调度，而非操作系统线程。一个 goroutine 仅占用几 KB 栈空间，可动态增长。

### 基本用法

```go
package main

import (
    "fmt"
    "time"
)

func sayHello(name string) {
    fmt.Printf("Hello, %s!\n", name)
}

func main() {
    // 启动 goroutine
    go sayHello("World")

    // 匿名函数启动 goroutine
    go func() {
        fmt.Println("匿名 goroutine 执行")
    }()

    time.Sleep(100 * time.Millisecond) // 等待 goroutine 完成
}
```

### 注意事项：循环变量捕获

```go
package main

import (
    "fmt"
    "sync"
)

func main() {
    // 错误示范：循环变量捕获问题
    for i := 0; i < 5; i++ {
        go func() {
            fmt.Println(i) // Go 1.22 之前，所有 goroutine 可能打印相同的值
        }()
    }

    // 正确做法 1：通过参数传递
    for i := 0; i < 5; i++ {
        go func(val int) {
            fmt.Println(val)
        }(i)
    }

    // 正确做法 2：使用局部变量
    for i := 0; i < 5; i++ {
        i := i // Go 1.22 之前需要这一行
        go func() {
            fmt.Println(i)
        }()
    }

    var wg sync.WaitGroup
    wg.Add(5)
    for i := 0; i < 5; i++ {
        go func(val int) {
            defer wg.Done()
            fmt.Println(val)
        }(i)
    }
    wg.Wait()
}
```

## GMP 调度模型详解

Go 运行时使用 GMP 模型调度 goroutine，这是理解 Go 并发性能的关键。

### GMP 三组件

```
┌───────────────────────────────────────────────────────────────────┐
│                      GMP 调度模型                                  │
│                                                                   │
│  G (Goroutine)     P (Processor)      M (Machine/OS Thread)      │
│  ┌───────────┐     ┌───────────┐      ┌───────────┐              │
│  │ 栈 (2-8KB)│     │ 本地队列   │      │ 系统线程   │              │
│  │ PC 寄存器  │     │ M 绑定    │      │ 执行栈    │              │
│  │ goroutine │     │ GOMAXPROCS│      │ 持有 P    │              │
│  │ ID        │     │ 数量      │      │           │              │
│  └───────────┘     └───────────┘      └───────────┘              │
│       │                  │                   │                    │
│       │    ┌─────────────┼───────────────────┘                    │
│       │    │             │                                        │
│       ▼    ▼             ▼                                        │
│   G 在 P 的本地队列等待 → M 绑定 P → M 执行 G 的代码              │
└───────────────────────────────────────────────────────────────────┘
```

- **G (Goroutine)**：每个 goroutine 是一个 G 结构体，包含栈（初始 2KB，可动态增长到 1GB）、程序计数器（PC）、状态等。
- **P (Processor)**：逻辑处理器，数量由 `GOMAXPROCS` 决定（默认 = CPU 核心数）。每个 P 维护一个本地 goroutine 队列。
- **M (Machine)**：操作系统线程。M 必须绑定一个 P 才能执行 G。

### 调度流程

```
                     全局队列 (Global Queue)
                    ┌───┬───┬───┬───┬───┐
                    │ G │ G │ G │ G │ G │ ...
                    └───┴───┴───┴───┴───┘
                         │       │
              ┌──────────┘       └──────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ P0              │           │ P1              │
    │ 本地队列: [G1,G2]│           │ 本地队列: [G3,G4]│
    │ 绑定 M0         │           │ 绑定 M1         │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ M0 (OS Thread)  │           │ M1 (OS Thread)  │
    │ 正在执行 G1      │           │ 正在执行 G3      │
    └─────────────────┘           └─────────────────┘
```

### 调度时机

goroutine 在以下情况会被调度器切换：

1. **系统调用**：goroutine 进入阻塞式系统调用（如文件 IO），M 会释放 P，让其他 M 绑定该 P 继续执行队列中的 G。
2. **channel 操作**：发送/接收阻塞时，G 被挂起。
3. **runtime.Gosched()**：主动让出 CPU。
4. **函数调用**：Go 在函数调用时检查栈增长和抢占标记。
5. **垃圾回收**：STW 阶段需要所有 goroutine 暂停。

### 工作窃取（Work Stealing）

当一个 P 的本地队列为空时，它会按以下顺序查找工作：

```
1. 从本地队列取 G          → 有则执行
2. 从全局队列取 G          → 有则执行（每 61 次检查一次全局队列）
3. 从网络轮询器取 G        → 有则执行
4. 随机从其他 P 偷取一半 G  → 工作窃取
5. 都没有 → M 休眠或自旋
```

```
P0 队列空          P1 队列: [G1, G2, G3, G4]
    │                     │
    │  窃取一半            │
    └────────────────────►│
                          ▼
P0 队列: [G1, G2]  P1 队列: [G3, G4]
```

### 栈管理

goroutine 的栈是动态增长的：

```
Go 1.3 之前: 分段栈 (Segmented Stack)
  - 栈由多个不连续的段组成
  - 问题: "hot split"——在栈边界频繁的函数调用导致反复分配/释放栈段

Go 1.4+: 连续栈 (Contiguous Stack)
  - 当栈空间不足时，分配一个 2 倍大的新栈
  - 将旧栈内容拷贝到新栈
  - 所有指向栈上变量的指针被更新（通过编译器生成的指针信息）

初始栈: 2 KB
最大栈: 1 GB (64位系统)

栈增长检查: 编译器在函数入口插入检查代码
  if 栈空间不足:
      调用 runtime.morestack_noctxt()
      分配新栈 → 拷贝 → 更新指针 → 继续执行
```

### GOMAXPROCS 调优

```go
package main

import (
    "fmt"
    "runtime"
    "time"
)

func main() {
    // 查看当前 GOMAXPROCS
    fmt.Println("GOMAXPROCS:", runtime.GOMAXPROCS(0))

    // 设置为 4
    runtime.GOMAXPROCS(4)

    // 在容器环境中，Go 1.19+ 自动感知 cgroup CPU 限制
    // 通过 runtime/debug.SetMemoryLimit 可以设置 GC 内存目标

    // 获取 goroutine 数量
    fmt.Println("NumGoroutine:", runtime.NumGoroutine())

    // 获取 CPU 核心数
    fmt.Println("NumCPU:", runtime.NumCPU())
}
```

**调优原则**：
- CPU 密集型：`GOMAXPROCS = NumCPU`（默认值通常最优）
- IO 密集型：`GOMAXPROCS` 可以大于 NumCPU，因为 goroutine 大部分时间在等待 IO
- 容器环境：Go 1.19+ 自动读取 cgroup CPU 配额

## 完整工程级示例：并发 HTTP 服务器

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "sync"
    "time"
)

// 请求上下文，携带请求级数据
type RequestData struct {
    RequestID string
    StartTime time.Time
}

// 并发安全的请求统计
type Stats struct {
    mu       sync.RWMutex
    total    int64
    errors   int64
    latencies []time.Duration
}

func (s *Stats) Record(latency time.Duration, isError bool) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.total++
    if isError {
        s.errors++
    }
    s.latencies = append(s.latencies, latency)
}

func (s *Stats) Snapshot() map[string]interface{} {
    s.mu.RLock()
    defer s.mu.RUnlock()
    avg := time.Duration(0)
    if len(s.latencies) > 0 {
        var sum time.Duration
        for _, l := range s.latencies {
            sum += l
        }
        avg = sum / time.Duration(len(s.latencies))
    }
    return map[string]interface{}{
        "total_requests": s.total,
        "errors":         s.errors,
        "avg_latency_ms": avg.Milliseconds(),
    }
}

func main() {
    stats := &Stats{}
    var wg sync.WaitGroup

    // 模拟数据库查询（并发安全）
    var dbMutex sync.RWMutex
    db := make(map[string]string)
    db["user:1"] = "Alice"
    db["user:2"] = "Bob"

    mux := http.NewServeMux()

    // 处理用户查询请求
    mux.HandleFunc("/user", func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        id := r.URL.Query().Get("id")

        // 通过 WaitGroup 追踪活跃请求
        wg.Add(1)
        defer wg.Done()

        // 使用 goroutine 并行处理验证和查询
        type result struct {
            user string
            err  error
        }
        ch := make(chan result, 1)

        go func() {
            dbMutex.RLock()
            user, ok := db["user:"+id]
            dbMutex.RUnlock()
            if !ok {
                ch <- result{"", fmt.Errorf("user not found")}
                return
            }
            ch <- result{user, nil}
        }()

        res := <-ch
        latency := time.Since(start)

        if res.err != nil {
            stats.Record(latency, true)
            http.Error(w, res.err.Error(), http.StatusNotFound)
            return
        }

        stats.Record(latency, false)
        json.NewEncoder(w).Encode(map[string]string{
            "id":   id,
            "name": res.user,
        })
    })

    // 统计端点
    mux.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
        json.NewEncoder(w).Encode(stats.Snapshot())
    })

    server := &http.Server{
        Addr:    ":8080",
        Handler: mux,
    }

    // 优雅关闭
    go func() {
        log.Println("服务器启动在 :8080")
        if err := server.ListenAndServe(); err != http.ErrServerClosed {
            log.Fatalf("服务器错误: %v", err)
        }
    }()

    // 等待信号后优雅关闭
    // shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    // defer cancel()
    // server.Shutdown(shutdownCtx)
    // wg.Wait()

    fmt.Println("并发 HTTP 服务器已启动在 :8080")
    select {} // 阻塞
}
```

## 无缓冲 Channel

无缓冲 channel 的发送和接收操作是同步的，发送方会阻塞直到接收方准备好。

```
发送方 ──→ ┌──────────┐ ──→ 接收方
           │  无缓冲   │
           │ (同步阻塞) │
           └──────────┘
```

```go
package main

import (
    "fmt"
)

func main() {
    ch := make(chan string) // 无缓冲 channel

    go func() {
        ch <- "数据" // 发送阻塞，直到有人接收
        fmt.Println("发送完成")
    }()

    msg := <-ch // 接收数据
    fmt.Println("收到:", msg)
}
```

### 应用场景：信号通知

```go
package main

import (
    "fmt"
    "time"
)

func worker(done chan bool) {
    fmt.Println("工作开始...")
    time.Sleep(time.Second)
    fmt.Println("工作完成")
    done <- true // 通知主线程
}

func main() {
    done := make(chan bool)
    go worker(done)

    <-done // 等待工作完成
    fmt.Println("主线程退出")
}
```

### Channel 的底层实现

Channel 在 Go 运行时中由 `hchan` 结构体实现：

```
type hchan struct {
    qcount   uint           // 队列中的元素数量
    dataqsiz uint           // 环形队列的容量（0 = 无缓冲）
    buf      unsafe.Pointer // 环形队列指针
    elemsize uint16         // 元素大小
    closed   uint32         // 关闭标记
    sendx    uint           // 发送索引（环形队列）
    recvx    uint           // 接收索引
    recvq    waitq          // 等待接收的 goroutine 队列
    sendq    waitq          // 等待发送的 goroutine 队列
    lock     mutex          // 互斥锁
}
```

```
有缓冲 channel (cap=3, len=2):

  hchan
  ┌───────────────────────────────────┐
  │ qcount=2, dataqsiz=3             │
  │ buf ──────────────┐               │
  │ sendx=2, recvx=0  │               │
  │ lock              ▼               │
  │ recvq: []  ┌───┬───┬───┐         │
  │ sendq: []  │ A │ B │   │         │
  │            └───┴───┴───┘         │
  │             recvx↑    ↑sendx      │
  └───────────────────────────────────┘

  环形队列: recvx 指向下一个要读取的位置
            sendx 指向下一个要写入的位置
```

**无缓冲 channel 的工作流程**：

```
发送方 (G1)                      接收方 (G2)
    │                                │
    ▼                                ▼
ch <- data                     <-ch
    │                                │
    ├─ lock                         ├─ lock
    ├─ recvq 非空? → 直接拷贝给 G2   ├─ sendq 非空? → 直接从 G1 拷贝
    ├─ 否则: G1 入 sendq, 挂起      ├─ 否则: G2 入 recvq, 挂起
    └─ unlock                       └─ unlock
```

关键优化：当直接配对时（一个发送者恰好有一个接收者等待），数据直接从发送者栈拷贝到接收者栈，无需经过环形缓冲区。这被称为 "handoff" 优化。

## 有缓冲 Channel

有缓冲 channel 允许在缓冲区未满时非阻塞发送，缓冲区非空时非阻塞接收。

```
发送方 ──→ ┌───┬───┬───┬───┐ ──→ 接收方
           │ 1 │ 2 │ 3 │   │
           └───┴───┴───┴───┘
           容量为 4 的缓冲区
```

```go
package main

import "fmt"

func main() {
    // 创建容量为 3 的有缓冲 channel
    ch := make(chan int, 3)

    // 发送不会阻塞（缓冲区未满）
    ch <- 1
    ch <- 2
    ch <- 3

    // 接收数据
    fmt.Println(<-ch) // 1
    fmt.Println(<-ch) // 2
    fmt.Println(<-ch) // 3

    // 查看缓冲区状态
    ch2 := make(chan int, 5)
    ch2 <- 10
    ch2 <- 20
    fmt.Println("长度:", len(ch2)) // 2
    fmt.Println("容量:", cap(ch2)) // 5
}
```

### 缓冲区大小选择指南

```
场景                        推荐缓冲大小        原因
──────────────────────────────────────────────────────────────────
信号通知                    0 (无缓冲)         需要同步确认
生产者-消费者（速率匹配）    0 或小值           降低延迟
生产者-消费者（速率不匹配）  适当缓冲           吸收突发流量
扇出模式                    0                  反压机制
批量处理                    任务数             一次性发送所有任务
```

## select 语句

`select` 让 goroutine 同时等待多个 channel 操作，类似 switch 但专用于 channel。

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(100 * time.Millisecond)
        ch1 <- "来自 ch1"
    }()

    go func() {
        time.Sleep(200 * time.Millisecond)
        ch2 <- "来自 ch2"
    }()

    // select 会等待第一个就绪的 case
    select {
    case msg := <-ch1:
        fmt.Println(msg)
    case msg := <-ch2:
        fmt.Println(msg)
    }
}
```

### 超时与 default

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch := make(chan string)

    // 超时控制
    select {
    case msg := <-ch:
        fmt.Println("收到:", msg)
    case <-time.After(2 * time.Second):
        fmt.Println("超时!")
    }

    // 非阻塞操作（default 分支）
    select {
    case msg := <-ch:
        fmt.Println("收到:", msg)
    default:
        fmt.Println("channel 暂无数据")
    }
}
```

### select 的公平性问题

select 在多个 case 同时就绪时，会**随机**选择一个执行（而非按代码顺序）：

```go
package main

import "fmt"

func main() {
    ch1 := make(chan int, 1)
    ch2 := make(chan int, 1)

    ch1 <- 1
    ch2 <- 2

    // 两个 case 都就绪，随机选择
    select {
    case v := <-ch1:
        fmt.Println("ch1:", v)
    case v := <-ch2:
        fmt.Println("ch2:", v)
    }
}
```

这一随机性是为了防止饥饿——如果总是选择第一个 case，后续的 case 可能永远得不到执行。

### select 实现竞态超时模式

```go
package main

import (
    "context"
    "fmt"
    "math/rand"
    "time"
)

// 带重试的 HTTP 请求模拟
func fetchWithRetry(ctx context.Context, url string, maxRetries int) (string, error) {
    for attempt := 0; attempt < maxRetries; attempt++ {
        resultCh := make(chan string, 1)
        errCh := make(chan error, 1)

        go func() {
            // 模拟网络请求
            delay := time.Duration(rand.Intn(500)) * time.Millisecond
            time.Sleep(delay)
            if rand.Float32() < 0.3 {
                errCh <- fmt.Errorf("请求失败 (attempt %d)", attempt)
            } else {
                resultCh <- fmt.Sprintf("%s 的数据", url)
            }
        }()

        select {
        case result := <-resultCh:
            return result, nil
        case err := <-errCh:
            fmt.Printf("重试: %v\n", err)
        case <-ctx.Done():
            return "", ctx.Err()
        }
    }
    return "", fmt.Errorf("达到最大重试次数")
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
    defer cancel()

    result, err := fetchWithRetry(ctx, "https://api.example.com", 5)
    if err != nil {
        fmt.Println("最终失败:", err)
    } else {
        fmt.Println("成功:", result)
    }
}
```

## WaitGroup

`WaitGroup` 用于等待一组 goroutine 全部完成。

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func fetchURL(url string, wg *sync.WaitGroup) {
    defer wg.Done() // 完成后计数减 1
    time.Sleep(time.Duration(len(url)) * time.Millisecond)
    fmt.Printf("已抓取: %s\n", url)
}

func main() {
    var wg sync.WaitGroup
    urls := []string{
        "https://example.com",
        "https://golang.org",
        "https://github.com",
    }

    for _, url := range urls {
        wg.Add(1) // 每启动一个 goroutine，计数加 1
        go fetchURL(url, &wg)
    }

    wg.Wait() // 等待所有 goroutine 完成
    fmt.Println("全部抓取完成")
}
```

### WaitGroup 的底层实现

```
type WaitGroup struct {
    noCopy noCopy
    state1 [3]uint32  // 高32位: waiter 数量, 低32位: 计数器
                      // 或者使用信号量
}

Add(delta):
  原子地将 delta 加到计数器上
  如果计数器变为 0，唤醒所有等待的 goroutine

Done():
  Add(-1) 的简写

Wait():
  如果计数器 > 0:
    增加 waiter 计数
    通过信号量休眠
  被唤醒后返回
```

**常见错误**：

```go
// 错误: 在 goroutine 内部调用 Add
for _, url := range urls {
    go func() {
        wg.Add(1) // 可能导致 Wait() 在所有 Add 之前返回
        defer wg.Done()
        fetchURL(url)
    }()
}
wg.Wait() // 可能在任何 Add 之前就返回了

// 正确: 在启动 goroutine 之前调用 Add
for _, url := range urls {
    wg.Add(1)
    go func(u string) {
        defer wg.Done()
        fetchURL(u)
    }(url)
}
wg.Wait()
```

## Context

`context` 包用于在 goroutine 之间传递取消信号、超时和截止时间。

```go
package main

import (
    "context"
    "fmt"
    "time"
)

func longRunningTask(ctx context.Context) string {
    select {
    case <-time.After(3 * time.Second):
        return "任务完成"
    case <-ctx.Done():
        return "任务被取消: " + ctx.Err().Error()
    }
}

func main() {
    // 1. WithTimeout: 超时自动取消
    ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
    defer cancel()

    result := longRunningTask(ctx)
    fmt.Println(result) // 任务被取消: context deadline exceeded

    // 2. WithCancel: 手动取消
    ctx2, cancel2 := context.WithCancel(context.Background())
    go func() {
        time.Sleep(500 * time.Millisecond)
        cancel2() // 手动取消
    }()
    result2 := longRunningTask(ctx2)
    fmt.Println(result2)

    // 3. WithValue: 传递请求范围的数据
    ctx3 := context.WithValue(context.Background(), "requestID", "abc-123")
    processRequest(ctx3)
}

func processRequest(ctx context.Context) {
    requestID := ctx.Value("requestID")
    fmt.Println("处理请求:", requestID)
}
```

### Context 的底层结构

```
Context 是接口，有四个具体实现:

context.Background()  ──→ emptyCtx (根节点，永不取消)
       │
       ├── WithCancel ──→ cancelCtx (可手动取消)
       │       │
       │       └── WithCancel ──→ cancelCtx (子节点)
       │
       ├── WithTimeout ──→ timerCtx (cancelCtx + 定时器)
       │       │
       │       └── 自动在超时后调用 cancel
       │
       └── WithValue ──→ valueCtx (存键值对)

取消传播是树状的:
  父 context 取消 → 所有子 context 自动取消
  子 context 取消 → 不影响父 context
```

### Context 的最佳实践

```go
// 1. Context 应作为第一个参数传递
func ProcessData(ctx context.Context, data []byte) error {
    // ...
    return nil
}

// 2. 不要将 Context 存在结构体中（除非该结构体本身就是一个请求）
// 3. 不要传递 nil Context，使用 context.TODO() 或 context.Background()
// 4. WithValue 只用于请求范围的数据（如 trace ID），不要用于传递可选参数
// 5. 同一个 Context 可以传递给多个 goroutine，取消信号会同时传播
```

### Context 在 HTTP 服务器中的实际使用

```go
func apiHandler(w http.ResponseWriter, r *http.Request) {
    // HTTP 请求自带 context，请求断开时自动取消
    ctx := r.Context()

    // 为下游调用设置超时
    ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
    defer cancel()

    // 并行调用多个微服务
    userCh := make(chan *User, 1)
    orderCh := make(chan *Order, 1)

    go func() { user, _ := fetchUser(ctx, userID); userCh <- user }()
    go func() { order, _ := fetchOrder(ctx, orderID); orderCh <- order }()

    select {
    case user := <-userCh:
        select {
        case order := <-orderCh:
            json.NewEncoder(w).Encode(Response{User: user, Order: order})
        case <-ctx.Done():
            http.Error(w, "timeout", http.StatusGatewayTimeout)
        }
    case <-ctx.Done():
        http.Error(w, "timeout", http.StatusGatewayTimeout)
    }
}
```

## Worker Pool 模式

Worker Pool 是 Go 中常用的并发模式，用固定数量的 goroutine 处理任务队列。

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type Job struct {
    ID   int
    Data string
}

type Result struct {
    JobID int
    Value string
}

func worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    for job := range jobs {
        fmt.Printf("Worker %d 处理任务 %d\n", id, job.ID)
        time.Sleep(100 * time.Millisecond) // 模拟处理
        results <- Result{
            JobID: job.ID,
            Value: fmt.Sprintf("处理完成: %s", job.Data),
        }
    }
}

func main() {
    const numWorkers = 3
    const numJobs = 10

    jobs := make(chan Job, numJobs)
    results := make(chan Result, numJobs)

    // 启动 worker
    var wg sync.WaitGroup
    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }

    // 发送任务
    for j := 1; j <= numJobs; j++ {
        jobs <- Job{ID: j, Data: fmt.Sprintf("job-%d", j)}
    }
    close(jobs) // 关闭 channel，通知 worker 没有更多任务

    // 在另一个 goroutine 中等待所有 worker 完成，然后关闭 results
    go func() {
        wg.Wait()
        close(results)
    }()

    // 收集结果
    for result := range results {
        fmt.Printf("结果: 任务 %d -> %s\n", result.JobID, result.Value)
    }
}
```

### 动态 Worker Pool

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// 动态 Worker Pool: 支持优雅关闭和动态扩缩容
type DynamicPool struct {
    mu        sync.RWMutex
    workers   int
    jobs      chan func()
    wg        sync.WaitGroup
    ctx       context.Context
    cancel    context.CancelFunc
}

func NewDynamicPool(initialWorkers int) *DynamicPool {
    ctx, cancel := context.WithCancel(context.Background())
    p := &DynamicPool{
        workers: initialWorkers,
        jobs:    make(chan func(), 100),
        ctx:     ctx,
        cancel:  cancel,
    }
    for i := 0; i < initialWorkers; i++ {
        p.addWorker(i)
    }
    return p
}

func (p *DynamicPool) addWorker(id int) {
    p.wg.Add(1)
    go func() {
        defer p.wg.Done()
        for {
            select {
            case job, ok := <-p.jobs:
                if !ok {
                    return
                }
                job()
            case <-p.ctx.Done():
                return
            }
        }
    }()
}

func (p *DynamicPool) Submit(job func()) {
    select {
    case p.jobs <- job:
    case <-p.ctx.Done():
        panic("提交到已关闭的 pool")
    }
}

func (p *DynamicPool) Shutdown() {
    p.cancel()
    close(p.jobs)
    p.wg.Wait()
}

func main() {
    pool := NewDynamicPool(4)

    // 提交 20 个任务
    for i := 0; i < 20; i++ {
        i := i
        pool.Submit(func() {
            fmt.Printf("任务 %d 执行中 (goroutine: %d)\n", i, runtime.NumGoroutine())
            time.Sleep(50 * time.Millisecond)
        })
    }

    pool.Shutdown()
    fmt.Println("Pool 已关闭")
}
```

## Channel 的关闭与方向

### 单向 Channel

```go
package main

import "fmt"

// 只能发送的 channel
func producer(out chan<- int) {
    for i := 0; i < 5; i++ {
        out <- i
    }
    close(out)
}

// 只能接收的 channel
func consumer(in <-chan int) {
    for val := range in {
        fmt.Println("消费:", val)
    }
}

func main() {
    ch := make(chan int, 3)
    go producer(ch)
    consumer(ch)
}
```

### 关闭 channel 的规则

```go
package main

import "fmt"

func main() {
    ch := make(chan int, 5)

    // 发送方关闭 channel
    go func() {
        for i := 0; i < 5; i++ {
            ch <- i
        }
        close(ch) // 由发送方关闭
    }()

    // 接收方检测 channel 是否关闭
    for {
        val, ok := <-ch
        if !ok {
            fmt.Println("channel 已关闭")
            break
        }
        fmt.Println("收到:", val)
    }

    // 更简洁的方式：使用 range
    ch2 := make(chan int, 3)
    go func() {
        ch2 <- 1
        ch2 <- 2
        ch2 <- 3
        close(ch2)
    }()
    for val := range ch2 {
        fmt.Println(val)
    }
}
```

### Channel 关闭的黄金法则

```
1. 只有发送方应该关闭 channel
2. 不要关闭有接收者的 channel（除非你确定不再发送）
3. 关闭已关闭的 channel 会 panic
4. 向已关闭的 channel 发送数据会 panic
5. 从已关闭的 channel 接收数据：立即返回零值和 false
6. 多个发送者时，使用额外的 channel 协调关闭
```

### 多个发送者的安全关闭模式

```go
package main

import (
    "fmt"
    "sync"
)

func main() {
    dataCh := make(chan int)
    doneCh := make(chan struct{})
    var wg sync.WaitGroup

    // 3 个发送者
    for i := 0; i < 3; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for j := 0; j < 5; j++ {
                select {
                case dataCh <- id*10 + j:
                case <-doneCh:
                    return
                }
            }
        }(i)
    }

    // 单独的 goroutine 等待所有发送者完成后关闭 dataCh
    go func() {
        wg.Wait()
        close(dataCh)
    }()

    // 接收者
    for val := range dataCh {
        fmt.Println("收到:", val)
    }
    close(doneCh)
}
```

## 常见陷阱详解

### 陷阱 1：Goroutine 泄漏

```go
// BUG: goroutine 永远不会退出
func leakyFunction() {
    ch := make(chan int)
    go func() {
        result := expensiveComputation()
        ch <- result // 如果没有人接收，这个 goroutine 永远阻塞
    }()
    // 如果这里发生 early return，goroutine 就泄漏了
}

// 修复：使用带缓冲的 channel 或 context
func fixedFunction(ctx context.Context) int {
    ch := make(chan int, 1) // 缓冲为1，即使没人接收也能写入
    go func() {
        result := expensiveComputation()
        select {
        case ch <- result:
        case <-ctx.Done():
            return // context 取消时退出
        }
    }()

    select {
    case result := <-ch:
        return result
    case <-ctx.Done():
        return 0
    }
}
```

### 陷阱 2：误用 WaitGroup

```go
// BUG: goroutine 中调用 Add 可能导致 Wait 提前返回
func buggyCode() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        go func(n int) {
            wg.Add(1) // 可能在 Wait() 之后才执行
            defer wg.Done()
            process(n)
        }(i)
    }
    wg.Wait() // 可能在所有 Add 之前就返回
}

// 修复：在启动 goroutine 之前调用 Add
func fixedCode() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1) // 先 Add
        go func(n int) {
            defer wg.Done()
            process(n)
        }(i)
    }
    wg.Wait()
}
```

### 陷阱 3：向已关闭的 channel 发送

```go
// BUG: panic: send on closed channel
func buggySend() {
    ch := make(chan int)
    go func() {
        time.Sleep(time.Second)
        ch <- 1 // panic!
    }()
    close(ch)
}

// 修复：由发送方负责关闭，且确保不再发送后才关闭
func fixedSend() {
    ch := make(chan int)
    go func() {
        defer close(ch) // 发送方关闭
        for i := 0; i < 10; i++ {
            ch <- i
        }
    }()
    for val := range ch {
        fmt.Println(val)
    }
}
```

### 陷阱 4：死锁 — 所有 goroutine 都在等待

```go
// BUG: fatal error: all goroutines are asleep - deadlock!
func deadlockDemo() {
    ch := make(chan int) // 无缓冲
    ch <- 1             // 发送阻塞，但没有其他 goroutine 接收
    fmt.Println(<-ch)
}

// 修复：在单独的 goroutine 中发送
func noDeadlock() {
    ch := make(chan int)
    go func() {
        ch <- 1
    }()
    fmt.Println(<-ch)
}
```

## 性能实测数据

```
操作                                耗时 (Go 1.21, AMD64)    说明
──────────────────────────────────────────────────────────────────────
goroutine 创建+销毁                 ~200-300 ns             对比 pthread ~15 μs
channel 发送/接收 (无竞争)          ~70-100 ns              直接拷贝
channel 发送/接收 (有缓冲)          ~100-150 ns             经过环形队列
select (2 个 case)                  ~50-80 ns               编译为跳转表
mutex.Lock (无竞争)                 ~17 ns                  原子操作
mutex.Lock (有竞争)                 ~200 ns - 2 μs          含系统调用
atomic.AddInt64 (无竞争)            ~8 ns                   单条 CPU 指令
context.WithCancel 创建             ~50 ns                  分配 cancelCtx
WaitGroup.Add(1)                    ~12 ns                  原子加

100 万 goroutine 内存占用:  ~2 GB (每个 goroutine 栈 ~2 KB)
100 万 OS 线程:             ~8 GB+ (每个线程栈 ~8 MB) + 无法创建
```

## 生产案例

### Kubernetes 中的 goroutine 使用

Kubernetes 的核心组件大量使用 goroutine 处理并发：

- **kubelet**：每个 Pod 对应一个 goroutine 池，管理容器生命周期
- **apiserver**：每个 HTTP 请求在独立的 goroutine 中处理，使用 `context.Context` 传递取消信号
- **scheduler**：使用 work queue + goroutine pool 实现调度循环
- **etcd watch**：每个 watch 连接对应一个 goroutine，通过 channel 推送事件

Kubernetes 的 informer 框架是一个经典的 Go 并发模式：list-watch 机制在后台 goroutine 中维护本地缓存，通过 event handler 回调通知消费者。

### Docker 的并发模式

Docker daemon 使用 goroutine pool 处理容器操作请求，每个容器的 IO 流（stdout/stderr）各用一个 goroutine 复制数据，防止一个流的阻塞影响另一个。

## 总结

| 概念 | 用途 | 特点 |
|------|------|------|
| go | 启动协程 | 轻量级，由 GMP 运行时调度 |
| 无缓冲 channel | 同步通信 | 发送接收配对阻塞，实现 rendezvous |
| 有缓冲 channel | 异步通信 | 可设置容量，吸收突发 |
| select | 多路复用 | 监听多个 channel，随机公平 |
| WaitGroup | 等待完成 | 计数器模式，不能重用 |
| context | 取消/超时 | 层级传播，协作取消 |
| Worker Pool | 并发处理 | 固定 goroutine 数量 + 任务队列 |
| GMP 模型 | 调度原理 | 工作窃取，M:N 调度 |
