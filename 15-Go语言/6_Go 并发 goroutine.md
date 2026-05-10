# Go 并发 goroutine


## ⚡ Go 并发 goroutine


goroutine 轻量线程、go 关键字、WaitGroup 同步、竞态条件、sync.Mutex/RWMutex、sync.Once、原子操作、GOMAXPROCS。


## goroutine 基础


```
// ========== goroutine ==========
// Go 的轻量级线程 (协程)
// 由 Go 运行时调度 (不是 OS 线程)
// 栈初始 2KB, 可动态增长

package main

import (
    "fmt"
    "sync"
    "time"
)

// ========== 启动 goroutine ==========
func sayHello() {
    fmt.Println("Hello from goroutine")
}

func main() {
    // 使用 go 关键字启动
    go sayHello()

    // 匿名函数
    go func() {
        fmt.Println("匿名 goroutine")
    }()

    // 带参数 (值复制, 安全)
    go func(name string) {
        fmt.Println("Hello,", name)
    }("Alice")

    // 等待 goroutine 执行
    time.Sleep(100 * time.Millisecond)
}
// 注意: main 返回时所有 goroutine 立即终止!

// ========== sync.WaitGroup ==========
// 等待一组 goroutine 完成

func main() {
    var wg sync.WaitGroup

    for i := 1; i <= 5; i++ {
        wg.Add(1)               // 计数器 +1
        go func(id int) {
            defer wg.Done()     // 计数器 -1
            fmt.Printf("任务 %d 执行中\n", id)
            time.Sleep(time.Second)
        }(i)
    }

    wg.Wait()                   // 等待所有完成 (阻塞)
    fmt.Println("所有任务完成")
}

// WaitGroup 注意:
// 1. Add 必须在 goroutine 外调用
// 2. Done 在 goroutine 结束处 (常用 defer)
// 3. Wait 阻塞直到计数器归零
// 4. 计数器不能为负

// ========== WaitGroup 模式 ==========
// 常见: 并发处理任务列表
func processBatch(items []int, workers int) {
    var wg sync.WaitGroup
    sem := make(chan struct{}, workers)  // 限制并发数

    for _, item := range items {
        wg.Add(1)
        go func(v int) {
            defer wg.Done()
            sem <- struct{}{}      // 获取令牌
            defer func() { <-sem }() // 释放令牌
            processItem(v)
        }(item)
    }

    wg.Wait()
}
```


## 竞态与互斥锁


```
// ========== 竞态条件 (Race Condition) ==========
// 多个 goroutine 同时读写同一变量

// ❌ 竞态示例:
var counter int

func main() {
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            counter++    // 非原子操作!
            wg.Done()
        }()
    }
    wg.Wait()
    fmt.Println(counter)    // 结果 < 1000!
}

// 检测竞态: go run -race main.go

// ========== sync.Mutex 互斥锁 ==========
type SafeCounter struct {
    mu     sync.Mutex
    count  int
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()         // 加锁
    c.count++            // 安全操作
    c.mu.Unlock()       // 解锁
}

func (c *SafeCounter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}

// ========== sync.RWMutex 读写锁 ==========
// 多读单写: 读不互斥, 写互斥

type SafeMap struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (m *SafeMap) Get(key string) interface{} {
    m.mu.RLock()            // 读锁 (可多个)
    defer m.mu.RUnlock()
    return m.data[key]
}

func (m *SafeMap) Set(key string, value interface{}) {
    m.mu.Lock()             // 写锁 (独占)
    defer m.mu.Unlock()
    m.data[key] = value
}

// RLock: 多个可同时读
// Lock:  只有一个能写 (读写都不行)

// ========== sync.Once ==========
// 只执行一次 (单例/初始化)

var (
    config     *Config
    configOnce sync.Once
)

func GetConfig() *Config {
    configOnce.Do(func() {
        config = loadConfig()    // 只执行一次
    })
    return config
}

// ========== 原子操作 ==========
// 无锁高并发计数器

import "sync/atomic"

var counter atomic.Int64

counter.Add(1)                // 原子 +1
value := counter.Load()       // 原子读取
counter.Store(42)             // 原子写入

// 旧版本:
// var counter int64
// atomic.AddInt64(&counter, 1)
```


## Channel 通道


```
// ========== Channel ==========
// goroutine 间通信的主要方式
// 类型: chan T (有缓冲/无缓冲)

func main() {
    // ========== 创建 Channel ==========
    ch1 := make(chan int)               // 无缓冲 (同步)
    ch2 := make(chan string, 10)        // 有缓冲 (异步)

    // ========== 发送与接收 ==========
    ch := make(chan int)

    // 发送: ch <- value
    // 接收: value := <-ch

    // 无缓冲 channel 示例:
    go func() {
        ch <- 42    // 发送 (阻塞直到接收)
    }()

    value := <-ch   // 接收 (阻塞直到发送)
    fmt.Println(value)  // 42

    // ========== 带缓冲 channel ==========
    bufCh := make(chan int, 3)

    bufCh <- 1     // 不阻塞 (缓冲区有空间)
    bufCh <- 2     // 不阻塞
    bufCh <- 3     // 不阻塞
    // bufCh <- 4  // 阻塞! 缓冲区满

    fmt.Println(<-bufCh)  // 1
    fmt.Println(<-bufCh)  // 2
    fmt.Println(<-bufCh)  // 3

    // ========== 关闭 Channel ==========
    jobs := make(chan int, 5)

    // 发送者关闭
    go func() {
        for i := 0; i < 5; i++ {
            jobs <- i
        }
        close(jobs)  // 关闭后不能再发送
    }()

    // 接收者用 range 遍历
    for job := range jobs {
        fmt.Println("收到任务:", job)
    }

    // 检查是否关闭:
    val, ok := <-jobs
    if !ok {
        fmt.Println("channel 已关闭")
    }

    // ========== 单向 Channel ==========
    // 限制 channel 方向, 增加类型安全

    // 只写: chan<- int
    // 只读: <-chan int

    func worker(jobs <-chan int, results chan<- int) {
        for job := range jobs {
            results <- job * 2
        }
    }
}

// ========== 并发工作池 ==========
func workerPool() {
    const numJobs = 10
    const numWorkers = 3

    jobs := make(chan int, numJobs)
    results := make(chan int, numJobs)

    // 启动 workers
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for job := range jobs {
                fmt.Printf("Worker %d 处理任务 %d\n", id, job)
                time.Sleep(time.Second)
                results <- job * 2
            }
        }(w)
    }

    // 发送任务
    for j := 0; j < numJobs; j++ {
        jobs <- j
    }
    close(jobs)     // 通知 worker 没有更多任务

    // 等待所有 worker 完成
    wg.Wait()
    close(results)

    // 收集结果
    for result := range results {
        fmt.Println("结果:", result)
    }
}
```


## select 多路复用


```
// ========== select ==========
// 同时等待多个 channel 操作
// 类似 switch 但用于 channel

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "消息 1"
    }()

    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "消息 2"
    }()

    select {
    case msg1 := <-ch1:
        fmt.Println("收到:", msg1)
    case msg2 := <-ch2:
        fmt.Println("收到:", msg2)
    case <-time.After(500 * time.Millisecond):
        fmt.Println("超时!")     // 500ms 超时
    default:
        fmt.Println("没有消息")   // 非阻塞
    }
}

// ========== select 典型用法 ==========

// 1. 超时控制
select {
case result := <-ch:
    fmt.Println(result)
case <-time.After(5 * time.Second):
    fmt.Println("操作超时")
}

// 2. 非阻塞收发
select {
case msg := <-ch:
    fmt.Println(msg)
default:
    fmt.Println("无消息, 不阻塞")
}

// 3. 退出信号
func run(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return    // 收到取消信号
        default:
            // 继续工作
        }
    }
}

// 4. 随机选择
// 多个 case 都满足时, select 随机选择

// ========== 并发模式: 超时 + 退出 ==========
func doWork(done <-chan struct{}) error {
    result := make(chan error, 1)

    go func() {
        // 执行耗时操作
        result <- longOperation()
    }()

    select {
    case err := <-result:
        return err
    case <-done:
        // 取消操作
        return fmt.Errorf("已取消")
    case <-time.After(30 * time.Second):
        return fmt.Errorf("超时")
    }
}
```


> **Note:** 💡 并发要点: go 关键字启动 goroutine; sync.WaitGroup 等待完成; Mutex/RWMutex 互斥; atomic 原子操作; sync.Once 单次初始化; channel 通信 (有缓冲/无缓冲); close + range 遍历; select 多路 + 超时 + 非阻塞; -race 检测竞态; GOMAXPROCS 控制并行。


## 练习


<!-- Converted from: 6_Go 并发 goroutine.html -->
