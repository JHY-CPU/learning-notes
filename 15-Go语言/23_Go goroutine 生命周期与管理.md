# Go goroutine 生命周期与管理


## 🔄 Go goroutine 生命周期与管理


goroutine 状态模型、调度器 GMP 模型、goroutine 泄漏预防、优雅退出模式、panic 传播与 recover、GOMAXPROCS 控制。


## GMP 调度模型


```
// ========== GMP 调度模型 ==========
// Go 调度器管理 goroutine 的生命周期
//
// G (Goroutine): 协程, 包含栈/指令指针/状态
// M (Machine):    OS 线程, 执行 Go 代码
// P (Processor):  逻辑处理器, 本地 G 队列
//
// 调度流程:
// 1. G 在 P 的本地队列等待
// 2. M 从 P 获取 G 执行
// 3. G 阻塞时 (syscall), M 和 P 解绑
// 4. P 从全局队列或 other P 偷取 G (work stealing)

// GOMAXPROCS 控制同时可执行的 P 数量
// 默认 = CPU 核心数

func gomaxprocs() {
    // 查看/设置
    fmt.Println(runtime.GOMAXPROCS(0))  // 当前值

    // 设置: runtime.GOMAXPROCS(8)
    // - IO 密集型: 可大于 CPU 核心数
    // - CPU 密集型: = CPU 核心数通常最优
    // - 注意: 不是上限, 是并行度
}

// ========== goroutine 状态 ==========
// goroutine 生命周期:
// 创建 (Goroutine) → 可运行 (Runnable) → 运行中 (Running)
// → 阻塞 (Waiting: chan/syscall/lock) → 可运行 → ... → 结束 (Dead)

// == 创建 vs 运行 ==
// go func() 创建成功: goroutine 进入可运行队列
// 实际开始执行: 取决于调度器

// ========== goroutine 栈 ==========
// 初始栈: 2KB (可动态增长)
// 最大栈: 1GB (64位)
// 对比 OS 线程栈: 默认 8MB (1MB Mac)

func stackInfo() {
    // 查看当前 goroutine 数量
    fmt.Println(runtime.NumGoroutine())

    // 查看栈信息
    buf := make([]byte, 1024)
    n := runtime.Stack(buf, false)  // false = 仅当前
    fmt.Println(string(buf[:n]))
}

// ========== goroutine 退出 ==========
// goroutine 在以下情况退出:
// 1. 函数返回
// 2. 调用 runtime.Goexit()
// 3. main 函数返回 (所有 goroutine 强制终止)

func goroutineExit() {
    go func() {
        defer fmt.Println("goroutine 退出")
        fmt.Println("工作...")
        // 函数结束后 goroutine 退出
    }()

    go func() {
        defer fmt.Println("Goexit 触发 defer")
        runtime.Goexit()  // 优雅退出 goroutine (运行 defer)
        fmt.Println("这行不会执行")
    }()
    time.Sleep(time.Millisecond)
}
```


## goroutine 泄漏


```
// ========== goroutine 泄漏 ==========
// goroutine 启动后永远无法退出 → 内存泄漏

// ❌ 泄漏模式 1: channel 发送阻塞, 无接收者
func leakSend() {
    ch := make(chan int)
    go func() {
        ch <- 42  // 永远阻塞! 没有人接收
        fmt.Println("不会执行")
    }()
    // goroutine 永不会退出
}

// ❌ 泄漏模式 2: 从 channel 接收, 无发送者
func leakRecv() {
    ch := make(chan int)
    go func() {
        <-ch  // 永远阻塞! 没有人发送
    }()
}

// ❌ 泄漏模式 3: 忘记关闭 channel, range 不退出
func leakRange() {
    ch := make(chan int)
    go func() {
        for v := range ch {  // ch 永不关闭
            fmt.Println(v)
        }
    }()
    ch <- 1
    // 未 close(ch), goroutine 永远等待
}

// ❌ 泄漏模式 4: select 阻塞
func leakSelect() {
    done := make(chan struct{})
    go func() {
        select {
        case <-done:
        // 没有 default, 没有超时, done 永不关闭
        }
    }()
}

// ❌ 泄漏模式 5: ticker 未停止
func leakTicker() {
    ticker := time.NewTicker(time.Second)
    go func() {
        for range ticker.C {  // 永远循环
            fmt.Println("tick")
        }
    }()
    // 没有 ticker.Stop()
}

// ========== 防止泄漏 ==========
// ✅ 方案 1: 使用 context 控制生命周期
func safeGoroutine(ctx context.Context) {
    go func() {
        for {
            select {
            case <-ctx.Done():
                fmt.Println("退出")
                return
            default:
                // 工作...
            }
        }
    }()
}

// ✅ 方案 2: done channel 模式
func safeDone() {
    done := make(chan struct{})
    ch := make(chan int)

    go func() {
        defer fmt.Println("goroutine 退出")
        for {
            select {
            case v := <-ch:
                fmt.Println(v)
            case <-done:
                return
            }
        }
    }()

    ch <- 1
    close(done)  // 通知 goroutine 退出
    time.Sleep(time.Millisecond)
}

// ✅ 方案 3: 超时保护
func safeTimeout() {
    ch := make(chan int)
    go func() {
        select {
        case ch <- result():
        case <-time.After(5 * time.Second):
            fmt.Println("超时, goroutine 放弃")
        }
    }()
}

// ✅ 方案 4: ticker 正确停止
func safeTicker() {
    ticker := time.NewTicker(time.Second)
    done := make(chan struct{})

    go func() {
        defer ticker.Stop()
        for {
            select {
            case <-ticker.C:
                fmt.Println("tick")
            case <-done:
                fmt.Println("停止")
                return
            }
        }
    }()

    time.Sleep(3 * time.Second)
    close(done)
}
```


## panic 与 recover


```
// ========== goroutine 中的 panic ==========
// 未 recover 的 panic 会使整个进程崩溃!

// ❌ 一个 goroutine panic 导致整个程序退出
func panicCrashesAll() {
    go func() {
        panic("出错!")  // 整个程序退出
    }()
    time.Sleep(time.Second)
    fmt.Println("不可达")  // 不会执行
}

// ✅ 每个 goroutine 独立 recover
func safePanic() {
    go func() {
        defer func() {
            if r := recover(); r != nil {
                fmt.Println("goroutine 恢复:", r)
            }
        }()
        // 可能 panic 的代码
        panic("出错了")
    }()
    time.Sleep(time.Millisecond)
    fmt.Println("主 goroutine 继续")  // 正常执行
}

// ========== goroutine 池管理 ==========
// 控制 goroutine 数量, 防止资源耗尽

type GoPool struct {
    sem chan struct{}    // 信号量
    wg  sync.WaitGroup
}

func NewGoPool(maxSize int) *GoPool {
    return &GoPool{
        sem: make(chan struct{}, maxSize),
    }
}

func (p *GoPool) Go(fn func()) {
    p.sem <- struct{}{}  // 获取令牌 (满则阻塞)
    p.wg.Add(1)

    go func() {
        defer func() {
            <-p.sem   // 释放令牌
            p.wg.Done()
        }()
        fn()
    }()
}

func (p *GoPool) Wait() {
    p.wg.Wait()
}

// 使用:
// pool := NewGoPool(10)
// for i := 0; i < 100; i++ {
//     pool.Go(func() {
//         fmt.Println("work")
//     })
// }
// pool.Wait()

// ========== 追踪 goroutine ==========
// 使用 runtime.NumGoroutine 检测泄漏

func detectLeak() {
    before := runtime.NumGoroutine()
    // 执行可能泄漏的代码...
    after := runtime.NumGoroutine()

    if after > before {
        fmt.Printf("可能的 goroutine 泄漏: %d -> %d\n",
            before, after)
    }
}

// 生产环境: pprof 分析
// import _ "net/http/pprof"
// 访问 /debug/pprof/goroutine
```


> **Note:** 💡 goroutine 管理要点: GMP 模型 (Goroutine/Machine/Processor), P 数量由 GOMAXPROCS 控制; 初始栈 2KB 动态增长; goroutine 泄漏 5 种模式 (chan 发送/接收阻塞、range 未 close、select 阻塞、ticker 未停); ctx.Done()/done channel/超时 防止泄漏; goroutine 的 panic 会崩溃整个进程, 每个 goroutine 需独立 recover; goroutine 池用信号量限制并发; runtime.NumGoroutine() 检测泄漏; pprof 生产分析。


## 练习


<!-- Converted from: 23_Go goroutine 生命周期与管理.html -->
