# Go select 高级用法


## 🔀 Go select 高级用法


select 机制与随机选择、超时控制、非阻塞通信、循环 select 模式、for-select 惯用法、动态 case 管理、select 与 context 集成。


## select 机制


```
// ========== select 核心规则 ==========
// select 像 switch 但用于 channel
// 规则:
// 1. 同时检查所有 case
// 2. 随机选择一个满足条件的执行
// 3. 没有满足条件的 → 执行 default (如果有)
// 4. 没有 default → 阻塞直到某个 case 满足

func selectMechanism() {
    ch1 := make(chan int)
    ch2 := make(chan int)

    go func() { ch1 <- 1 }()
    go func() { ch2 <- 2 }()

    time.Sleep(time.Millisecond)

    select {
    case v := <-ch1:
        fmt.Println("ch1:", v)
    case v := <-ch2:
        fmt.Println("ch2:", v)
    // 两个都可用! 随机选择
    }
}

// ========== 非阻塞通信 ==========
// default 分支使 select 不阻塞

func nonBlocking() {
    ch := make(chan int, 1)

    // 非阻塞发送
    select {
    case ch <- 42:
        fmt.Println("发送成功")
    default:
        fmt.Println("缓冲区满, 丢弃")
    }

    // 非阻塞接收
    select {
    case v := <-ch:
        fmt.Println("收到:", v)
    default:
        fmt.Println("无数据")
    }
}

// ========== 超时控制 ==========
// time.After 提供超时

func timeoutSelect() {
    ch := make(chan int)

    go func() {
        time.Sleep(3 * time.Second)
        ch <- 42
    }()

    select {
    case v := <-ch:
        fmt.Println("收到:", v)
    case <-time.After(1 * time.Second):
        fmt.Println("超时!")  // 1秒后就超时
    }
}

// ========== 循环 select ==========
// for + select 持续处理

func loopSelect() {
    ch1 := make(chan int)
    ch2 := make(chan int)
    done := make(chan struct{})

    // 生产者
    go func() {
        for i := 0; i < 5; i++ {
            ch1 <- i
        }
        close(done)
    }()

    // 消费者
    for {
        select {
        case v := <-ch1:
            fmt.Println("ch1:", v)
        case v := <-ch2:
            fmt.Println("ch2:", v)
        case <-done:
            fmt.Println("完成, 退出")
            return
        }
    }
}
```


## for-select 惯用法


```
// ========== for-select 的 3 种模式 ==========

// 模式 1: 发送/接收循环
func sendLoop(items []int) <-chan int {
    out := make(chan int)
    go func() {
        for _, item := range items {
            select {
            case out <- item:
            case <-time.After(time.Second):
                fmt.Println("发送超时")
                return
            }
        }
        close(out)
    }()
    return out
}

// 模式 2: 无限循环等待事件
func eventLoop() {
    ch := make(chan int)
    done := make(chan struct{})

    go func() {
        for {
            select {
            case v := <-ch:
                fmt.Println("事件:", v)
            case <-done:
                fmt.Println("停止")
                return
            }
        }
    }()

    ch <- 1
    ch <- 2
    close(done)
    time.Sleep(time.Millisecond)
}

// 模式 3: 工作循环
type Worker struct {
    jobs    <-chan Job
    results chan<- Result
    done    <-chan struct{}
}

func (w *Worker) Run() {
    for {
        select {
        case job, ok := <-w.jobs:
            if !ok {
                return  // channel 关闭
            }
            result := process(job)
            select {
            case w.results <- result:
            case <-w.done:
                return
            }
        case <-w.done:
            return
        }
    }
}

// ========== select + nil channel ==========
// nil channel 在 select 中永远不可选
// 动态启用/禁用 case

func dynamicCase() {
    ch1 := make(chan int)
    ch2 := make(chan int)
    done := make(chan struct{})

    go func() {
        ch1 <- 1
        ch1 <- 2
        ch1 <- 3
        close(ch1)
    }()

    go func() {
        time.Sleep(100 * time.Millisecond)
        ch2 <- 10
        ch2 <- 20
        close(ch2)
    }()

    // 用 nil channel 控制活跃度
    var activeChan chan int
    var activeVal int

    for {
        select {
        case v, ok := <-ch1:
            if ok {
                activeChan = ch2  // 收到 ch1 后启用 ch2
                activeVal = v
                fmt.Println("ch1:", v)
            }
        case v, ok := <-activeChan:
            if ok {
                fmt.Println("ch2:", v)
            }
        case <-done:
            return
        default:
            if activeChan == nil {
                fmt.Println("等待中...")
                time.Sleep(50 * time.Millisecond)
            }
        }
    }
}

// ========== select + context ==========
// 标准集成模式

func selectWithContext(ctx context.Context) {
    ch := make(chan int)

    go func() {
        // 长时间操作
        time.Sleep(2 * time.Second)
        ch <- 42
    }()

    select {
    case v := <-ch:
        fmt.Println("结果:", v)
    case <-ctx.Done():
        fmt.Println("取消:", ctx.Err())
    }
}
```


## select 高级模式


```
// ========== 1. 心跳 (Heartbeat) ==========
// 定期发送状态信号

func heartbeat(done <-chan struct{}, pulseInterval time.Duration) <-chan struct{} {
    heartbeat := make(chan struct{})
    go func() {
        defer close(heartbeat)
        ticker := time.NewTicker(pulseInterval)
        defer ticker.Stop()

        for {
            select {
            case <-done:
                return
            case <-ticker.C:
                select {
                case heartbeat <- struct{}{}:
                default:
                    // 无人监听心跳, 不阻塞
                }
            }
        }
    }()
    return heartbeat
}

// ========== 2. 复制请求 (Replicated Request) ==========
// 向多个服务发送相同请求, 用最快的结果

func replicatedRequest(ctx context.Context, endpoints []string) (string, error) {
    resultCh := make(chan string, len(endpoints))

    for _, endpoint := range endpoints {
        go func(url string) {
            resp, err := http.Get(url)
            if err != nil {
                return
            }
            defer resp.Body.Close()
            body, _ := io.ReadAll(resp.Body)
            resultCh <- string(body[:100])
        }(endpoint)
    }

    select {
    case result := <-resultCh:
        return result, nil
    case <-ctx.Done():
        return "", ctx.Err()
    }
}

// ========== 3. 超时 + 重试 ==========
func retryWithTimeout(ctx context.Context, fn func() error, maxRetries int) error {
    var err error

    for i := 0; i < maxRetries; i++ {
        // 在 select 中执行
        done := make(chan struct{}, 1)
        go func() {
            err = fn()
            done <- struct{}{}
        }()

        select {
        case <-done:
            if err == nil {
                return nil
            }
            // 继续重试
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(time.Duration(1<
💡 select 要点: 所有 case 同时检查, 就绪的随机选; default 实现非阻塞; for + select 是主流模式; nil channel 在 select 中永不就绪 (用于动态开关 case); time.After 做超时要防止泄漏 (使用 time.NewTimer); heartbeat 模式定期发送信号; replicated request 取最快响应; 速率限制 ticker; select 与 context 集成是标准做法; case 过多有性能开销。
```


## 练习


## 练习


<!-- Converted from: 25_Go select 高级用法.html -->
