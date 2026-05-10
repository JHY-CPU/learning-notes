# Go pprof 与调试


## 🔍 Go pprof 与调试


net/http/pprof 端点、runtime/pprof 编程、CPU/内存/goroutine/阻塞/锁分析、go tool trace、dlv 调试器、生产实践。


## net/http/pprof


```
// ========== 集成 pprof ==========
// 导入即可自动注册端点
import _ "net/http/pprof"

func main() {
    // 方式 1: 默认 mux (可能暴露其他路由)
    go func() {
        log.Println(http.ListenAndServe("localhost:6060", nil))
    }()

    // 方式 2: 独立 mux (推荐, 仅 pprof)
    pprofMux := http.NewServeMux()
    pprofMux.HandleFunc("/debug/pprof/", pprof.Index)
    pprofMux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
    pprofMux.HandleFunc("/debug/pprof/profile", pprof.Profile)
    pprofMux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
    pprofMux.HandleFunc("/debug/pprof/trace", pprof.Trace)

    go func() {
        log.Println(http.ListenAndServe(":6060", pprofMux))
    }()

    // 业务服务器
    http.ListenAndServe(":8080", mux)
}

// ========== Gin 集成 pprof ==========
// import "github.com/gin-contrib/pprof"

func ginWithPprof() {
    r := gin.Default()
    pprof.Register(r)  // 注册 /debug/pprof 路由
    r.Run(":8080")
}

// 或者手动注册:
// r.GET("/debug/pprof/*any", gin.WrapH(http.DefaultServeMux))

// ========== pprof 端点一览 ==========
// /debug/pprof/              — 概览页面
// /debug/pprof/goroutine     — goroutine 栈 (当前运行)
// /debug/pprof/heap          — 堆内存 (当前活跃对象)
// /debug/pprof/alloc         — 累积分配 (全生命周期)
// /debug/pprof/profile       — CPU profile (30s 采样)
// /debug/pprof/block         — 阻塞分析
// /debug/pprof/mutex         — 锁竞争
// /debug/pprof/goroutine?debug=2  — 完整 goroutine 栈

// ========== 命令行抓取 ==========
// curl -o goroutine.pprof http://localhost:6060/debug/pprof/goroutine
// curl -o heap.pprof http://localhost:6060/debug/pprof/heap
// curl -o cpu.pprof http://localhost:6060/debug/pprof/profile?seconds=30
// curl -o block.pprof http://localhost:6060/debug/pprof/block
// curl -o mutex.pprof http://localhost:6060/debug/pprof/mutex
// curl -o trace.out http://localhost:6060/debug/pprof/trace?seconds=5
```


## pprof 分析实战


```
// ========== goroutine 分析 ==========
// 问题: goroutine 数量异常
// 命令: go tool pprof http://localhost:6060/debug/pprof/goroutine

// (pprof) top      — 显示最多 goroutine 的函数
// (pprof) traces   — 查看所有 goroutine 栈
// (pprof) web      — SVG 图表

// 排查步骤:
// 1. top 查看数量分布
// 2. 确认是否泄漏 (数量持续增长)
// 3. traces 查看具体阻塞点

// ========== 堆内存分析 ==========
// 问题: 内存泄漏 / 高内存占用
// 命令: go tool pprof -http=:8080 http://localhost:6060/debug/pprof/heap

// (pprof) top
// 显示:
// flat  flat%   sum%   cum   cum%
// 512MB 25.00% 25.00% 1.2GB 60.00%  main.allocateLarge

// 视图:
// - graph: 调用图 (方框越大, 内存越多)
// - flame graph: 火焰图
// - peek: 调用者/被调用者
// - source: 源码行级内存分配

// 注意: heap 显示当前活跃对象
// alloc 显示累积分配 (GC 已回收的也计入)

// ========== CPU 分析 ==========
// 问题: CPU 使用率高 / 慢请求
// 命令: go tool pprof -http=:8080 http://localhost:6060/debug/pprof/profile?seconds=30

// (pprof) top10 — 最耗 CPU 的函数
// (pprof) list  functionName — 查看函数内每行耗时

// 关注:
// - flat: 函数自身 CPU 消耗
// - cum: 函数+调用的 CPU 消耗
// 两者接近 → 函数自身重
// flat << cum → 调用的子函数重

// ========== 阻塞与锁分析 ==========
// 需要先设置采样率:

func init() {
    runtime.SetBlockProfileRate(1)     // 所有阻塞事件
    runtime.SetMutexProfileFraction(1) // 所有锁事件
}

// 分析阻塞:
// go tool pprof http://localhost:6060/debug/pprof/block
// (pprof) top — 最常阻塞的操作

// 分析锁:
// go tool pprof http://localhost:6060/debug/pprof/mutex
// (pprof) top — 锁竞争最激烈的函数

// ========== 比较分析 ==========
// 两次 profile 对比
// go tool pprof -base=heap_before.pprof heap_after.pprof
// 显示新增分配, 用于找泄漏
```


## go tool trace


```
// ========== 运行时追踪 ==========
// go tool trace 可视化 goroutine 调度

import "runtime/trace"

func traceExample() {
    f, _ := os.Create("trace.out")
    defer f.Close()

    trace.Start(f)
    defer trace.Stop()

    // 要追踪的代码
    doWork()
}

// 或通过 HTTP:
// curl -o trace.out http://localhost:6060/debug/pprof/trace?seconds=5
// go tool trace trace.out

// ========== trace 能看到的 ==========
// 1. goroutine 创建/销毁/阻塞/唤醒
// 2. 网络等待
// 3. 系统调用
// 4. GC 事件
// 5. 调度器决策
// 6. 用户自定义事件 (trace.Log)

// ========== 用户自定义事件 ==========
func tracedFunction() {
    trace.Log(context.Background(), "event", "start processing")

    // 区域追踪
    ctx, task := trace.NewTask(context.Background(), "process")
    defer task.End()

    trace.Log(ctx, "event", "processing item 1")
    doStep1()

    trace.Log(ctx, "event", "processing item 2")
    doStep2()
}

// 在 UI 中看到自定义事件的时间线

// ========== trace 分析技巧 ==========
// go tool trace trace.out → 打开浏览器

// 关键视图:
// 1. View trace     — 完整时间线 (goroutine/GC/proc)
// 2. Goroutine analysis — 每个 goroutine 的状态统计
// 3. Network blocking profile — 网络等待
// 4. Synchronization blocking profile — 同步阻塞
// 5. Syscall blocking profile — 系统调用
// 6. Scheduler latency profiler — 调度延迟
// 7. User defined tasks — 自定义任务

// 排查问题:
// - 大量 goroutine 在等待 → 增加并发数
// - GC 频繁 → 减少内存分配
// - 系统调用阻塞 → 使用异步 IO
```


## dlv 调试器


```
// ========== dlv 调试 ==========
// go install github.com/go-delve/delve/cmd/dlv@latest

// 基础命令:
// dlv debug main.go              — 调试 main 包
// dlv test ./...                  — 调试测试
// dlv attach                 — 附加到进程
// dlv exec ./app                  — 调试已编译程序
// dlv connect localhost:2345      — 连接远程调试

// ========== 常用 dlv 命令 ==========
// (dlv) break main.go:42          — 设置断点
// (dlv) break main.main           — 函数断点
// (dlv) breakpoints               — 查看所有断点
// (dlv) continue                  — 继续执行
// (dlv) next                      — 单步跳过
// (dlv) step                      — 单步进入
// (dlv) stepout                   — 跳出函数
// (dlv) print variable            — 打印变量
// (dlv) locals                    — 查看所有本地变量
// (dlv) args                      — 查看函数参数
// (dlv) goroutines                — 查看所有 goroutine
// (dlv) goroutine 2               — 切换到 goroutine 2
// (dlv) bt                        — 调用栈
// (dlv) list                      — 查看源码
// (dlv) clear 1                   — 清除断点

// ========== dlv 条件断点 ==========
// (dlv) break main.go:42 if userID == 100
// (dlv) trace main.ServeHTTP      — 跟踪函数（不打断）

// ========== dlv 远程调试 ==========
// 服务器:
// dlv debug --headless --listen=:2345 --log
//
// 客户端:
// dlv connect localhost:2345
//
// IDE 集成: VS Code, GoLand 原生支持

// ========== GODEBUG 环境变量 ==========
// GODEBUG=schedtrace=1000 ./app   // 调度跟踪 (每 1s)
// GODEBUG=scheddetail=1,schedtrace=1000 ./app  // 详细调度
// GODEBUG=gctrace=1 ./app         // GC 日志
// GODEBUG=inittrace=1 ./app       // init 函数执行
// GODEBUG=cpu.extension=off ./app // 禁用 CPU 扩展

// 输出示例 (schedtrace):
// SCHED 0ms: gomaxprocs=8 idleprocs=6 threads=5 spinningthreads=1 idlethreads=0 runqueue=0 [0 0 0 0 0 0 0 0]
//   gomaxprocs: P 数量
//   idleprocs: 空闲 P
//   threads: 线程数
//   runqueue: 全局队列
```


> **Note:** 💡 pprof 与调试要点: net/http/pprof 端口 6060; goroutine/heap/profile/block/mutex 五个端点; go tool pprof -http=:8080 可视化; flat vs cum 区分自身/总耗时; runtime.SetBlockProfileRate/SetMutexProfileFraction 启用阻塞/锁; go tool trace 可视化调度; dlv debug/attach/connect 调试; GODEBUG=schedtrace/gctrace 运行时诊断; 内存泄漏用 -base 对比 profile; 火焰图快速定位热点。


## 练习


<!-- Converted from: 44_Go pprof 与调试.html -->
