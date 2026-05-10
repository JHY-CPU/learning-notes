# Go errgroup 与并发编排


## 🔗 Go errgroup 与并发编排


errgroup 基础与源码分析、Context 传播、并发限制、常用编排模式、分批处理、bounded concurrency、完整示例。


## errgroup 基础


```
// ========== errgroup ==========
// golang.org/x/sync/errgroup
// 提供: 并发执行 + 错误收集 + Context 取消

import "golang.org/x/sync/errgroup"

// 使用:
func basicErrgroup() {
    g, ctx := errgroup.WithContext(context.Background())

    // 启动多个 goroutine
    for i := 0; i < 3; i++ {
        id := i
        g.Go(func() error {
            // 检查是否已被取消
            select {
            case <-ctx.Done():
                return ctx.Err()
            default:
            }
            return doWork(id)
        })
    }

    // 等待所有完成, 返回第一个错误
    if err := g.Wait(); err != nil {
        log.Printf("出错: %v", err)
    }
}

// ========== errgroup 源码简析 ==========
// 核心结构:
// type Group struct {
//     cancel func()          // Context 取消函数
//     wg     sync.WaitGroup  // 等待所有 goroutine
//     errOnce sync.Once      // 只保存第一个错误
//     err     error          // 第一个错误
// }

// Go 方法:
// func (g *Group) Go(f func() error) {
//     g.wg.Add(1)
//     go func() {
//         defer g.wg.Done()
//         if err := f(); err != nil {
//             g.errOnce.Do(func() {
//                 g.err = err
//                 if g.cancel != nil {
//                     g.cancel()  // 取消所有其他 goroutine
//                 }
//             })
//         }
//     }()
// }

// Wait 方法:
// func (g *Group) Wait() error {
//     g.wg.Wait()
//     if g.cancel != nil {
//         g.cancel()
//     }
//     return g.err
// }

// 关键特性: 任意 goroutine 返回错误 → 自动取消其他
```


## 并发限制


```
// ========== 限制并发数 ==========
// errgroup 默认无限制 (所有 goroutine 同时启动)
// 需要手动限制

// 方法 1: 信号量
func boundedSemaphore(items []int, limit int) error {
    g, ctx := errgroup.WithContext(context.Background())
    sem := make(chan struct{}, limit)

    for _, item := range items {
        item := item
        g.Go(func() error {
            select {
            case sem <- struct{}{}:
            case <-ctx.Done():
                return ctx.Err()
            }
            defer func() { <-sem }()

            return processItem(ctx, item)
        })
    }

    return g.Wait()
}

// 方法 2: SetLimit (Go 1.20+)
// g.SetLimit(10)  // 最多 10 个并发

func setLimitExample(items []int) error {
    g, ctx := errgroup.WithContext(context.Background())
    g.SetLimit(10)  // 限制并发

    for _, item := range items {
        item := item
        g.Go(func() error {
            return processItem(ctx, item)
        })
    }

    return g.Wait()
}

// 方法 3: 分批处理
func batchProcess(items []int, batchSize int) error {
    g, ctx := errgroup.WithContext(context.Background())

    for i := 0; i < len(items); i += batchSize {
        end := i + batchSize
        if end > len(items) {
            end = len(items)
        }
        batch := items[i:end]

        g.Go(func() error {
            for _, item := range batch {
                if err := processItem(ctx, item); err != nil {
                    return err
                }
            }
            return nil
        })
    }

    return g.Wait()
}

// ========== 收集结果 ==========
// errgroup 不直接支持收集结果
// 需要自行管理

func collectResults(items []int) ([]string, error) {
    g, ctx := errgroup.WithContext(context.Background())
    results := make([]string, len(items))

    for i, item := range items {
        i, item := i, item
        g.Go(func() error {
            result, err := processWithResult(ctx, item)
            if err != nil {
                return err
            }
            results[i] = result  // 每个 goroutine 写自己的索引
            return nil
        })
    }

    err := g.Wait()
    return results, err
}

// 注意: 每个 goroutine 只写自己的索引, 不需要锁
// 因为索引不重叠
```


## 编排模式


```
// ========== 模式 1: 扇出/扇入 ==========
// 并发请求多个服务, 合并结果

type ServiceResult struct {
    Service string
    Data    string
    Err     error
}

func fanOutFanIn(ctx context.Context) ([]ServiceResult, error) {
    g, ctx := errgroup.WithContext(ctx)
    results := make(chan ServiceResult, 3)

    // 扇出: 同时请求 3 个服务
    services := []string{"users", "orders", "products"}
    for _, svc := range services {
        svc := svc
        g.Go(func() error {
            data, err := callService(ctx, svc)
            results <- ServiceResult{
                Service: svc,
                Data:    data,
                Err:     err,
            }
            return err  // 错误会取消其他
        })
    }

    // 等待所有完成, 关闭 results
    go func() {
        g.Wait()
        close(results)
    }()

    // 收集结果
    var out []ServiceResult
    for r := range results {
        out = append(out, r)
    }

    return out, g.Wait()
}

// ========== 模式 2: 最快响应 ==========
// 多个相同服务, 取最快返回

func fastestResponse(ctx context.Context, endpoints []string) (string, error) {
    g, ctx := errgroup.WithContext(ctx)
    resultCh := make(chan string, 1)

    for _, ep := range endpoints {
        ep := ep
        g.Go(func() error {
            resp, err := callService(ctx, ep)
            if err != nil {
                return err
            }

            // 非阻塞发送: 第一个写入, 后续忽略
            select {
            case resultCh <- resp:
            default:
            }
            return nil
        })
    }

    // 等待第一个成功或全部失败
    select {
    case result := <-resultCh:
        return result, nil
    case <-ctx.Done():
        return "", ctx.Err()
    }
}

// ========== 模式 3: 分页拉取全部 ==========
func fetchAllPages(ctx context.Context, baseURL string) ([]Item, error) {
    g, ctx := errgroup.WithContext(ctx)
    var mu sync.Mutex
    var allItems []Item

    for page := 1; ; page++ {
        page := page
        var pageItems []Item
        var err error

        // 获取当前页
        pageItems, err = fetchPage(ctx, baseURL, page)
        if err != nil {
            return nil, err
        }
        if len(pageItems) == 0 {
            break  // 最后一页
        }

        // 并发处理每页
        mu.Lock()
        allItems = append(allItems, pageItems...)
        mu.Unlock()
    }

    return allItems, nil
}

// ========== 模式 4: pipeline + errgroup ==========
// 每个阶段用 errgroup 管理

func pipelineWithErrgroup(ctx context.Context, ids []int) ([]Result, error) {
    // 阶段 1: 获取数据
    fetchGroup, ctx := errgroup.WithContext(ctx)
    fetched := make([]Data, len(ids))

    for i, id := range ids {
        i, id := i, id
        fetchGroup.Go(func() error {
            data, err := fetchData(ctx, id)
            if err != nil {
                return err
            }
            fetched[i] = data
            return nil
        })
    }

    if err := fetchGroup.Wait(); err != nil {
        return nil, err
    }

    // 阶段 2: 处理数据 (等阶段 1 完成)
    processGroup, ctx := errgroup.WithContext(ctx)
    results := make([]Result, len(fetched))

    for i, data := range fetched {
        i, data := i, data
        processGroup.Go(func() error {
            result, err := processData(ctx, data)
            if err != nil {
                return err
            }
            results[i] = result
            return nil
        })
    }

    err := processGroup.Wait()
    return results, err
}

// ========== errgroup 最佳实践 ==========
// 1. 限制并发: g.SetLimit(N) 避免资源耗尽
// 2. 检查 ctx: goroutine 中检查 ctx.Done()
// 3. 收集结果: 用索引分配或 channel
// 4. 不要在 g.Go 中启动子 goroutine
// 5. 正确处理 context 取消
```


## 完整示例


```
// ========== 完整: 并发用户导入 ==========

type ImportResult struct {
    UserID   int
    Status   string
    Error    error
}

func importUsers(ctx context.Context, users []User) error {
    g, ctx := errgroup.WithContext(ctx)
    g.SetLimit(20)  // 最多 20 个并发

    results := make(chan ImportResult, len(users))
    done := make(chan struct{})

    // 收集结果 (单独 goroutine)
    go func() {
        for r := range results {
            if r.Error != nil {
                log.Printf("用户 %d 导入失败: %v", r.UserID, r.Error)
            } else {
                log.Printf("用户 %d 导入成功", r.UserID)
            }
        }
        close(done)
    }()

    // 并发导入
    for _, user := range users {
        user := user
        g.Go(func() error {
            // 检查取消
            select {
            case <-ctx.Done():
                results <- ImportResult{
                    UserID: user.ID,
                    Status: "cancelled",
                    Error:  ctx.Err(),
                }
                return ctx.Err()
            default:
            }

            // 导入
            err := importUser(ctx, user)
            results <- ImportResult{
                UserID: user.ID,
                Status: status(err),
                Error:  err,
            }
            return err
        })
    }

    // 等待导入完成
    err := g.Wait()
    close(results)
    <-done

    return err
}

// ========== errgroup vs WaitGroup ==========
// errgroup:
//   - 并发执行返回错误
//   - 自动 Context 取消
//   - 适合: 任务相互独立, 任一失败可整体取消

// WaitGroup:
//   - 只等待完成, 不关心错误
//   - 无 Context 集成
//   - 适合: 所有任务都必须完成

// 选择:
// - 需要错误传播 + 取消 → errgroup
// - 只需要等待 → WaitGroup

// ========== 替代方案 ==========
// 如果不引入外部包, 可自己实现:

type Group struct {
    cancel func()
    wg     sync.WaitGroup
    err    atomic.Value
}

func (g *Group) Go(fn func() error) {
    g.wg.Add(1)
    go func() {
        defer g.wg.Done()
        if err := fn(); err != nil {
            g.err.Store(err)
            if g.cancel != nil {
                g.cancel()
            }
        }
    }()
}

func (g *Group) Wait() error {
    g.wg.Wait()
    if e := g.err.Load(); e != nil {
        return e.(error)
    }
    return nil
}
```


> **Note:** 💡 errgroup 要点: goroutine 并发 + 错误收集 + Context 自动取消; g.SetLimit(N) 限制并发; 收集结果需自行管理 (索引分配 / channel); 常与 context.WithCancel 配合 (任一失败取消其他); 模式: 扇出扇入 (并发请求多服务), 最快响应 (多个 endpoint), 分页拉取, pipeline 分阶段; errgroup vs WaitGroup: errgroup 需错误传播, WaitGroup 只需等待; Go 1.20+ 支持 SetLimit。


## 练习


<!-- Converted from: 34_Go errgroup 与并发编排.html -->
