# 并发编程专题

本模块系统整理并发编程的核心概念、常用模式和跨语言实现，涵盖 Java、Go、Python、Rust、C++ 五种主流语言的并发编程技术。

## 目录

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | [并发编程基础概念](01_并发编程基础概念.md) | 竞态条件、死锁、活锁、饥饿、线程安全、happens-before |
| 02 | [Java 并发编程](02_Java并发编程.md) | Thread/Runnable、synchronized、ReentrantLock、ConcurrentHashMap、CompletableFuture、ExecutorService |
| 03 | [Java 并发高级](03_Java并发高级.md) | CountDownLatch、CyclicBarrier、Semaphore、ForkJoinPool、StampedLock、VarHandle |
| 04 | [Goroutine 与 Channel](04_Goroutine与Channel.md) | go 关键字、有缓冲/无缓冲 channel、select、WaitGroup、context、worker pool |
| 05 | [Go 并发模式](05_Go并发模式.md) | Fan-in/Fan-out、pipeline、取消机制、限流、errgroup |
| 06 | [Python 并发编程](06_Python并发编程.md) | threading、multiprocessing、concurrent.futures、GIL、异步方案 |
| 07 | [Python AsyncIO](07_Python_AsyncIO.md) | async/await、event loop、aiohttp、asyncio.gather/queue、生产者-消费者 |
| 08 | [Rust 并发编程](08_Rust并发编程.md) | 所有权与并发、Arc/Mutex、mpsc channel、Send/Sync、Rayon |
| 09 | [Rust 异步编程](09_Rust异步编程.md) | async/await、tokio runtime、select!、join!、async streams |
| 10 | [C++ 并发编程](10_C++并发编程.md) | std::thread、mutex/lock_guard、condition_variable、future/promise、atomic |
| 11 | [并发数据结构](11_并发数据结构.md) | 无锁队列、并发哈希表、读写锁、CAS 操作 |
| 12 | [并发设计模式](12_并发设计模式.md) | 生产者-消费者、读者-写者、线程池、主动对象、CSP、Actor 模型 |
| 13 | [死锁检测与预防](13_死锁检测与预防.md) | 锁排序、超时检测、银行家算法、无锁替代方案 |
| 14 | [内存模型与可见性](14_内存模型与可见性.md) | JMM、C++ memory_order、Rust 内存模型、volatile、fence |

## 学习建议

1. **入门阶段**：先阅读第 01 章建立基础概念，再根据需要的语言选择对应章节
2. **进阶阶段**：学习第 11-12 章的并发数据结构与设计模式
3. **深入阶段**：掌握第 13-14 章的死锁处理与内存模型

## 适用语言

- **Java**：第 02-03 章
- **Go**：第 04-05 章
- **Python**：第 06-07 章
- **Rust**：第 08-09 章
- **C++**：第 10 章
- **跨语言**：第 01、11-14 章
