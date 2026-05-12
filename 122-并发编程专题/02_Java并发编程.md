# Java 并发编程

## Thread 与 Runnable

Java 中创建线程有两种基本方式：继承 `Thread` 类或实现 `Runnable` 接口。

### 继承 Thread

```java
public class ThreadDemo extends Thread {
    @Override
    public void run() {
        System.out.println("线程运行: " + Thread.currentThread().getName());
    }

    public static void main(String[] args) {
        ThreadDemo t1 = new ThreadDemo();
        t1.start(); // 启动新线程
    }
}
```

### 实现 Runnable（推荐）

```java
public class RunnableDemo implements Runnable {
    @Override
    public void run() {
        System.out.println("任务执行: " + Thread.currentThread().getName());
    }

    public static void main(String[] args) {
        Thread t1 = new Thread(new RunnableDemo(), "worker-1");
        t1.start();

        // Lambda 简写
        Thread t2 = new Thread(() -> {
            System.out.println("Lambda 任务: " + Thread.currentThread().getName());
        }, "worker-2");
        t2.start();
    }
}
```

实现 `Runnable` 接口优于继承 `Thread`，因为 Java 只支持单继承，而接口可以多实现，代码复用性更好。

## synchronized 关键字

`synchronized` 是 Java 内置的互斥同步机制，保证同一时刻只有一个线程执行同步代码块。

### 三种使用方式

```java
public class SynchronizedDemo {
    private int count = 0;

    // 1. 同步实例方法（锁对象为 this）
    public synchronized void increment() {
        count++;
    }

    // 2. 同步静态方法（锁对象为类的 Class 对象）
    public static synchronized void staticMethod() {
        System.out.println("静态同步方法");
    }

    // 3. 同步代码块（指定锁对象）
    public void decrement() {
        synchronized (this) {
            count--;
        }
    }

    public int getCount() {
        synchronized (this) {
            return count;
        }
    }
}
```

### synchronized 的特性

- **可重入性**：同一个线程可以多次获取同一把锁，不会死锁
- **可见性**：释放锁时会将工作内存刷新到主内存，获取锁时会从主内存读取
- **互斥性**：同一时刻只有一个线程能持有锁

```java
public class ReentrantDemo {
    public synchronized void outer() {
        System.out.println("外层方法");
        inner(); // 可以调用，因为是可重入锁
    }

    public synchronized void inner() {
        System.out.println("内层方法");
    }
}
```

## ReentrantLock

`ReentrantLock` 是 `java.util.concurrent` 包提供的显式锁，功能比 `synchronized` 更丰富。

### 基本用法

```java
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;

public class ReentrantLockDemo {
    private final ReentrantLock lock = new ReentrantLock();
    private int balance = 0;

    public void deposit(int amount) {
        lock.lock(); // 必须手动加锁
        try {
            balance += amount;
            System.out.println("存款: " + amount + ", 余额: " + balance);
        } finally {
            lock.unlock(); // 必须在 finally 中释放锁
        }
    }

    public void withdraw(int amount) throws InterruptedException {
        lock.lock();
        try {
            while (balance < amount) {
                Thread.sleep(10); // 也可以用 Condition 替代
            }
            balance -= amount;
            System.out.println("取款: " + amount + ", 余额: " + balance);
        } finally {
            lock.unlock();
        }
    }
}
```

### Condition 实现等待/通知

```java
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

public class BoundedQueue<T> {
    private final Queue<T> queue = new LinkedList<>();
    private final int capacity;
    private final ReentrantLock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();

    public BoundedQueue(int capacity) {
        this.capacity = capacity;
    }

    public void put(T item) throws InterruptedException {
        lock.lock();
        try {
            while (queue.size() == capacity) {
                notFull.await(); // 队列满，等待消费者取走元素
            }
            queue.add(item);
            notEmpty.signal(); // 通知消费者
        } finally {
            lock.unlock();
        }
    }

    public T take() throws InterruptedException {
        lock.lock();
        try {
            while (queue.isEmpty()) {
                notEmpty.await(); // 队列空，等待生产者放入元素
            }
            T item = queue.poll();
            notFull.signal(); // 通知生产者
            return item;
        } finally {
            lock.unlock();
        }
    }
}
```

### tryLock 与可中断锁

```java
public class TryLockDemo {
    private final ReentrantLock lock = new ReentrantLock();

    public boolean tryDoWork() {
        // 尝试获取锁，不等待
        if (lock.tryLock()) {
            try {
                System.out.println("获取到锁，执行工作");
                return true;
            } finally {
                lock.unlock();
            }
        }
        System.out.println("未能获取锁，执行其他逻辑");
        return false;
    }

    public void interruptibleWork() throws InterruptedException {
        // 可被中断地获取锁
        lock.lockInterruptibly();
        try {
            System.out.println("执行可中断任务");
        } finally {
            lock.unlock();
        }
    }
}
```

## ConcurrentHashMap

`ConcurrentHashMap` 是线程安全的哈希表，采用分段锁（Java 7）或 CAS + synchronized（Java 8+）实现高并发。

### 基本操作

```java
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

public class ConcurrentHashMapDemo {
    private final ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

    // 原子性的 put-if-absent
    public void putIfAbsentDemo() {
        map.putIfAbsent("key", 1); // 仅在 key 不存在时放入
    }

    // 原子性的 compute
    public void computeDemo(String key) {
        map.compute(key, (k, v) -> (v == null) ? 1 : v + 1);
    }

    // 原子性的累加
    public void atomicIncrement(String key) {
        map.merge(key, 1, Integer::sum); // key 不存在时值为 1，存在时加 1
    }

    // 批量操作
    public void bulkOperation() {
        // 并行遍历（使用公共线程池）
        map.forEach(1, (k, v) -> System.out.println(k + " = " + v));

        // 搜索
        String result = map.search(1, (k, v) -> v > 10 ? k : null);
        System.out.println("第一个值大于 10 的 key: " + result);

        // 归约
        int sum = map.reduce(1, (k, v) -> v, Integer::sum);
        System.out.println("所有值的总和: " + sum);
    }
}
```

## ExecutorService 线程池

`ExecutorService` 是 Java 的线程池抽象，避免频繁创建和销毁线程的开销。

### 创建线程池

```java
import java.util.concurrent.*;

public class ExecutorServiceDemo {

    // 1. 固定大小线程池
    ExecutorService fixedPool = Executors.newFixedThreadPool(4);

    // 2. 可缓存线程池（无上限，空闲 60 秒回收）
    ExecutorService cachedPool = Executors.newCachedThreadPool();

    // 3. 单线程池（串行执行）
    ExecutorService singlePool = Executors.newSingleThreadExecutor();

    // 4. 定时线程池
    ScheduledExecutorService scheduledPool = Executors.newScheduledThreadPool(2);

    // 5. 推荐：手动创建线程池，精确控制参数
    ThreadPoolExecutor customPool = new ThreadPoolExecutor(
        4,                // 核心线程数
        8,                // 最大线程数
        60, TimeUnit.SECONDS, // 空闲线程存活时间
        new ArrayBlockingQueue<>(100), // 工作队列
        new ThreadPoolExecutor.CallerRunsPolicy() // 拒绝策略
    );
}
```

### 提交任务

```java
import java.util.concurrent.*;
import java.util.List;
import java.util.ArrayList;

public class TaskSubmissionDemo {
    public static void main(String[] args) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(3);

        // 1. execute: 无返回值
        pool.execute(() -> System.out.println("execute 任务"));

        // 2. submit: 返回 Future
        Future<String> future = pool.submit(() -> {
            Thread.sleep(1000);
            return "计算结果";
        });

        // 获取结果（阻塞）
        String result = future.get(5, TimeUnit.SECONDS);
        System.out.println("Future 结果: " + result);

        // 3. invokeAll: 提交所有任务，等待全部完成
        List<Callable<Integer>> tasks = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            final int id = i;
            tasks.add(() -> {
                Thread.sleep(id * 100);
                return id * id;
            });
        }
        List<Future<Integer>> results = pool.invokeAll(tasks);
        for (Future<Integer> f : results) {
            System.out.println("结果: " + f.get());
        }

        // 4. invokeAny: 返回第一个完成的任务结果
        String fastest = pool.invokeAny(List.of(
            () -> { Thread.sleep(300); return "slow"; },
            () -> { Thread.sleep(100); return "fast"; },
            () -> { Thread.sleep(200); return "medium"; }
        ));
        System.out.println("最快的结果: " + fastest); // fast

        pool.shutdown();
        pool.awaitTermination(10, TimeUnit.SECONDS);
    }
}
```

## CompletableFuture

`CompletableFuture` 是 Java 8 引入的异步编程工具，支持链式调用和组合操作。

### 基本创建方式

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CompletableFutureDemo {

    public static void main(String[] args) throws Exception {
        // 1. supplyAsync: 异步执行有返回值的任务
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> {
            sleep(200);
            return "Hello";
        });

        // 2. runAsync: 异步执行无返回值的任务
        CompletableFuture<Void> future2 = CompletableFuture.runAsync(() -> {
            System.out.println("后台任务执行");
        });

        // 3. 指定自定义线程池
        ExecutorService pool = Executors.newFixedThreadPool(4);
        CompletableFuture<String> future3 = CompletableFuture.supplyAsync(() -> {
            return "使用自定义线程池";
        }, pool);
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
```

### 链式转换

```java
public class CompletableFutureChain {
    public static void main(String[] args) throws Exception {
        // thenApply: 转换结果
        String result = CompletableFuture.supplyAsync(() -> "world")
            .thenApply(s -> "hello " + s)
            .thenApply(String::toUpperCase)
            .get();
        System.out.println(result); // HELLO WORLD

        // thenAccept: 消费结果，无返回值
        CompletableFuture.supplyAsync(() -> 42)
            .thenAccept(n -> System.out.println("得到: " + n));

        // thenCompose: 扁平化嵌套的 CompletableFuture
        CompletableFuture.supplyAsync(() -> "user-123")
            .thenCompose(id -> CompletableFuture.supplyAsync(() -> "User: " + id))
            .thenAccept(System.out::println);

        // thenCombine: 组合两个独立的 Future
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "World");
        CompletableFuture<String> combined = future1.thenCombine(future2,
            (a, b) -> a + " " + b);
        System.out.println(combined.get()); // Hello World
    }
}
```

### 异常处理

```java
public class CompletableFutureException {
    public static void main(String[] args) throws Exception {
        // exceptionally: 异常时返回默认值
        String result1 = CompletableFuture.supplyAsync(() -> {
            if (true) throw new RuntimeException("出错了");
            return "success";
        }).exceptionally(ex -> {
            System.out.println("异常: " + ex.getMessage());
            return "默认值";
        }).get();
        System.out.println(result1); // 默认值

        // handle: 无论成功或失败都会调用
        String result2 = CompletableFuture.supplyAsync(() -> {
            throw new RuntimeException("错误");
        }).handle((value, ex) -> {
            if (ex != null) {
                return "恢复的值";
            }
            return value;
        }).get();
        System.out.println(result2); // 恢复的值

        // whenComplete: 观察结果（不转换）
        CompletableFuture.supplyAsync(() -> "data")
            .whenComplete((value, ex) -> {
                if (ex == null) {
                    System.out.println("完成: " + value);
                }
            });
    }
}
```

### 组合多个 Future

```java
import java.util.concurrent.CompletableFuture;

public class CompletableFutureAllAny {
    public static void main(String[] args) throws Exception {
        // allOf: 等待所有 Future 完成
        CompletableFuture<Void> all = CompletableFuture.allOf(
            CompletableFuture.supplyAsync(() -> { sleep(300); return "A"; }),
            CompletableFuture.supplyAsync(() -> { sleep(200); return "B"; }),
            CompletableFuture.supplyAsync(() -> { sleep(100); return "C"; })
        );
        all.get(); // 等待全部完成

        // anyOf: 返回第一个完成的
        CompletableFuture<Object> any = CompletableFuture.anyOf(
            CompletableFuture.supplyAsync(() -> { sleep(300); return "慢"; }),
            CompletableFuture.supplyAsync(() -> { sleep(100); return "快"; })
        );
        System.out.println("第一个完成: " + any.get()); // 快
    }

    private static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
```

## 原子类

`java.util.concurrent.atomic` 包提供了一系列无锁的线程安全原子操作。

```java
import java.util.concurrent.atomic.*;

public class AtomicDemo {
    // AtomicInteger 基本操作
    private final AtomicInteger counter = new AtomicInteger(0);

    public void atomicOperations() {
        counter.incrementAndGet();     // ++i
        counter.decrementAndGet();     // --i
        counter.addAndGet(5);          // i += 5
        counter.getAndUpdate(n -> n * 2); // 原子更新
        counter.compareAndSet(10, 20); // CAS: 如果当前值是 10 则改为 20
    }

    // AtomicReference: 原子引用
    private final AtomicReference<String> ref = new AtomicReference<>("initial");

    public void updateReference() {
        ref.compareAndSet("initial", "updated");
    }

    // LongAdder: 高并发下比 AtomicInteger 性能更好
    private final LongAdder longAdder = new LongAdder();

    public void highConcurrencyCounter() {
        longAdder.increment();
        longAdder.add(5);
        long sum = longAdder.sum();
    }
}
```

## 总结

| 工具 | 用途 | 特点 |
|------|------|------|
| synchronized | 内置互斥锁 | 简单易用，功能有限 |
| ReentrantLock | 显式锁 | 支持公平锁、超时、条件变量 |
| ConcurrentHashMap | 线程安全哈希表 | 高并发读写，原子操作 |
| ExecutorService | 线程池 | 线程复用，资源管理 |
| CompletableFuture | 异步编程 | 链式调用，组合操作 |
| AtomicInteger | 原子操作 | 无锁，高性能 |
