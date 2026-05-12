# Java 并发高级

## CountDownLatch

`CountDownLatch` 是一个同步工具，允许一个或多个线程等待其他线程完成操作。计数器只能减少，不能重置。

### 基本用法

```java
import java.util.concurrent.CountDownLatch;

public class CountDownLatchDemo {
    public static void main(String[] args) throws InterruptedException {
        // 等待 3 个任务完成
        CountDownLatch latch = new CountDownLatch(3);

        for (int i = 1; i <= 3; i++) {
            final int id = i;
            new Thread(() -> {
                try {
                    Thread.sleep(id * 500L);
                    System.out.println("任务 " + id + " 完成");
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown(); // 计数器减 1
                }
            }).start();
        }

        System.out.println("主线程等待所有任务完成...");
        latch.await(); // 阻塞直到计数器为 0
        System.out.println("所有任务完成，主线程继续");
    }
}
```

### 应用场景：并行加载配置

```java
import java.util.concurrent.CountDownLatch;

public class ParallelConfigLoader {
    private final CountDownLatch latch = new CountDownLatch(3);
    private String dbConfig;
    private String cacheConfig;
    private String mqConfig;

    public void loadAll() throws InterruptedException {
        new Thread(() -> {
            dbConfig = loadFromDB();
            latch.countDown();
        }).start();

        new Thread(() -> {
            cacheConfig = loadFromCache();
            latch.countDown();
        }).start();

        new Thread(() -> {
            mqConfig = loadMQConfig();
            latch.countDown();
        }).start();

        latch.await(); // 等待所有配置加载完成
        System.out.println("配置加载完成: DB=" + dbConfig
            + ", Cache=" + cacheConfig + ", MQ=" + mqConfig);
    }

    private String loadFromDB() { return "db-config"; }
    private String loadFromCache() { return "cache-config"; }
    private String loadMQConfig() { return "mq-config"; }
}
```

## CyclicBarrier

`CyclicBarrier` 允许一组线程互相等待，直到所有线程都到达屏障点后，再一起继续执行。与 `CountDownLatch` 不同，它可以重置重复使用。

### 基本用法

```java
import java.util.concurrent.CyclicBarrier;

public class CyclicBarrierDemo {
    public static void main(String[] args) {
        // 3 个线程都到达屏障后，先执行 barrierAction，再继续
        CyclicBarrier barrier = new CyclicBarrier(3, () -> {
            System.out.println("所有线程已到达，开始下一阶段计算");
        });

        for (int i = 1; i <= 3; i++) {
            final int id = i;
            new Thread(() -> {
                try {
                    System.out.println("线程 " + id + " 阶段 1 完成");
                    barrier.await(); // 等待其他线程

                    System.out.println("线程 " + id + " 阶段 2 完成");
                    barrier.await(); // 第二次使用（可重用）

                    System.out.println("线程 " + id + " 阶段 3 完成");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        }
    }
}
```

### 应用场景：多线程矩阵计算

```java
import java.util.concurrent.CyclicBarrier;

public class MatrixComputation {
    private static final int SIZE = 4;
    private static final double[][] matrix = new double[SIZE][SIZE];
    private static final CyclicBarrier barrier = new CyclicBarrier(SIZE);

    public static void main(String[] args) throws InterruptedException {
        // 每个线程计算矩阵的一行
        for (int row = 0; row < SIZE; row++) {
            final int r = row;
            new Thread(() -> {
                try {
                    // 第一阶段：计算当前行
                    for (int j = 0; j < SIZE; j++) {
                        matrix[r][j] = Math.random() * 100;
                    }
                    System.out.println("行 " + r + " 计算完成");
                    barrier.await(); // 等待所有行计算完成

                    // 第二阶段：归约（需要所有行数据）
                    double sum = 0;
                    for (int j = 0; j < SIZE; j++) {
                        sum += matrix[r][j];
                    }
                    System.out.println("行 " + r + " 总和: " + sum);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }).start();
        }
    }
}
```

## Semaphore

`Semaphore`（信号量）控制同时访问特定资源的线程数量，常用于限流。

### 基本用法

```java
import java.util.concurrent.Semaphore;

public class SemaphoreDemo {
    // 最多允许 3 个线程同时访问
    private static final Semaphore semaphore = new Semaphore(3);

    public static void main(String[] args) {
        for (int i = 1; i <= 10; i++) {
            final int id = i;
            new Thread(() -> {
                try {
                    semaphore.acquire(); // 获取许可
                    System.out.println("线程 " + id + " 获取许可，剩余: " + semaphore.availablePermits());
                    Thread.sleep(2000); // 模拟工作
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    semaphore.release(); // 释放许可
                    System.out.println("线程 " + id + " 释放许可");
                }
            }).start();
        }
    }
}
```

### 应用场景：数据库连接池

```java
import java.util.concurrent.Semaphore;

public class ConnectionPool {
    private final Semaphore semaphore;
    private final boolean[] used;
    private final int poolSize;

    public ConnectionPool(int size) {
        this.poolSize = size;
        this.semaphore = new Semaphore(size, true); // 公平模式
        this.used = new boolean[size];
    }

    public int acquire() throws InterruptedException {
        semaphore.acquire();
        synchronized (this) {
            for (int i = 0; i < poolSize; i++) {
                if (!used[i]) {
                    used[i] = true;
                    return i;
                }
            }
        }
        throw new IllegalStateException("无可用连接");
    }

    public void release(int id) {
        synchronized (this) {
            used[id] = false;
        }
        semaphore.release();
    }
}
```

## ForkJoinPool

`ForkJoinPool` 采用分治策略，将大任务拆分为小任务并行执行，特别适合递归计算。

### ForkJoinTask 基本用法

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class ForkJoinSum extends RecursiveTask<Long> {
    private static final int THRESHOLD = 10000;
    private final long[] array;
    private final int start;
    private final int end;

    public ForkJoinSum(long[] array, int start, int end) {
        this.array = array;
        this.start = start;
        this.end = end;
    }

    @Override
    protected Long compute() {
        int length = end - start;
        if (length <= THRESHOLD) {
            // 足够小，直接计算
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += array[i];
            }
            return sum;
        }
        // 拆分任务
        int mid = start + length / 2;
        ForkJoinSum left = new ForkJoinSum(array, start, mid);
        ForkJoinSum right = new ForkJoinSum(array, mid, end);
        left.fork();  // 异步执行左半部分
        long rightResult = right.compute(); // 当前线程计算右半部分
        long leftResult = left.join(); // 等待左半部分完成
        return leftResult + rightResult;
    }

    public static void main(String[] args) {
        long[] array = new long[1_000_000];
        for (int i = 0; i < array.length; i++) {
            array[i] = i + 1;
        }

        ForkJoinPool pool = new ForkJoinPool();
        Long sum = pool.invoke(new ForkJoinSum(array, 0, array.length));
        System.out.println("总和: " + sum); // 500000500000
    }
}
```

### 并行流与 ForkJoinPool

```java
import java.util.stream.LongStream;

public class ParallelStreamDemo {
    public static void main(String[] args) {
        // 并行流底层使用 ForkJoinPool
        long sum = LongStream.rangeClosed(1, 1_000_000)
            .parallel()
            .sum();
        System.out.println("并行求和: " + sum);

        // 指定自定义 ForkJoinPool
        ForkJoinPool customPool = new ForkJoinPool(4);
        Long result = customPool.submit(() ->
            LongStream.rangeClosed(1, 1_000_000)
                .parallel()
                .sum()
        ).join();
        System.out.println("自定义池求和: " + result);
    }
}
```

## StampedLock

`StampedLock` 是 Java 8 引入的读写锁改进版，支持乐观读，减少读写冲突。

### 三种模式

```java
import java.util.concurrent.locks.StampedLock;

public class Point {
    private double x, y;
    private final StampedLock lock = new StampedLock();

    // 写锁（排他）
    public void move(double deltaX, double deltaY) {
        long stamp = lock.writeLock();
        try {
            x += deltaX;
            y += deltaY;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    // 悲观读锁
    public double distanceFromOrigin() {
        long stamp = lock.readLock();
        try {
            return Math.sqrt(x * x + y * y);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    // 乐观读：不加锁，验证期间是否有写操作
    public double distanceOptimistic() {
        long stamp = lock.tryOptimisticRead(); // 获取乐观读标记
        double currentX = x;
        double currentY = y;

        if (!lock.validate(stamp)) {
            // 乐观读期间发生了写操作，降级为悲观读
            stamp = lock.readLock();
            try {
                currentX = x;
                currentY = y;
            } finally {
                lock.unlockRead(stamp);
            }
        }
        return Math.sqrt(currentX * currentX + currentY * currentY);
    }

    // 读锁升级为写锁
    public void moveIfAtOrigin(double newX, double newY) {
        long stamp = lock.readLock();
        try {
            while (x == 0.0 && y == 0.0) {
                long ws = lock.tryConvertToWriteLock(stamp);
                if (ws != 0L) {
                    stamp = ws;
                    x = newX;
                    y = newY;
                    return;
                }
                lock.unlockRead(stamp);
                stamp = lock.writeLock();
            }
        } finally {
            lock.unlock(stamp);
        }
    }
}
```

## VarHandle

`VarHandle` 是 Java 9 引入的底层变量访问工具，提供比 `AtomicXxx` 更灵活的原子操作和内存语义控制。

### 基本用法

```java
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class VarHandleDemo {
    private volatile int count = 0;
    private int plainCount = 0;
    private static final VarHandle COUNT_HANDLE;
    private static final VarHandle PLAIN_COUNT_HANDLE;

    static {
        try {
            MethodHandles.Lookup lookup = MethodHandles.lookup();
            COUNT_HANDLE = lookup.findVarHandle(
                VarHandleDemo.class, "count", int.class);
            PLAIN_COUNT_HANDLE = lookup.findVarHandle(
                VarHandleDemo.class, "plainCount", int.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    public void atomicIncrement() {
        // 原子递增
        COUNT_HANDLE.getAndAdd(this, 1);
    }

    public int atomicRead() {
        // volatile 读
        return (int) COUNT_HANDLE.getVolatile(this);
    }

    public void atomicWrite(int value) {
        // volatile 写
        COUNT_HANDLE.setVolatile(this, value);
    }

    public boolean compareAndSwap(int expected, int newValue) {
        // CAS 操作
        return COUNT_HANDLE.compareAndSet(this, expected, newValue);
    }

    // 内存屏障控制
    public void orderedWrite() {
        PLAIN_COUNT_HANDLE.set(this, 42);
        VarHandle.fullFence(); // 全屏障：保证之前的操作对后续读可见
    }
}
```

### VarHandle 与数组

```java
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

public class VarHandleArrayDemo {
    // 数组元素的 VarHandle
    private static final VarHandle ARRAY_HANDLE = MethodHandles.arrayElementVarHandle(int[].class);

    public static void main(String[] args) {
        int[] array = new int[1000];

        // 原子性地更新数组元素
        ARRAY_HANDLE.getAndAdd(array, 0, 1);  // array[0] += 1
        ARRAY_HANDLE.compareAndSet(array, 0, 1, 2); // array[0] == 1 ? array[0] = 2

        System.out.println("array[0] = " + ARRAY_HANDLE.get(array, 0));
    }
}
```

## CountDownLatch 与 CyclicBarrier 对比

| 特性 | CountDownLatch | CyclicBarrier |
|------|---------------|---------------|
| 计数器 | 不可重置 | 可重复使用 |
| 等待方式 | 主线程等待子线程 | 线程互相等待 |
| 回调 | 无 | 有 barrierAction |
| 适用场景 | 一次性等待 | 分阶段执行 |

## 总结

| 工具 | 核心功能 | 适用场景 |
|------|---------|---------|
| CountDownLatch | 一次性等待 | 启动等待、并行加载 |
| CyclicBarrier | 分阶段同步 | 迭代计算、多阶段任务 |
| Semaphore | 并发控制 | 限流、连接池 |
| ForkJoinPool | 分治并行 | 递归计算、大数据处理 |
| StampedLock | 乐观读锁 | 读多写少场景 |
| VarHandle | 底层原子操作 | 高性能无锁数据结构 |
