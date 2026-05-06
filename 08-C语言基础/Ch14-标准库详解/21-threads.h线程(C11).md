# threads.h - 多线程（C11）

## 1. 概述

`<threads.h>`（C11引入）提供了标准化的多线程API，包括线程创建与管理、互斥锁和条件变量。此前多线程编程依赖平台特定API（如POSIX pthreads）。

## 2. 核心类型

```c
#include <threads.h>

thrd_t          // 线程标识符类型
mtx_t           // 互斥锁类型
cnd_t           // 条件变量类型
tss_t           // 线程本地存储键类型
once_flag       // 一次性调用标志
thrd_start_t    // 线程函数类型: int (*)(void*)
```

### 返回码

```c
thrd_success    // 成功
thrd_timedout   // 超时
thrd_busy       // 资源忙
thrd_nomem      // 内存不足
thrd_error      // 错误
```

## 3. 线程管理

### 3.1 创建与等待

```c
#include <stdio.h>
#include <threads.h>
#include <time.h>

// 线程函数
int thread_func(void *arg) {
    int id = *(int*)arg;
    printf("线程 %d 开始执行\n", id);

    // 模拟工作
    for (int i = 0; i < 3; i++) {
        printf("线程 %d: 工作中... (%d/3)\n", id, i + 1);
        thrd_sleep(&(struct timespec){.tv_sec = 1}, NULL);
    }

    printf("线程 %d 完成\n", id);
    return id * 10;  // 返回值
}

int main(void) {
    thrd_t thread1, thread2;
    int id1 = 1, id2 = 2;

    // 创建线程
    if (thrd_create(&thread1, thread_func, &id1) != thrd_success) {
        fprintf(stderr, "创建线程1失败\n");
        return 1;
    }

    if (thrd_create(&thread2, thread_func, &id2) != thrd_success) {
        fprintf(stderr, "创建线程2失败\n");
        return 1;
    }

    // 等待线程完成
    int result1, result2;
    thrd_join(thread1, &result1);
    thrd_join(thread2, &result2);

    printf("线程1返回: %d\n", result1);
    printf("线程2返回: %d\n", result2);

    return 0;
}
```

### 3.2 线程分离

```c
#include <stdio.h>
#include <threads.h>

int detached_thread(void *arg) {
    (void)arg;
    printf("分离线程执行中\n");
    return 0;
}

int main(void) {
    thrd_t thr;

    thrd_create(&thr, detached_thread, NULL);
    thrd_detach(thr);  // 分离线程，资源自动回收

    // 主线程不需要join
    printf("主线程继续\n");

    // 给分离线程一点执行时间
    thrd_sleep(&(struct timespec){.tv_sec = 1}, NULL);

    return 0;
}
```

## 4. 互斥锁

### 4.1 基本互斥

```c
#include <stdio.h>
#include <threads.h>

mtx_t mutex;
int shared_counter = 0;

int worker(void *arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 10000; i++) {
        mtx_lock(&mutex);
        shared_counter++;
        mtx_unlock(&mutex);
    }

    printf("线程 %d 完成\n", id);
    return 0;
}

int main(void) {
    thrd_t threads[4];
    int ids[4] = {1, 2, 3, 4};

    // 初始化互斥锁
    mtx_init(&mutex, mtx_plain);

    // 创建线程
    for (int i = 0; i < 4; i++) {
        thrd_create(&threads[i], worker, &ids[i]);
    }

    // 等待所有线程
    for (int i = 0; i < 4; i++) {
        thrd_join(threads[i], NULL);
    }

    printf("最终计数: %d (期望: 40000)\n", shared_counter);

    // 销毁互斥锁
    mtx_destroy(&mutex);

    return 0;
}
```

### 4.2 带超时的互斥锁

```c
#include <stdio.h>
#include <threads.h>
#include <time.h>

mtx_t timed_mutex;

int timed_worker(void *arg) {
    (void)arg;

    // 获取当前时间并加上超时
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    ts.tv_sec += 2;  // 2秒超时

    if (mtx_timedlock(&timed_mutex, &ts) == thrd_success) {
        printf("获取到锁\n");
        mtx_unlock(&timed_mutex);
    } else {
        printf("获取锁超时\n");
    }

    return 0;
}

int main(void) {
    thrd_t thr;
    mtx_init(&timed_mutex, mtx_timed);

    thrd_create(&thr, timed_worker, NULL);
    thrd_join(thr, NULL);

    mtx_destroy(&timed_mutex);
    return 0;
}
```

## 5. 条件变量

```c
#include <stdio.h>
#include <threads.h>

mtx_t cond_mutex;
cnd_t cond_var;
int data_ready = 0;
int shared_data = 0;

int producer(void *arg) {
    (void)arg;

    mtx_lock(&cond_mutex);

    // 生产数据
    shared_data = 42;
    data_ready = 1;
    printf("生产者: 数据已准备好\n");

    // 通知等待的线程
    cnd_signal(&cond_var);

    mtx_unlock(&cond_mutex);
    return 0;
}

int consumer(void *arg) {
    (void)arg;

    mtx_lock(&cond_mutex);

    // 等待数据就绪
    while (!data_ready) {
        printf("消费者: 等待数据...\n");
        cnd_wait(&cond_var, &cond_mutex);
    }

    printf("消费者: 收到数据 = %d\n", shared_data);

    mtx_unlock(&cond_mutex);
    return 0;
}

int main(void) {
    thrd_t prod, cons;

    mtx_init(&cond_mutex, mtx_plain);
    cnd_init(&cond_var);

    // 先启动消费者（会等待）
    thrd_create(&cons, consumer, NULL);
    thrd_create(&prod, producer, NULL);

    thrd_join(prod, NULL);
    thrd_join(cons, NULL);

    cnd_destroy(&cond_var);
    mtx_destroy(&cond_mutex);

    return 0;
}
```

## 6. 线程本地存储（TSS）

```c
#include <stdio.h>
#include <threads.h>
#include <stdlib.h>

tss_t thread_local_key;

// TSS析构函数
void cleanup(void *data) {
    printf("清理线程本地数据: %d\n", *(int*)data);
    free(data);
}

int thread_func(void *arg) {
    int id = *(int*)arg;

    // 设置线程本地数据
    int *local_data = malloc(sizeof(int));
    *local_data = id * 100;
    tss_set(thread_local_key, local_data);

    // 获取线程本地数据
    int *retrieved = tss_get(thread_local_key);
    printf("线程 %d: 本地数据 = %d\n", id, *retrieved);

    return 0;
}

int main(void) {
    thrd_t threads[3];
    int ids[3] = {1, 2, 3};

    // 创建TSS键
    tss_create(&thread_local_key, cleanup);

    for (int i = 0; i < 3; i++) {
        thrd_create(&threads[i], thread_func, &ids[i]);
    }

    for (int i = 0; i < 3; i++) {
        thrd_join(threads[i], NULL);
    }

    tss_delete(thread_local_key);

    return 0;
}
```

## 7. call_once - 一次性调用

```c
#include <stdio.h>
#include <threads.h>

once_flag init_flag = ONCE_FLAG_INIT;

void init_function(void) {
    printf("初始化函数只执行一次\n");
}

int thread_func(void *arg) {
    int id = *(int*)arg;
    printf("线程 %d: 调用call_once\n", id);
    call_once(&init_flag, init_function);
    return 0;
}

int main(void) {
    thrd_t threads[4];
    int ids[4] = {1, 2, 3, 4};

    for (int i = 0; i < 4; i++) {
        thrd_create(&threads[i], thread_func, &ids[i]);
    }

    for (int i = 0; i < 4; i++) {
        thrd_join(threads[i], NULL);
    }

    return 0;
}
```

## 8. 重要注意事项

> **要点一**：C11线程API比POSIX pthreads更简洁，但功能也更少。

> **要点二**：互斥锁有四种类型：`mtx_plain`、`mtx_timed`、`mtx_recursive`、`mtx_plain | mtx_recursive`。

> **要点三**：`cnd_wait` 会在等待前自动释放互斥锁，返回前重新获取。

> **要点四**：线程函数的返回值通过 `thrd_join` 的第二个参数获取。

> **要点五**：`call_once` 保证初始化函数在多线程环境中只执行一次。

> **要点六**：`thrd_sleep` 类似于 `nanosleep`，但更简单。

> **要点七**：并非所有编译器都完整支持 `<threads.h>`，可能需要检查编译器文档。

> **要点八**：线程本地存储（TSS）提供了每个线程独立的数据存储。
