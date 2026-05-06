# stdatomic.h - 原子操作（C11）

## 1. 概述

`<stdatomic.h>`（C11引入）提供了原子类型和原子操作，用于无锁并发编程。原子操作保证在多线程环境中的操作是不可分割的。

## 2. 原子类型

```c
#include <stdatomic.h>

// 原子类型定义
atomic_int      // 原子int
atomic_uint     // 原子unsigned int
atomic_long     // 原子long
atomic_bool     // 原子bool
atomic_char     // 原子char
// 以及 _Atomic(type) 通用形式

// 泛型宏
atomic_int counter;
// 等价于: _Atomic int counter;
```

## 3. 基本原子操作

### 3.1 加载与存储

```c
#include <stdio.h>
#include <stdatomic.h>
#include <threads.h>

atomic_int shared_counter = ATOMIC_VAR_INIT(0);

int increment_thread(void *arg) {
    (void)arg;
    for (int i = 0; i < 100000; i++) {
        // 原子加载-修改-存储
        int old_val = atomic_load(&shared_counter);
        while (!atomic_compare_exchange_weak(&shared_counter,
                                              &old_val, old_val + 1)) {
            // CAS失败，重试
        }
    }
    return 0;
}

int main(void) {
    thrd_t threads[4];

    for (int i = 0; i < 4; i++) {
        thrd_create(&threads[i], increment_thread, NULL);
    }

    for (int i = 0; i < 4; i++) {
        thrd_join(threads[i], NULL);
    }

    printf("最终计数: %d (期望: 400000)\n",
           atomic_load(&shared_counter));

    return 0;
}
```

### 3.2 原子算术操作

```c
#include <stdio.h>
#include <stdatomic.h>

int main(void) {
    atomic_int val = ATOMIC_VAR_INIT(100);

    // 原子加法
    int old = atomic_fetch_add(&val, 10);
    printf("fetch_add(10): old=%d, new=%d\n", old,
           atomic_load(&val));  // old=100, new=110

    // 原子减法
    old = atomic_fetch_sub(&val, 5);
    printf("fetch_sub(5): old=%d, new=%d\n", old,
           atomic_load(&val));  // old=110, new=105

    // 原子按位操作
    atomic_uint flags = ATOMIC_VAR_INIT(0xFF);
    old = atomic_fetch_or(&flags, 0x100);
    printf("fetch_or: old=0x%X, new=0x%X\n", old,
           atomic_load(&flags));

    old = atomic_fetch_and(&flags, 0x0F);
    printf("fetch_and: old=0x%X, new=0x%X\n", old,
           atomic_load(&flags));

    old = atomic_fetch_xor(&flags, 0x01);
    printf("fetch_xor: old=0x%X, new=0x%X\n", old,
           atomic_load(&flags));

    return 0;
}
```

## 4. 内存序（Memory Order）

```c
#include <stdatomic.h>

// 内存序选项
memory_order_relaxed    // 最宽松，只保证原子性
memory_order_consume    // 数据依赖排序
memory_order_acquire    // 获取语义
memory_order_release    // 释放语义
memory_order_acq_rel    // 获取+释放
memory_order_seq_cst    // 顺序一致性（默认）
```

### 4.1 各内存序的含义

```c
#include <stdio.h>
#include <stdatomic.h>
#include <threads.h>

atomic_int data = ATOMIC_VAR_INIT(0);
atomic_bool ready = ATOMIC_VAR_INIT(false);

int writer(void *arg) {
    (void)arg;

    // 写入数据
    data = 42;

    // 使用release语义：保证data的写入在ready之前可见
    atomic_store_explicit(&ready, true, memory_order_release);

    return 0;
}

int reader(void *arg) {
    (void)arg;

    // 使用acquire语义：等待ready变为true
    while (!atomic_load_explicit(&ready, memory_order_acquire)) {
        // 自旋等待
    }

    // 此时保证能看到data=42
    printf("读取数据: %d\n", data);

    return 0;
}

int main(void) {
    thrd_t w, r;

    thrd_create(&r, reader, NULL);
    thrd_create(&w, writer, NULL);

    thrd_join(w, NULL);
    thrd_join(r, NULL);

    return 0;
}
```

### 4.2 宽松内存序

```c
#include <stdio.h>
#include <stdatomic.h>
#include <threads.h>

// 适用于简单的计数器（不需要同步其他数据）
atomic_int stats_counter = ATOMIC_VAR_INIT(0);

int worker(void *arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 10000; i++) {
        // relaxed: 只保证原子性，不保证顺序
        atomic_fetch_add_explicit(&stats_counter, 1,
                                   memory_order_relaxed);
    }

    printf("线程 %d 完成计数\n", id);
    return 0;
}

int main(void) {
    thrd_t threads[4];
    int ids[4] = {1, 2, 3, 4};

    for (int i = 0; i < 4; i++) {
        thrd_create(&threads[i], worker, &ids[i]);
    }

    for (int i = 0; i < 4; i++) {
        thrd_join(threads[i], NULL);
    }

    printf("总计数: %d\n",
           atomic_load_explicit(&stats_counter, memory_order_relaxed));

    return 0;
}
```

## 5. CAS操作

```c
#include <stdio.h>
#include <stdatomic.h>

int main(void) {
    atomic_int val = ATOMIC_VAR_INIT(100);

    // compare_exchange_weak: 可能虚假失败
    int expected = 100;
    int desired = 200;

    if (atomic_compare_exchange_weak(&val, &expected, desired)) {
        printf("CAS成功: %d -> %d\n", expected, desired);
    }

    // compare_exchange_strong: 不会虚假失败
    expected = 200;
    desired = 300;

    if (atomic_compare_exchange_strong(&val, &expected, desired)) {
        printf("强CAS成功: %d -> %d\n", expected, desired);
    }

    // CAS失败时，expected被更新为当前值
    expected = 999;
    if (!atomic_compare_exchange_strong(&val, &expected, 400)) {
        printf("CAS失败: 当前值为 %d\n", expected);  // 300
    }

    return 0;
}
```

## 6. 原子标志

```c
#include <stdio.h>
#include <stdatomic.h>

int main(void) {
    atomic_flag lock = ATOMIC_FLAG_INIT;

    // 自旋锁实现
    // 获取锁
    while (atomic_flag_test_and_set(&lock)) {
        // 自旋等待
    }

    // 临界区
    printf("进入临界区\n");

    // 释放锁
    atomic_flag_clear(&lock);

    return 0;
}
```

## 7. 重要注意事项

> **要点一**：默认的内存序是 `memory_order_seq_cst`（顺序一致性），最安全但性能最低。

> **要点二**：`memory_order_relaxed` 只保证原子性，不保证操作顺序，适用于简单的计数器。

> **要点三**：`compare_exchange_weak` 可能虚假失败（即使值匹配也可能返回false），应放在循环中使用。

> **要点四**：`compare_exchange_strong` 不会虚假失败，但可能较慢。

> **要点五**：`atomic_flag` 是唯一保证无锁的原子类型，其他原子类型可能使用锁实现。

> **要点六**：原子操作的正确使用需要深入理解内存模型，建议使用 `memory_order_seq_cst` 除非有明确的性能需求。

> **要点七**：`ATOMIC_VAR_INIT` 在C17中已被弃用（C23中移除），可直接初始化。

> **要点八**：`atomic_is_lock_free` 可以查询某个原子类型是否无锁。
