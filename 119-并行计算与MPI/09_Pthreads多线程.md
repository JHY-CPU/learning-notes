# Pthreads 多线程编程

POSIX Threads（Pthreads）是 POSIX 标准定义的 C 语言多线程 API，提供底层线程控制，是理解共享内存并行的基础。

## 1. 线程创建与回收

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// 线程函数：必须返回 void*，参数为 void*
void *thread_func(void *arg) {
    int thread_id = *(int *)arg;
    printf("线程 %d 正在运行, ID = %lu\n", thread_id, (unsigned long)pthread_self());
    return NULL;
}

int main() {
    int NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    // 创建线程
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        /*
         * int pthread_create(pthread_t *thread,
         *                    const pthread_attr_t *attr,
         *                    void *(*start_routine)(void *),
         *                    void *arg);
         */
        int rc = pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
        if (rc != 0) {
            fprintf(stderr, "创建线程失败: %d\n", rc);
            exit(1);
        }
    }

    // 等待所有线程完成
    for (int i = 0; i < NUM_THREADS; i++) {
        void *retval;
        pthread_join(threads[i], &retval);
        printf("线程 %d 已退出\n", i);
    }

    printf("所有线程已完成\n");
    return 0;
}
```

编译：`gcc -pthread -o threads threads.c`

## 2. 互斥锁（Mutex）

互斥锁确保同一时刻只有一个线程访问共享资源，防止数据竞争。

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 10
#define NUM_INCREMENTS 100000

int counter = 0;
pthread_mutex_t lock;

void *increment(void *arg) {
    for (int i = 0; i < NUM_INCREMENTS; i++) {
        pthread_mutex_lock(&lock);
        counter++;           // 临界区：只有一个线程能进入
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];

    // 初始化互斥锁
    pthread_mutex_init(&lock, NULL);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, increment, NULL);

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    printf("counter = %d (期望 %d)\n", counter, NUM_THREADS * NUM_INCREMENTS);

    // 销毁互斥锁
    pthread_mutex_destroy(&lock);
    return 0;
}
```

## 3. 条件变量（Condition Variable）

条件变量用于线程间的事件通知机制：一个线程等待某个条件，另一个线程在条件满足时唤醒它。

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define BUFFER_SIZE 5
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 2

int buffer[BUFFER_SIZE];
int count = 0;  // 缓冲区中的元素数量

pthread_mutex_t mutex;
pthread_cond_t not_full;   // 缓冲区未满
pthread_cond_t not_empty;  // 缓冲区非空

void *producer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 10; i++) {
        pthread_mutex_lock(&mutex);

        // 等待缓冲区未满
        while (count == BUFFER_SIZE) {
            printf("生产者%d: 缓冲区满，等待...\n", id);
            pthread_cond_wait(&not_full, &mutex);
            // pthread_cond_wait 做了三件事：
            // 1. 释放 mutex  2. 阻塞等待信号  3. 收到信号后重新获取 mutex
        }

        buffer[count++] = i;
        printf("生产者%d: 放入 %d (缓冲区有 %d 个)\n", id, i, count);

        pthread_cond_signal(&not_empty);  // 通知消费者
        pthread_mutex_unlock(&mutex);

        usleep(rand() % 100000);
    }
    return NULL;
}

void *consumer(void *arg) {
    int id = *(int *)arg;
    for (int i = 0; i < 10; i++) {
        pthread_mutex_lock(&mutex);

        while (count == 0) {
            printf("消费者%d: 缓冲区空，等待...\n", id);
            pthread_cond_wait(&not_empty, &mutex);
        }

        int item = buffer[--count];
        printf("消费者%d: 取出 %d (缓冲区有 %d 个)\n", id, item, count);

        pthread_cond_signal(&not_full);  // 通知生产者
        pthread_mutex_unlock(&mutex);

        usleep(rand() % 200000);
    }
    return NULL;
}

int main() {
    pthread_t prods[NUM_PRODUCERS], cons[NUM_CONSUMERS];
    int prod_ids[NUM_PRODUCERS], cons_ids[NUM_CONSUMERS];

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&not_full, NULL);
    pthread_cond_init(&not_empty, NULL);

    for (int i = 0; i < NUM_PRODUCERS; i++) {
        prod_ids[i] = i;
        pthread_create(&prods[i], NULL, producer, &prod_ids[i]);
    }
    for (int i = 0; i < NUM_CONSUMERS; i++) {
        cons_ids[i] = i;
        pthread_create(&cons[i], NULL, consumer, &cons_ids[i]);
    }

    for (int i = 0; i < NUM_PRODUCERS; i++)
        pthread_join(prods[i], NULL);
    for (int i = 0; i < NUM_CONSUMERS; i++)
        pthread_join(cons[i], NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&not_full);
    pthread_cond_destroy(&not_empty);

    printf("生产者-消费者演示完成\n");
    return 0;
}
```

## 4. 线程安全数据结构——并发链表

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// 线程安全的并发链表
typedef struct Node {
    int data;
    struct Node *next;
} Node;

typedef struct {
    Node *head;
    pthread_mutex_t lock;
} ConcurrentList;

void list_init(ConcurrentList *list) {
    list->head = NULL;
    pthread_mutex_init(&list->lock, NULL);
}

// 头部插入（线程安全）
void list_insert(ConcurrentList *list, int data) {
    Node *new_node = (Node *)malloc(sizeof(Node));
    new_node->data = data;

    pthread_mutex_lock(&list->lock);
    new_node->next = list->head;
    list->head = new_node;
    pthread_mutex_unlock(&list->lock);
}

// 查找（线程安全）
int list_find(ConcurrentList *list, int data) {
    pthread_mutex_lock(&list->lock);
    Node *curr = list->head;
    while (curr) {
        if (curr->data == data) {
            pthread_mutex_unlock(&list->lock);
            return 1;
        }
        curr = curr->next;
    }
    pthread_mutex_unlock(&list->lock);
    return 0;
}

// 删除（线程安全）
int list_delete(ConcurrentList *list, int data) {
    pthread_mutex_lock(&list->lock);
    Node *curr = list->head;
    Node *prev = NULL;
    while (curr) {
        if (curr->data == data) {
            if (prev) prev->next = curr->next;
            else list->head = curr->next;
            free(curr);
            pthread_mutex_unlock(&list->lock);
            return 1;
        }
        prev = curr;
        curr = curr->next;
    }
    pthread_mutex_unlock(&list->lock);
    return 0;
}

void list_destroy(ConcurrentList *list) {
    pthread_mutex_lock(&list->lock);
    Node *curr = list->head;
    while (curr) {
        Node *tmp = curr;
        curr = curr->next;
        free(tmp);
    }
    list->head = NULL;
    pthread_mutex_unlock(&list->lock);
    pthread_mutex_destroy(&list->lock);
}

// 多线程测试
ConcurrentList g_list;

void *worker(void *arg) {
    int base = *(int *)arg;
    for (int i = 0; i < 1000; i++) {
        list_insert(&g_list, base + i);
    }
    return NULL;
}

int main() {
    list_init(&g_list);

    int NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    int bases[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        bases[i] = i * 10000;
        pthread_create(&threads[i], NULL, worker, &bases[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    // 验证：统计节点数
    pthread_mutex_lock(&g_list.lock);
    int cnt = 0;
    Node *p = g_list.head;
    while (p) { cnt++; p = p->next; }
    pthread_mutex_unlock(&g_list.lock);
    printf("链表中有 %d 个节点 (期望 %d)\n", cnt, NUM_THREADS * 1000);

    list_destroy(&g_list);
    return 0;
}
```

## 5. 线程属性

```c
pthread_attr_t attr;
pthread_attr_init(&attr);

// 设置分离状态（分离线程不需要 join）
pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

// 设置栈大小
pthread_attr_setstacksize(&attr, 2 * 1024 * 1024);  // 2MB

pthread_t thread;
pthread_create(&thread, &attr, thread_func, NULL);

pthread_attr_destroy(&attr);
```
