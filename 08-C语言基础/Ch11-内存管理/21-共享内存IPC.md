# 共享内存IPC

## 一、共享内存概述

共享内存是最快的进程间通信(IPC)方式。多个进程将同一块物理内存映射到各自的虚拟地址空间，直接读写共享数据。

```
共享内存示意:

  进程A虚拟空间                 进程B虚拟空间
  ┌───────────┐               ┌───────────┐
  │ 0x1000    │               │ 0x2000    │
  │    ↓      │               │    ↓      │
  │  映射区域  │               │  映射区域  │
  └─────┬─────┘               └─────┬─────┘
        │                           │
        └───────────┬───────────────┘
                    ↓
           ┌─────────────────┐
           │   物理内存       │  ← 同一块内存
           │   (共享区域)     │
           └─────────────────┘
```

## 二、System V共享内存

### 2.1 基本API

```c
#include <sys/ipc.h>
#include <sys/shm.h>

int shmget(key_t key, size_t size, int shmflg);  // 创建/获取
void *shmat(int shmid, const void *shmaddr, int shmflg);  // 附加
int shmdt(const void *shmaddr);  // 分离
int shmctl(int shmid, int cmd, struct shmid_ds *buf);  // 控制
```

### 2.2 写入端

```c
/* shm_writer.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define SHM_KEY  12345
#define SHM_SIZE 4096

typedef struct {
    int ready;
    char message[256];
    int counter;
} SharedData;

int main(void) {
    // 创建共享内存
    int shmid = shmget(SHM_KEY, SHM_SIZE, IPC_CREAT | 0666);
    if (shmid < 0) {
        perror("shmget");
        return 1;
    }
    printf("共享内存ID: %d\n", shmid);

    // 附加到进程地址空间
    SharedData *shared = (SharedData *)shmat(shmid, NULL, 0);
    if (shared == (void *)-1) {
        perror("shmat");
        return 1;
    }

    // 写入数据
    shared->ready = 0;
    shared->counter = 0;

    for (int i = 0; i < 5; i++) {
        sprintf(shared->message, "消息 #%d: Hello from Writer!", i);
        shared->counter = i;
        shared->ready = 1;
        printf("写入: %s\n", shared->message);

        // 等待读取
        while (shared->ready) {
            usleep(10000);  // 10ms
        }
    }

    // 分离
    shmdt(shared);

    // 删除共享内存
    shmctl(shmid, IPC_RMID, NULL);
    printf("共享内存已删除\n");

    return 0;
}
```

### 2.3 读取端

```c
/* shm_reader.c */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define SHM_KEY  12345

typedef struct {
    int ready;
    char message[256];
    int counter;
} SharedData;

int main(void) {
    // 获取已存在的共享内存
    int shmid = shmget(SHM_KEY, 0, 0);
    if (shmid < 0) {
        perror("shmget (请先运行writer)");
        return 1;
    }

    SharedData *shared = (SharedData *)shmat(shmid, NULL, 0);
    if (shared == (void *)-1) {
        perror("shmat");
        return 1;
    }

    printf("连接到共享内存ID: %d\n", shmid);

    // 读取数据
    for (int i = 0; i < 5; i++) {
        // 等待数据就绪
        while (!shared->ready) {
            usleep(10000);
        }

        printf("读取: counter=%d, %s\n",
               shared->counter, shared->message);

        shared->ready = 0;  // 通知写入端已读取
    }

    shmdt(shared);
    printf("分离共享内存\n");

    return 0;
}
```

## 三、mmap共享内存

```c
/* mmap_writer.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef struct {
    int counter;
    char buffer[256];
} SharedMem;

int main(void) {
    const char *shm_name = "/my_shm";

    // 创建共享内存对象
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) { perror("shm_open"); return 1; }

    ftruncate(fd, sizeof(SharedMem));

    // 映射
    SharedMem *shm = mmap(NULL, sizeof(SharedMem),
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, 0);
    if (shm == MAP_FAILED) { perror("mmap"); return 1; }

    // 写入数据
    for (int i = 0; i < 5; i++) {
        shm->counter = i;
        sprintf(shm->buffer, "mmap共享消息 #%d", i);
        printf("写入: %s\n", shm->buffer);
        sleep(1);
    }

    munmap(shm, sizeof(SharedMem));
    close(fd);
    shm_unlink(shm_name);
    return 0;
}
```

## 四、关键要点

> **共享内存IPC要点**
> 1. 共享内存是最快的IPC方式，无需内核数据拷贝
> 2. 需要信号量或互斥锁同步访问
> 3. System V: shmget/shmat/shmdt/shmctl
> 4. POSIX: shm_open/mmap/shm_unlink
> 5. 共享内存不会自动清理，需要手动删除
> 6. 适合大量数据的进程间传输
