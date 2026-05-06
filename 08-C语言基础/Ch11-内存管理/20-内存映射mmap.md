# 内存映射mmap

## 一、mmap概念

`mmap`将文件或设备映射到进程的虚拟地址空间，使文件操作变成内存操作。也可以创建匿名映射用于进程间共享内存。

```
mmap映射示意:

  文件系统                        虚拟地址空间
  ┌──────────┐                   ┌──────────────┐
  │ 文件A    │    mmap映射       │              │
  │ (磁盘)   │ ═══════════════→ │  映射区域     │
  │          │                   │  (内存访问    │
  └──────────┘                   │   等同于文件  │
                                 │   读写)       │
                                 └──────────────┘
```

## 二、mmap函数原型

```c
#include <sys/mman.h>

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void *addr, size_t length);
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `addr` | 期望的映射地址(NULL=内核自动选择) |
| `length` | 映射长度(字节) |
| `prot` | 保护标志: PROT_READ, PROT_WRITE, PROT_EXEC, PROT_NONE |
| `flags` | 映射类型: MAP_SHARED, MAP_PRIVATE, MAP_ANONYMOUS |
| `fd` | 文件描述符(匿名映射为-1) |
| `offset` | 文件偏移(必须是页大小的整数倍) |

## 三、文件映射示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

void file_mapping_demo(void) {
    const char *filename = "mmap_test.txt";

    // 1. 创建测试文件
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("open"); return; }

    const char *text = "Hello, mmap! 这是内存映射测试文件。\n";
    write(fd, text, strlen(text));

    // 调整文件大小
    ftruncate(fd, 4096);

    // 2. 内存映射
    char *mapped = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }

    // 3. 像操作内存一样操作文件
    printf("映射内容: %s\n", mapped);

    // 修改映射区域 → 直接写入文件
    strcpy(mapped + strlen(text), "mmap追加的内容！\n");
    printf("修改后: %s\n", mapped);

    // 4. 解除映射
    munmap(mapped, 4096);
    close(fd);

    // 验证文件已被修改
    fd = open(filename, O_RDONLY);
    char buf[512];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        printf("文件内容: %s\n", buf);
    }
    close(fd);

    unlink(filename);
}

int main(void) {
    file_mapping_demo();
    return 0;
}
```

## 四、匿名映射

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

void anonymous_mapping_demo(void) {
    // 匿名映射：不关联文件，用于分配大块内存
    size_t size = 4096 * 100;  // 400KB

    void *region = mmap(NULL, size,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1, 0);
    if (region == MAP_FAILED) {
        perror("mmap");
        return;
    }

    // 使用匿名映射内存
    int *data = (int *)region;
    for (int i = 0; i < 1000; i++) {
        data[i] = i * i;
    }

    printf("匿名映射前10个值:\n");
    for (int i = 0; i < 10; i++) {
        printf("  data[%d] = %d\n", i, data[i]);
    }

    munmap(region, size);
}

int main(void) {
    anonymous_mapping_demo();
    return 0;
}
```

## 五、mmap vs malloc

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

void compare_mmap_malloc(void) {
    size_t size = 1024 * 1024 * 100;  // 100MB

    // malloc (大分配内部也用mmap)
    clock_t start = clock();
    void *p1 = malloc(size);
    clock_t end = clock();
    printf("malloc 100MB: %.6f 秒\n", (double)(end - start) / CLOCKS_PER_SEC);
    free(p1);

    // mmap
    start = clock();
    void *p2 = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    end = clock();
    printf("mmap 100MB:   %.6f 秒\n", (double)(end - start) / CLOCKS_PER_SEC);
    munmap(p2, size);

    printf("\n适用场景:\n");
    printf("  malloc: 通用小块内存分配\n");
    printf("  mmap: 大块内存、文件映射、进程间共享\n");
}

int main(void) {
    compare_mmap_malloc();
    return 0;
}
```

## 六、关键要点

> **mmap要点**
> 1. mmap将文件映射到内存，可直接通过指针读写文件
> 2. MAP_SHARED修改会写回文件，MAP_PRIVATE是写时复制
> 3. 匿名映射(MAP_ANONYMOUS)不关联文件，类似大块malloc
> 4. 大文件处理用mmap避免多次read/write系统调用
> 5. 进程间通信可用MAP_SHARED实现共享内存
> 6. 用munmap解除映射，映射区域自动按页对齐
> 7. 文件偏移必须是页大小(通常4KB)的整数倍
