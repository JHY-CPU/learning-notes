# 进程间通信（IPC）

## 1. IPC 概述

进程间通信（Inter-Process Communication）是指在不同进程之间传递数据或信号的机制。Linux提供了多种IPC方式，各有适用场景。

| IPC方式 | 特点 | 适用场景 |
|---------|------|----------|
| 管道 | 半双工、亲缘关系进程 | 父子进程、简单数据流 |
| 命名管道(FIFO) | 无亲缘关系可用 | 无关进程间通信 |
| 消息队列 | 结构化消息、有类型 | 异步消息传递 |
| 共享内存 | 最快、需要同步 | 大数据量高性能 |
| 信号量 | 同步控制 | 资源计数、互斥 |
| Socket | 网络/本地通用 | 网络通信、跨机器 |

## 2. 匿名管道（Pipe）

管道是最古老的IPC方式，只能用于有共同祖先的进程间通信，数据是单向的。

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

// 基本管道：父进程向子进程发送数据
void pipe_basic_demo() {
    int pipefd[2]; // pipefd[0]读端，pipefd[1]写端

    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // 子进程：读取数据
        close(pipefd[1]); // 关闭写端

        char buf[256];
        ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            printf("子进程收到: %s\n", buf);
        }

        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    } else {
        // 父进程：发送数据
        close(pipefd[0]); // 关闭读端

        const char *msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);
        wait(NULL);
    }
}
```

### 管道实现进程间命令管道

```c
// 模拟 "ls | grep .c" 的效果
void pipe_command_demo() {
    int pipefd[2];
    pipe(pipefd);

    pid_t pid1 = fork();
    if (pid1 == 0) {
        // 子进程1：执行 ls
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO); // stdout -> 管道写端
        close(pipefd[1]);
        execlp("ls", "ls", NULL);
        perror("execlp ls");
        _exit(1);
    }

    pid_t pid2 = fork();
    if (pid2 == 0) {
        // 子进程2：执行 grep
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO); // stdin <- 管道读端
        close(pipefd[0]);
        execlp("grep", "grep", ".c", NULL);
        perror("execlp grep");
        _exit(1);
    }

    close(pipefd[0]);
    close(pipefd[1]);
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);
}
```

### 管道的限制与注意事项

- 管道缓冲区大小通常为64KB（`/proc/sys/fs/pipe-max-size` 可调整）
- 写满管道时 `write` 会阻塞
- 读空管道时 `read` 会阻塞
- 所有写端关闭后，读端 `read` 返回0
- 所有读端关闭后，写端 `write` 收到 `SIGPIPE` 信号

## 3. 命名管道（FIFO）

命名管道存在于文件系统中，无关进程也可以通过文件路径访问。

```c
// 创建FIFO（也可用命令 mkfifo /tmp/myfifo）
#include <sys/stat.h>

void fifo_demo() {
    const char *fifo_path = "/tmp/myfifo";

    // 创建命名管道（如果不存在）
    if (mkfifo(fifo_path, 0666) == -1 && errno != EEXIST) {
        perror("mkfifo");
        exit(EXIT_FAILURE);
    }

    // 写入端（在另一个终端运行读取端程序）
    printf("打开FIFO写入端...\n");
    int fd = open(fifo_path, O_WRONLY); // 阻塞直到有读者
    printf("已连接\n");

    char msg[] = "Hello via FIFO!";
    write(fd, msg, strlen(msg));
    close(fd);

    unlink(fifo_path); // 删除FIFO文件
}
```

## 4. 消息队列（Message Queue）

消息队列提供结构化的消息传递，支持消息类型和优先级。

```c
#include <sys/msg.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// 消息结构体（必须以long mtype开头）
typedef struct {
    long mtype;      // 消息类型（必须>0）
    char mtext[256]; // 消息内容
} Message;

void msg_queue_sender() {
    // 创建消息队列
    key_t key = ftok("/tmp", 'A');
    int msqid = msgget(key, IPC_CREAT | 0666);

    Message msg;
    msg.mtype = 1; // 类型1的消息
    strcpy(msg.mtext, "这是类型1的消息");

    // 发送消息（最后一个参数为标志，0=阻塞，IPC_NOWAIT=非阻塞）
    if (msgsnd(msqid, &msg, strlen(msg.mtext) + 1, 0) == -1) {
        perror("msgsnd");
    }

    // 发送类型2的消息（优先级更高）
    msg.mtype = 2;
    strcpy(msg.mtext, "这是类型2的消息（优先级更高）");
    msgsnd(msqid, &msg, strlen(msg.mtext) + 1, 0);

    printf("消息已发送\n");
}

void msg_queue_receiver() {
    key_t key = ftok("/tmp", 'A');
    int msqid = msgget(key, 0666);

    Message msg;

    // 接收类型2的消息（优先处理）
    if (msgrcv(msqid, &msg, sizeof(msg.mtext), 2, 0) != -1) {
        printf("收到类型%ld: %s\n", msg.mtype, msg.mtext);
    }

    // 接收类型1的消息
    if (msgrcv(msqid, &msg, sizeof(msg.mtext), 1, 0) != -1) {
        printf("收到类型%ld: %s\n", msg.mtype, msg.mtext);
    }

    // 删除消息队列
    msgctl(msqid, IPC_RMID, NULL);
}
```

## 5. 共享内存（Shared Memory）

共享内存是最快的IPC方式——多个进程直接映射同一块物理内存，无需数据拷贝。但需要额外的同步机制。

```c
#include <sys/shm.h>
#include <sys/sem.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int ready;         // 简单的同步标志
    char data[1024];   // 共享数据区
} SharedData;

void shared_memory_writer() {
    // 创建共享内存段
    key_t key = ftok("/tmp", 'S');
    int shmid = shmget(key, sizeof(SharedData), IPC_CREAT | 0666);

    // 附加到进程地址空间
    SharedData *shared = (SharedData*)shmat(shmid, NULL, 0);

    // 写入数据
    shared->ready = 0;
    strcpy(shared->data, "共享内存中的数据");
    shared->ready = 1; // 标记数据就绪

    printf("写入完成，等待读取...\n");
    while (shared->ready == 1) {
        usleep(100000);
    }

    // 分离共享内存
    shmdt(shared);

    // 删除共享内存段
    shmctl(shmid, IPC_RMID, NULL);
    printf("写入端退出\n");
}

void shared_memory_reader() {
    key_t key = ftok("/tmp", 'S');
    int shmid = shmget(key, sizeof(SharedData), 0666);

    SharedData *shared = (SharedData*)shmat(shmid, NULL, 0);

    // 等待数据就绪
    while (shared->ready == 0) {
        usleep(100000);
    }

    printf("读取到: %s\n", shared->data);
    shared->ready = 0; // 通知写入端

    shmdt(shared);
}
```

## 6. 信号量用于IPC同步

System V信号量用于控制对共享资源的访问。

```c
#include <sys/sem.h>

// 信号量操作联合体
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short *array;
};

void semaphore_ipc_demo() {
    key_t key = ftok("/tmp", 'M');
    int semid = semget(key, 1, IPC_CREAT | 0666);

    // 初始化信号量为1（互斥信号量）
    union semun arg;
    arg.val = 1;
    semctl(semid, 0, SETVAL, arg);

    // P操作（等待）
    struct sembuf p_op = {0, -1, 0}; // sem_num=0, sem_op=-1, sem_flg=0
    semop(semid, &p_op, 1);

    // 临界区
    printf("进入临界区 (PID=%d)\n", getpid());
    sleep(2);

    // V操作（释放）
    struct sembuf v_op = {0, 1, 0};
    semop(semid, &v_op, 1);

    printf("离开临界区\n");

    // 清理
    semctl(semid, 0, IPC_RMID);
}
```

## 7. POSIX 信号量

POSIX信号量提供更简洁的API，支持命名和匿名两种形式。

```c
#include <semaphore.h>
#include <fcntl.h>    // O_CREAT, O_EXCL
#include <sys/stat.h> // mode constants

void posix_named_semaphore() {
    // 命名信号量（可用于无关进程）
    sem_t *sem = sem_open("/my_semaphore", O_CREAT, 0666, 1);

    sem_wait(sem); // P操作
    printf("临界区 (PID=%d)\n", getpid());
    sleep(1);
    sem_post(sem); // V操作

    sem_close(sem);
    sem_unlink("/my_semaphore"); // 删除命名信号量
}
```

## 8. Unix Domain Socket

Unix域socket可用于同一台机器上的进程通信，支持流式和数据报两种模式。

```c
#include <sys/socket.h>
#include <sys/un.h>

#define SOCKET_PATH "/tmp/unix_socket"

void unix_socket_server() {
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);
    unlink(SOCKET_PATH);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 5);

    printf("等待连接...\n");
    int client_fd = accept(server_fd, NULL, NULL);

    char buf[256];
    ssize_t n = read(client_fd, buf, sizeof(buf));
    buf[n] = '\0';
    printf("服务器收到: %s\n", buf);

    write(client_fd, "已收到", 6);
    close(client_fd);
    close(server_fd);
    unlink(SOCKET_PATH);
}

void unix_socket_client() {
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr));

    write(sock_fd, "Hello Server!", 13);

    char buf[256];
    ssize_t n = read(sock_fd, buf, sizeof(buf));
    buf[n] = '\0';
    printf("客户端收到: %s\n", buf);

    close(sock_fd);
}
```

## 9. IPC方式对比总结

| 方式 | 性能 | 复杂度 | 跨机器 | 持久性 | 双向通信 |
|------|------|--------|--------|--------|----------|
| 匿名管道 | 高 | 低 | 否 | 否 | 半双工 |
| 命名管道 | 高 | 低 | 否 | 是 | 半双工 |
| 消息队列 | 中 | 中 | 否 | 内核持久 | 是 |
| 共享内存 | 最高 | 高 | 否 | 内核持久 | 是 |
| Unix Socket | 高 | 中 | 否 | 可选 | 全双工 |
| TCP Socket | 中 | 中 | 是 | 否 | 全双工 |

## 重点与注意事项

1. **管道的阻塞行为**：读空阻塞、写满阻塞、读端关闭写端收到SIGPIPE
2. **共享内存需同步**：共享内存本身不提供同步，需要信号量或互斥锁配合
3. **IPC资源清理**：程序异常退出可能遗留IPC资源，需用 `ipcs`/`ipcrm` 清理
4. **POSIX vs System V**：POSIX API更简洁，System V更广泛支持，新项目推荐POSIX
5. **选择建议**：简单父子通信用管道；高性能用共享内存+信号量；网络通用用socket
