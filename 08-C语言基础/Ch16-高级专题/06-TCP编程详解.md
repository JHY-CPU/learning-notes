# TCP编程详解

## 1. TCP协议特点

TCP（Transmission Control Protocol）是一种面向连接的、可靠的传输层协议。

核心特性：
- **面向连接**：通信前需建立连接（三次握手）
- **可靠传输**：有确认机制、重传机制、流量控制
- **有序传输**：数据按发送顺序到达
- **全双工**：双向同时通信
- **流式传输**：没有消息边界概念

## 2. TCP连接生命周期

```
  客户端                      服务器
    |                           |
    |-------- SYN -----------> |  (1) 客户端发起
    | <------ SYN+ACK -------- |  (2) 服务器确认
    |-------- ACK -----------> |  (3) 客户端确认
    |                           |
    | ======= 数据传输 =======  |
    |                           |
    |-------- FIN -----------> |  (4) 客户端关闭
    | <------ ACK ------------ |  (5) 服务器确认
    | <------ FIN ------------ |  (6) 服务器关闭
    |-------- ACK -----------> |  (7) 客户端确认
```

## 3. 完整的TCP服务器实现

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define BACKLOG 128
#define BUFFER_SIZE 4096

static volatile int running = 1;

void signal_handler(int sig) {
    (void)sig;
    running = 0;
}

// 设置socket为非阻塞模式
#include <fcntl.h>
int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int create_server(int port) {
    // 1. 创建socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return -1;
    }

    // 设置socket选项
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    // Linux特有：端口复用
    // setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));

    // 2. 绑定地址
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return -1;
    }

    // 3. 开始监听
    if (listen(server_fd, BACKLOG) < 0) {
        perror("listen");
        close(server_fd);
        return -1;
    }

    printf("TCP服务器启动，监听端口 %d\n", port);
    return server_fd;
}
```

## 4. 客户端处理与数据收发

```c
// 处理单个客户端连接
void handle_client(int client_fd, struct sockaddr_in *client_addr) {
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr->sin_addr, client_ip, INET_ADDRSTRLEN);
    int client_port = ntohs(client_addr->sin_port);

    printf("新连接: %s:%d (fd=%d)\n", client_ip, client_port, client_fd);

    char buffer[BUFFER_SIZE];

    while (1) {
        // 接收数据
        ssize_t received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);

        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 非阻塞模式下暂无数据
                usleep(10000);
                continue;
            }
            perror("recv");
            break;
        } else if (received == 0) {
            // 客户端关闭连接
            printf("客户端断开: %s:%d\n", client_ip, client_port);
            break;
        }

        buffer[received] = '\0';
        printf("收到来自 %s:%d [%zd字节]: %s\n",
               client_ip, client_port, received, buffer);

        // 回显数据（可靠发送：处理部分发送情况）
        ssize_t total_sent = 0;
        while (total_sent < received) {
            ssize_t sent = send(client_fd, buffer + total_sent,
                               received - total_sent, MSG_NOSIGNAL);
            if (sent < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue; // 发送缓冲区满，重试
                }
                perror("send");
                goto cleanup;
            }
            total_sent += sent;
        }
    }

cleanup:
    close(client_fd);
}
```

## 5. 多进程TCP服务器

```c
#include <sys/wait.h>

void sigchld_handler(int sig) {
    (void)sig;
    while (waitpid(-1, NULL, WNOHANG) > 0);
}

void multiprocess_server() {
    signal(SIGCHLD, sigchld_handler);
    signal(SIGINT, signal_handler);

    int server_fd = create_server(PORT);
    if (server_fd < 0) return;

    while (running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        pid_t pid = fork();
        if (pid == 0) {
            // 子进程处理客户端
            close(server_fd);
            handle_client(client_fd, &client_addr);
            exit(0);
        } else if (pid > 0) {
            // 父进程继续接受连接
            close(client_fd);
        } else {
            perror("fork");
            close(client_fd);
        }
    }

    close(server_fd);
}
```

## 6. 多线程TCP服务器

```c
#include <pthread.h>

typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
} ClientInfo;

void* client_thread(void* arg) {
    ClientInfo *info = (ClientInfo*)arg;

    // 分离线程，自动回收资源
    pthread_detach(pthread_self());

    handle_client(info->client_fd, &info->client_addr);
    free(info);

    return NULL;
}

void multithread_server() {
    signal(SIGINT, signal_handler);

    int server_fd = create_server(PORT);
    if (server_fd < 0) return;

    while (running) {
        ClientInfo *info = malloc(sizeof(ClientInfo));
        socklen_t client_len = sizeof(info->client_addr);

        info->client_fd = accept(server_fd,
            (struct sockaddr*)&info->client_addr, &client_len);

        if (info->client_fd < 0) {
            free(info);
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        pthread_t tid;
        if (pthread_create(&tid, NULL, client_thread, info) != 0) {
            perror("pthread_create");
            close(info->client_fd);
            free(info);
        }
    }

    close(server_fd);
}
```

## 7. TCP客户端完整实现

```c
int tcp_client(const char *server_ip, int port) {
    // 1. 创建socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    // 2. 设置服务器地址
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "无效地址: %s\n", server_ip);
        close(sockfd);
        return -1;
    }

    // 3. 连接服务器（带超时处理）
    int ret = connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (ret < 0) {
        perror("connect");
        close(sockfd);
        return -1;
    }

    printf("已连接到 %s:%d\n", server_ip, port);

    // 4. 交互循环
    char send_buf[BUFFER_SIZE], recv_buf[BUFFER_SIZE];

    while (fgets(send_buf, sizeof(send_buf), stdin) != NULL) {
        // 去掉换行符
        size_t len = strlen(send_buf);
        if (len > 0 && send_buf[len-1] == '\n') {
            send_buf[len-1] = '\0';
            len--;
        }

        if (strcmp(send_buf, "quit") == 0) break;

        // 发送
        if (send(sockfd, send_buf, len, 0) < 0) {
            perror("send");
            break;
        }

        // 接收
        ssize_t n = recv(sockfd, recv_buf, sizeof(recv_buf) - 1, 0);
        if (n <= 0) {
            printf("服务器断开连接\n");
            break;
        }
        recv_buf[n] = '\0';
        printf("服务器: %s\n", recv_buf);
    }

    // 5. 优雅关闭
    shutdown(sockfd, SHUT_WR); // 发送FIN，但仍可接收
    // 可以继续接收服务器最后的数据
    close(sockfd);

    return 0;
}
```

## 8. TCP粘包问题

TCP是流式协议，没有消息边界。发送方的多次write可能被合并，也可能被拆分。

```c
// 解决方案1：定长消息
#define MSG_LEN 100
void fixed_length_demo() {
    char msg[MSG_LEN];
    ssize_t total = 0;
    while (total < MSG_LEN) {
        ssize_t n = recv(sockfd, msg + total, MSG_LEN - total, 0);
        if (n <= 0) break;
        total += n;
    }
}

// 解决方案2：长度前缀（推荐）
void send_with_length(int fd, const void *data, uint32_t len) {
    uint32_t net_len = htonl(len);
    send(fd, &net_len, 4, 0); // 先发4字节长度
    ssize_t sent = 0;
    while (sent < (ssize_t)len) {
        ssize_t n = send(fd, (char*)data + sent, len - sent, 0);
        if (n <= 0) break;
        sent += n;
    }
}

ssize_t recv_with_length(int fd, void *buf, size_t buf_size) {
    uint32_t net_len;
    ssize_t n = recv(fd, &net_len, 4, MSG_WAITALL);
    if (n != 4) return -1;

    uint32_t len = ntohl(net_len);
    if (len > buf_size) return -1;

    n = recv(fd, buf, len, MSG_WAITALL);
    return n;
}

// 解决方案3：分隔符
ssize_t recv_until(int fd, char *buf, size_t size, char delim) {
    size_t pos = 0;
    while (pos < size - 1) {
        ssize_t n = recv(fd, buf + pos, 1, 0);
        if (n <= 0) return -1;
        if (buf[pos] == delim) {
            buf[pos] = '\0';
            return pos;
        }
        pos++;
    }
    return -1;
}
```

## 9. 半关闭与优雅关闭

```c
void graceful_shutdown(int sockfd) {
    // shutdown vs close:
    // close: 减少引用计数，计数为0时关闭
    // shutdown: 直接关闭连接的某个方向

    // 关闭写方向（发送FIN）
    shutdown(sockfd, SHUT_WR);

    // 继续接收剩余数据
    char buf[1024];
    ssize_t n;
    while ((n = recv(sockfd, buf, sizeof(buf), 0)) > 0) {
        printf("收到最后的数据: %zd字节\n", n);
    }

    // 最后关闭
    close(sockfd);
}
```

## 重点与注意事项

1. **SO_REUSEADDR**：服务器必须设置，否则TIME_WAIT状态下bind会失败
2. **SIGPIPE处理**：向已关闭的连接写入会收到SIGPIPE，可设置 `MSG_NOSIGNAL` 或忽略该信号
3. **部分发送/接收**：`send` 和 `recv` 可能只处理部分数据，必须循环处理
4. **粘包处理**：TCP是流式协议，需要应用层自己处理消息边界
5. **优雅关闭**：先 `shutdown(SHUT_WR)` 再 `close()`，确保数据完整送达
6. **accept被信号中断**：`accept` 可能因信号返回 `EINTR`，需重新调用
