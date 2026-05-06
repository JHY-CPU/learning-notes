# Socket 编程进阶

## 1. 非阻塞 Socket

```c
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>

/*
 * 设置 socket 非阻塞的方式:
 * 1. fcntl(fd, F_SETFL, O_NONBLOCK)
 * 2. socket 时使用 SOCK_NONBLOCK 标志
 */

int set_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void nonblocking_connect(const char *ip, int port)
{
    int sock = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    // 或：set_nonblocking(sock);

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);

    int ret = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
    if (ret == -1) {
        if (errno == EINPROGRESS) {
            // 连接进行中，使用 poll/select 等待
            struct pollfd pfd = {sock, POLLOUT, 0};
            if (poll(&pfd, 1, 5000) > 0) {
                int err;
                socklen_t len = sizeof(err);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, &err, &len);
                if (err == 0) {
                    printf("Connected!\n");
                }
            }
        }
    }
    close(sock);
}
```

## 2. Socket 超时设置

```c
#include <sys/socket.h>
#include <sys/select.h>
#include <poll.h>
#include <stdio.h>
#include <errno.h>

// 方式1：SO_RCVTIMEO / SO_SNDTIMEO
void socket_timeout_options(int sock)
{
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;

    // 接收超时
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // 发送超时
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // 如果超时，read/write 返回 -1，errno = EAGAIN
}

// 方式2：select 带超时
int read_with_timeout(int sock, char *buf, int len, int timeout_sec)
{
    fd_set readfds;
    struct timeval tv = {timeout_sec, 0};

    FD_ZERO(&readfds);
    FD_SET(sock, &readfds);

    int ret = select(sock + 1, &readfds, NULL, NULL, &tv);
    if (ret < 0) return -1;
    if (ret == 0) { errno = ETIMEDOUT; return -1; }

    return read(sock, buf, len);
}

// 方式3：poll 带超时
int read_with_timeout_poll(int sock, char *buf, int len, int timeout_ms)
{
    struct pollfd pfd;
    pfd.fd = sock;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, timeout_ms);
    if (ret < 0) return -1;
    if (ret == 0) { errno = ETIMEDOUT; return -1; }

    if (pfd.revents & POLLIN) {
        return read(sock, buf, len);
    }
    return -1;
}
```

## 3. 广播与多播

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/*
 * 广播 (Broadcast):
 *   向同一网段的所有主机发送数据
 *   地址：255.255.255.255（受限广播）
 *         x.x.x.255（定向广播）
 *   仅 UDP 支持
 *
 * 多播 (Multicast):
 *   向一组主机发送数据
 *   地址：224.0.0.0 - 239.255.255.255
 */

// 广播发送端
void broadcast_sender(int port)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    int broadcast = 1;
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast,
               sizeof(broadcast));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("255.255.255.255");

    const char *msg = "Broadcast message!\n";
    sendto(sock, msg, strlen(msg), 0,
           (struct sockaddr *)&addr, sizeof(addr));
    printf("Broadcast sent\n");

    close(sock);
}

// 广播接收端
void broadcast_receiver(int port)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sock, (struct sockaddr *)&addr, sizeof(addr));

    char buf[1024];
    struct sockaddr_in sender;
    socklen_t len = sizeof(sender);

    printf("Waiting for broadcasts...\n");
    ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                         (struct sockaddr *)&sender, &len);
    if (n > 0) {
        buf[n] = '\0';
        printf("Broadcast from %s: %s",
               inet_ntoa(sender.sin_addr), buf);
    }

    close(sock);
}

// 多播发送端
void multicast_sender(const char *group, int port)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, group, &addr.sin_addr);

    const char *msg = "Multicast message!\n";
    sendto(sock, msg, strlen(msg), 0,
           (struct sockaddr *)&addr, sizeof(addr));

    close(sock);
}

// 多播接收端
void multicast_receiver(const char *group, int port)
{
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;
    bind(sock, (struct sockaddr *)&addr, sizeof(addr));

    // 加入多播组
    struct ip_mreq mreq;
    inet_pton(AF_INET, group, &mreq.imr_multiaddr);
    mreq.imr_interface.s_addr = INADDR_ANY;
    setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));

    char buf[1024];
    recv(sock, buf, sizeof(buf), 0);
    printf("Multicast: %s\n", buf);

    close(sock);
}
```

## 4. Unix 域 Socket

```c
#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/*
 * Unix 域 Socket:
 *   - 同一主机上的进程间通信
 *   - 比网络 socket 更高效
 *   - 支持 SOCK_STREAM 和 SOCK_DGRAM
 *   - 可以传递文件描述符和凭证
 */

// Unix 域流式 socket 服务器
void unix_stream_server(const char *path)
{
    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    unlink(path);  // 先删除旧的 socket 文件

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("bind");
        close(listen_fd);
        return;
    }

    listen(listen_fd, 5);
    printf("Unix server listening on %s\n", path);

    int conn_fd = accept(listen_fd, NULL, NULL);
    char buf[256];
    ssize_t n = read(conn_fd, buf, sizeof(buf));
    if (n > 0) {
        buf[n] = '\0';
        printf("Received: %s\n", buf);
        write(conn_fd, "Hello from unix server\n", 23);
    }

    close(conn_fd);
    close(listen_fd);
    unlink(path);
}

// Unix 域流式 socket 客户端
void unix_stream_client(const char *path)
{
    int sock = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("connect");
        close(sock);
        return;
    }

    write(sock, "Hello from unix client\n", 23);

    char buf[256];
    ssize_t n = read(sock, buf, sizeof(buf));
    if (n > 0) {
        buf[n] = '\0';
        printf("Reply: %s\n", buf);
    }

    close(sock);
}

// Unix 域数据报 socket
void unix_dgram_example(void)
{
    const char *server_path = "/tmp/unix_dgram_server";
    const char *client_path = "/tmp/unix_dgram_client";

    int sock = socket(AF_UNIX, SOCK_DGRAM, 0);

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, server_path, sizeof(addr.sun_path) - 1);

    unlink(server_path);
    bind(sock, (struct sockaddr *)&addr, sizeof(addr));

    // 客户端路径
    struct sockaddr_un client_addr;
    memset(&client_addr, 0, sizeof(client_addr));
    client_addr.sun_family = AF_UNIX;
    strncpy(client_addr.sun_path, client_path,
            sizeof(client_addr.sun_path) - 1);

    char buf[256];
    struct sockaddr_un from;
    socklen_t from_len = sizeof(from);

    ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                         (struct sockaddr *)&from, &from_len);
    printf("From %s: %.*s\n", from.sun_path, (int)n, buf);

    close(sock);
    unlink(server_path);
}
```
