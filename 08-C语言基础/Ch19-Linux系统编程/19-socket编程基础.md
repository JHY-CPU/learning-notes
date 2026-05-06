# Socket 编程基础

## 1. Socket 概述

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/*
 * int socket(int domain, int type, int protocol);
 *
 * domain (地址族):
 *   AF_INET   - IPv4
 *   AF_INET6  - IPv6
 *   AF_UNIX   - Unix 域
 *
 * type:
 *   SOCK_STREAM - 字节流（TCP）
 *   SOCK_DGRAM  - 数据报（UDP）
 *   SOCK_RAW    - 原始套接字
 *
 * protocol:
 *   0 - 自动选择
 *   IPPROTO_TCP - TCP
 *   IPPROTO_UDP - UDP
 *
 * 服务器端流程:
 *   socket() -> bind() -> listen() -> accept() -> read/write -> close()
 *
 * 客户端流程:
 *   socket() -> connect() -> read/write -> close()
 */
```

## 2. 地址结构

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/un.h>
#include <stdio.h>
#include <string.h>

/*
 * struct sockaddr {
 *     sa_family_t sa_family;
 *     char        sa_data[14];
 * };
 *
 * struct sockaddr_in {
 *     sa_family_t    sin_family;  // AF_INET
 *     in_port_t      sin_port;    // 端口号（网络字节序）
 *     struct in_addr sin_addr;    // IP 地址
 * };
 *
 * struct in_addr {
 *     uint32_t s_addr;  // IPv4 地址（网络字节序）
 * };
 */

void address_examples(void)
{
    struct sockaddr_in addr;

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);          // 主机字节序 -> 网络字节序
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // 监听所有接口

    // IP 地址转换
    // 字符串 -> 网络地址
    inet_pton(AF_INET, "192.168.1.100", &addr.sin_addr);

    // 网络地址 -> 字符串
    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));
    printf("IP: %s\n", ip_str);

    // 获取本机 IP
    addr.sin_addr.s_addr = INADDR_ANY;
    inet_ntop(AF_INET, &addr.sin_addr, ip_str, sizeof(ip_str));
    printf("Any: %s\n", ip_str);  // 0.0.0.0

    // 字节序转换
    uint16_t port = 8080;
    printf("Host port: %u, Network port: %u\n", port, ntohs(htons(port)));
    printf("Host 0x1234: %04x, Network: %04x\n",
           0x1234, ntohs(htons(0x1234)));
}
```

## 3. TCP 服务器

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#define PORT 8080
#define BACKLOG 10
#define BUF_SIZE 1024

/*
 * listen(int sockfd, int backlog);
 *   backlog: 连接队列最大长度
 *
 * accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
 *   返回新的连接 socket，原 socket 继续监听
 */

// 基本 TCP 服务器
int tcp_server(void)
{
    int listen_fd, conn_fd;
    struct sockaddr_in addr, client_addr;
    socklen_t client_len;
    char buf[BUF_SIZE];

    // 创建监听 socket
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == -1) { perror("socket"); return -1; }

    // 允许地址重用
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 绑定地址
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("bind");
        close(listen_fd);
        return -1;
    }

    // 开始监听
    if (listen(listen_fd, BACKLOG) == -1) {
        perror("listen");
        close(listen_fd);
        return -1;
    }

    printf("Server listening on port %d...\n", PORT);

    while (1) {
        client_len = sizeof(client_addr);

        // 接受连接（阻塞）
        conn_fd = accept(listen_fd, (struct sockaddr *)&client_addr,
                         &client_len);
        if (conn_fd == -1) {
            perror("accept");
            continue;
        }

        printf("Client connected: %s:%d\n",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port));

        // 处理客户端
        ssize_t n;
        while ((n = read(conn_fd, buf, sizeof(buf) - 1)) > 0) {
            buf[n] = '\0';
            printf("Received: %s", buf);

            // Echo 回去
            write(conn_fd, buf, n);
        }

        printf("Client disconnected\n");
        close(conn_fd);
    }

    close(listen_fd);
    return 0;
}
```

## 4. TCP 客户端

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int tcp_client(const char *server_ip, int port)
{
    int sock_fd;
    struct sockaddr_in server_addr;
    char buf[1024];

    // 创建 socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd == -1) { perror("socket"); return -1; }

    // 服务器地址
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);

    // 连接服务器
    if (connect(sock_fd, (struct sockaddr *)&server_addr,
                sizeof(server_addr)) == -1) {
        perror("connect");
        close(sock_fd);
        return -1;
    }

    printf("Connected to %s:%d\n", server_ip, port);

    // 发送数据
    const char *msg = "Hello from client!\n";
    write(sock_fd, msg, strlen(msg));

    // 接收回复
    ssize_t n = read(sock_fd, buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        printf("Server says: %s", buf);
    }

    close(sock_fd);
    return 0;
}
```

## 5. UDP 服务器与客户端

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// UDP 服务器
int udp_server(int port)
{
    int sock_fd;
    struct sockaddr_in addr, client_addr;
    socklen_t client_len;
    char buf[1024];

    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd == -1) { perror("socket"); return -1; }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
        perror("bind");
        close(sock_fd);
        return -1;
    }

    printf("UDP server listening on port %d\n", port);

    while (1) {
        client_len = sizeof(client_addr);

        ssize_t n = recvfrom(sock_fd, buf, sizeof(buf) - 1, 0,
                             (struct sockaddr *)&client_addr, &client_len);
        if (n < 0) { perror("recvfrom"); continue; }

        buf[n] = '\0';
        printf("From %s:%d: %s",
               inet_ntoa(client_addr.sin_addr),
               ntohs(client_addr.sin_port), buf);

        // 回复客户端
        sendto(sock_fd, buf, n, 0,
               (struct sockaddr *)&client_addr, client_len);
    }

    close(sock_fd);
    return 0;
}

// UDP 客户端
int udp_client(const char *server_ip, int port)
{
    int sock_fd;
    struct sockaddr_in server_addr;
    char buf[1024];

    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd == -1) { perror("socket"); return -1; }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);

    const char *msg = "Hello UDP!\n";

    // 发送数据
    sendto(sock_fd, msg, strlen(msg), 0,
           (struct sockaddr *)&server_addr, sizeof(server_addr));

    // 接收回复
    ssize_t n = recvfrom(sock_fd, buf, sizeof(buf) - 1, 0, NULL, NULL);
    if (n > 0) {
        buf[n] = '\0';
        printf("Server reply: %s", buf);
    }

    close(sock_fd);
    return 0;
}
```

## 6. getaddrinfo - 地址解析

```c
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/*
 * int getaddrinfo(const char *node, const char *service,
 *                 const struct addrinfo *hints,
 *                 struct addrinfo **res);
 * void freeaddrinfo(struct addrinfo *res);
 *
 * 替代 gethostbyname 和 getservbyname
 * 支持 IPv4/IPv6，线程安全
 *
 * struct addrinfo {
 *     int              ai_flags;
 *     int              ai_family;    - AF_INET, AF_INET6, AF_UNSPEC
 *     int              ai_socktype;  - SOCK_STREAM, SOCK_DGRAM
 *     int              ai_protocol;
 *     size_t           ai_addrlen;
 *     struct sockaddr *ai_addr;
 *     char            *ai_canonname;
 *     struct addrinfo *ai_next;      - 链表下一个
 * };
 */

// 客户端使用 getaddrinfo 连接
int connect_to_host(const char *host, const char *service)
{
    struct addrinfo hints, *res, *rp;
    int sock_fd;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;      // IPv4 或 IPv6
    hints.ai_socktype = SOCK_STREAM;  // TCP

    int ret = getaddrinfo(host, service, &hints, &res);
    if (ret != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
        return -1;
    }

    // 遍历所有地址，尝试连接
    for (rp = res; rp != NULL; rp = rp->ai_next) {
        sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sock_fd == -1) continue;

        if (connect(sock_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;  // 连接成功
        }

        close(sock_fd);
    }

    freeaddrinfo(res);

    if (rp == NULL) {
        fprintf(stderr, "Could not connect\n");
        return -1;
    }

    return sock_fd;
}

// 服务器使用 getaddrinfo 监听
int bind_server(const char *service)
{
    struct addrinfo hints, *res, *rp;
    int listen_fd;

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;  // 用于 bind

    int ret = getaddrinfo(NULL, service, &hints, &res);
    if (ret != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(ret));
        return -1;
    }

    for (rp = res; rp != NULL; rp = rp->ai_next) {
        listen_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (listen_fd == -1) continue;

        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        if (bind(listen_fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            break;
        }

        close(listen_fd);
    }

    freeaddrinfo(res);

    if (rp == NULL) {
        fprintf(stderr, "Could not bind\n");
        return -1;
    }

    listen(listen_fd, 10);
    return listen_fd;
}
```
