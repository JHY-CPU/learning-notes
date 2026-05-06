# UDP编程详解

## 1. UDP协议特点

UDP（User Datagram Protocol）是一种无连接的传输层协议。

核心特性：
- **无连接**：不需要建立/断开连接，直接发送
- **不可靠**：不保证送达，不保证顺序，没有重传机制
- **数据报传输**：有消息边界，一个send对应一个recv
- **高效**：头部开销小（8字节 vs TCP的20字节）
- **支持广播/多播**：TCP不支持

适用场景：实时音视频、DNS查询、DHCP、游戏、IoT传感器数据

## 2. UDP基本编程模型

```
    UDP客户端                     UDP服务器
      |                             |
      |  socket()                   |  socket()
      |                             |  bind()
      |  sendto() ---- 数据 ------> |  recvfrom()
      |  recvfrom() <--- 数据 ----- |  sendto()
      |                             |
      |  close()                    |  close()
```

## 3. UDP服务器实现

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define UDP_PORT 9090
#define BUFFER_SIZE 65536 // UDP最大数据报大小

int udp_server(int port) {
    // 1. 创建UDP socket
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return -1;
    }

    // 2. 绑定地址
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(sockfd);
        return -1;
    }

    printf("UDP服务器监听端口 %d...\n", port);

    // 3. 接收和发送数据
    char buffer[BUFFER_SIZE];
    struct sockaddr_in client_addr;
    socklen_t client_len;

    while (1) {
        client_len = sizeof(client_addr);

        // recvfrom阻塞等待数据，同时获取发送方地址
        ssize_t n = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                             (struct sockaddr*)&client_addr, &client_len);

        if (n < 0) {
            perror("recvfrom");
            continue;
        }

        buffer[n] = '\0';
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);

        printf("收到来自 %s:%d 的 [%zd字节]: %s\n",
               client_ip, ntohs(client_addr.sin_port), n, buffer);

        // 回显响应
        sendto(sockfd, buffer, n, 0,
               (struct sockaddr*)&client_addr, client_len);
    }

    close(sockfd);
    return 0;
}
```

## 4. UDP客户端实现

```c
int udp_client(const char *server_ip, int port) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);

    char send_buf[BUFFER_SIZE], recv_buf[BUFFER_SIZE];

    printf("UDP客户端已启动 (输入 'quit' 退出)\n");

    while (fgets(send_buf, sizeof(send_buf), stdin) != NULL) {
        size_t len = strlen(send_buf);
        if (len > 0 && send_buf[len - 1] == '\n') {
            send_buf[len - 1] = '\0';
            len--;
        }

        if (strcmp(send_buf, "quit") == 0) break;

        // 发送数据报
        ssize_t sent = sendto(sockfd, send_buf, len, 0,
                              (struct sockaddr*)&server_addr, sizeof(server_addr));
        if (sent < 0) {
            perror("sendto");
            continue;
        }

        // 设置接收超时
        struct timeval tv = {5, 0}; // 5秒
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // 接收响应
        ssize_t n = recvfrom(sockfd, recv_buf, sizeof(recv_buf) - 1, 0, NULL, NULL);
        if (n < 0) {
            perror("recvfrom (可能超时)");
            continue;
        }

        recv_buf[n] = '\0';
        printf("服务器响应: %s\n", recv_buf);
    }

    close(sockfd);
    return 0;
}
```

## 5. UDP connect 机制

UDP虽然无连接，但可以使用 `connect` 绑定固定的目标地址。

```c
void udp_connect_demo() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(9090);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);

    // UDP connect的作用：
    // 1. 记录目标地址，之后可以用send/recv替代sendto/recvfrom
    // 2. 内核会过滤非该地址发来的数据
    // 3. 可以检测某些错误（如端口不可达）
    connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));

    // 之后可以使用send/recv
    send(sockfd, "Hello", 5, 0);

    char buf[1024];
    recv(sockfd, buf, sizeof(buf), 0);

    // 可以重新connect到不同地址
    server_addr.sin_port = htons(9091);
    connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));

    close(sockfd);
}
```

## 6. 广播（Broadcast）

UDP广播向同一子网内的所有主机发送数据。

```c
void udp_broadcast_sender() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    // 启用广播
    int broadcast = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    struct sockaddr_in broadcast_addr;
    memset(&broadcast_addr, 0, sizeof(broadcast_addr));
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(9090);
    broadcast_addr.sin_addr.s_addr = inet_addr("255.255.255.255"); // 广播地址

    // 也可以使用子网广播地址，如 192.168.1.255

    const char *msg = "Broadcast message!";
    sendto(sockfd, msg, strlen(msg), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));

    printf("广播消息已发送\n");
    close(sockfd);
}

void udp_broadcast_receiver() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9090);
    addr.sin_addr.s_addr = INADDR_ANY; // 接收任意来源的广播

    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    printf("等待广播...\n");

    char buf[1024];
    struct sockaddr_in sender;
    socklen_t sender_len = sizeof(sender);

    ssize_t n = recvfrom(sockfd, buf, sizeof(buf) - 1, 0,
                         (struct sockaddr*)&sender, &sender_len);
    if (n > 0) {
        buf[n] = '\0';
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &sender.sin_addr, ip, INET_ADDRSTRLEN);
        printf("收到广播来自 %s: %s\n", ip, buf);
    }

    close(sockfd);
}
```

## 7. 多播（Multicast）

多播只向加入特定组的主机发送数据，比广播更高效。

```c
#include <netinet/in.h>

void multicast_sender() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    // 设置TTL（多播数据包跳数限制）
    int ttl = 5;
    setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));

    struct sockaddr_in group_addr;
    memset(&group_addr, 0, sizeof(group_addr));
    group_addr.sin_family = AF_INET;
    group_addr.sin_port = htons(9090);
    // 多播地址范围: 224.0.0.0 ~ 239.255.255.255
    group_addr.sin_addr.s_addr = inet_addr("239.1.1.1");

    const char *msg = "Multicast message!";
    sendto(sockfd, msg, strlen(msg), 0,
           (struct sockaddr*)&group_addr, sizeof(group_addr));

    printf("多播消息已发送\n");
    close(sockfd);
}

void multicast_receiver() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);

    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(9090);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));

    // 加入多播组
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr("239.1.1.1");
    mreq.imr_interface.s_addr = INADDR_ANY; // 使用默认接口
    setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));

    printf("已加入多播组 239.1.1.1，等待消息...\n");

    char buf[1024];
    ssize_t n = recvfrom(sockfd, buf, sizeof(buf) - 1, 0, NULL, NULL);
    if (n > 0) {
        buf[n] = '\0';
        printf("收到多播: %s\n", buf);
    }

    // 离开多播组
    setsockopt(sockfd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &mreq, sizeof(mreq));
    close(sockfd);
}
```

## 8. UDP可靠性增强

UDP本身不可靠，但可以在应用层添加可靠性机制。

```c
#include <stdint.h>
#include <time.h>

// 简单的可靠UDP协议头
typedef struct {
    uint32_t seq;      // 序列号
    uint32_t ack;      // 确认号
    uint16_t flags;    // 标志位
    uint16_t len;      // 数据长度
} ReliableUDPHeader;

#define FLAG_SYN  0x01
#define FLAG_ACK  0x02
#define FLAG_DATA 0x04
#define FLAG_FIN  0x08

#define TIMEOUT_MS 1000
#define MAX_RETRIES 5

// 带超时和重传的发送
ssize_t reliable_sendto(int fd, const void *data, size_t len,
                        struct sockaddr *addr, socklen_t addr_len) {
    ReliableUDPHeader header;
    header.seq = (uint32_t)time(NULL); // 简化：用时间戳作序号
    header.flags = FLAG_DATA;
    header.len = htons((uint16_t)len);

    // 组装数据包
    char packet[sizeof(header) + len];
    memcpy(packet, &header, sizeof(header));
    memcpy(packet + sizeof(header), data, len);

    // 设置接收超时
    struct timeval tv = {0, TIMEOUT_MS * 1000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    for (int retry = 0; retry < MAX_RETRIES; retry++) {
        sendto(fd, packet, sizeof(packet), 0, addr, addr_len);

        // 等待ACK
        ReliableUDPHeader ack_header;
        ssize_t n = recvfrom(fd, &ack_header, sizeof(ack_header), 0, NULL, NULL);

        if (n >= (ssize_t)sizeof(ack_header) &&
            (ack_header.flags & FLAG_ACK) &&
            ack_header.ack == header.seq) {
            return len; // 成功
        }

        printf("重传 %d/%d\n", retry + 1, MAX_RETRIES);
    }

    return -1; // 超过最大重试次数
}
```

## 9. TCP vs UDP 对比

| 特性 | TCP | UDP |
|------|-----|-----|
| 连接 | 面连接 | 无连接 |
| 可靠性 | 可靠（确认/重传） | 不可靠 |
| 有序性 | 保证顺序 | 不保证 |
| 边界 | 流式（无边界） | 数据报（有边界） |
| 速度 | 较慢（控制开销） | 快（最小开销） |
| 广播/多播 | 不支持 | 支持 |
| 头部大小 | 20字节+ | 8字节 |
| 典型应用 | HTTP、FTP、SSH | DNS、DHCP、视频流 |

## 重点与注意事项

1. **数据报大小**：UDP数据报最大65535字节（含头部），实际建议不超过MTU（通常~1400字节）以避免分片
2. **无流量控制**：发送方不考虑接收方能力，可能导致丢包
3. **广播需要SO_BROADCAST**：默认不允许发送广播
4. **多播TTL设置**：控制多播范围，0=本机，1=本地网络，>1可跨路由器
5. **UDP也可以connect**：`connect` 后可使用 `send/recv`，且内核会过滤非目标地址的数据
6. **应用层可靠性**：如果需要可靠传输，需要在应用层实现确认和重传机制
