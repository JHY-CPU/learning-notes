# Socket基础

## 🔌 Socket 基础

Socket 类型、地址结构、字节序、Socket API 详解、阻塞 vs 非阻塞。

## Socket 类型与地址族
```
// ========== Socket 地址族 ==========
// AF_INET (AF = Address Family):
//   IPv4 地址族
//   struct sockaddr_in
//
// AF_INET6:
//   IPv6 地址族
//   struct sockaddr_in6
//
// AF_UNIX (本地通信):
//   同一台机器的进程间通信
//   使用文件路径作为地址
//   比 TCP localhost 更高效
//
// ========== Socket 类型 ==========
// SOCK_STREAM:
//   面向连接的可靠字节流
//   对应 TCP
//
// SOCK_DGRAM:
//   无连接的数据报
//   对应 UDP
//
// SOCK_RAW:
//   原始套接字,直接操作 IP 层
//   需要 root 权限
//   用于 ping, traceroute, 抓包等
```
## 字节序 (Byte Order)

网络字节序统一为大端序 (Big-Endian)，不同架构的机器间通信需要转换。
```
// ========== 字节序概念 ==========
// 大端序 (Big-Endian / 网络字节序):
//   高位字节存在低地址
//   例: 0x1234 → [0x12] [0x34]
//
// 小端序 (Little-Endian / 主机字节序):
//   低位字节存在低地址
//   例: 0x1234 → [0x34] [0x12]
//
// x86/x64 架构 = 小端序
// 网络协议 = 大端序
//
// ========== 字节序转换函数 ==========
// htons():  host to network short (16位)
// htonl():  host to network long (32位)
// ntohs():  network to host short
// ntohl():  network to host long
//
// C 语言示例:
//   struct sockaddr_in addr;
//   addr.sin_family = AF_INET;
//   addr.sin_port = htons(8080);  // 端口转网络字节序
//   addr.sin_addr.s_addr = htonl(INADDR_ANY);
//
// 高级语言通常封装了字节序处理
// Node.js: 无需手动处理
// Python: socket.htons(port)

// ========== 地址结构 ==========
// struct sockaddr_in {
//     sa_family_t    sin_family;  // AF_INET
//     in_port_t      sin_port;    // 端口 (网络字节序)
//     struct in_addr sin_addr;    // IP 地址
//     char           sin_zero[8]; // 填充
// };
```
## Socket API 详解
```
// ========== 核心 API ==========
//
// socket(domain, type, protocol):
//   创建套接字,返回文件描述符
//   例: socket(AF_INET, SOCK_STREAM, 0)
//
// bind(sockfd, addr, addrlen):
//   绑定地址和端口
//   服务器端必须调用
//   例: bind(sock, (struct sockaddr*)&addr, sizeof(addr))
//
// listen(sockfd, backlog):
//   TCP 服务器监听连接
//   backlog: 等待队列最大长度
//   例: listen(sock, 128)
//
// accept(sockfd, addr, addrlen):
//   接受客户端连接
//   返回新 socket 用于通信
//
// connect(sockfd, addr, addrlen):
//   客户端连接服务器
//
// recv(sockfd, buf, len, flags):
//   接收数据
//
// send(sockfd, buf, len, flags):
//   发送数据
//
// close(sockfd):
//   关闭套接字
```
## 阻塞 vs 非阻塞
```
// ========== 阻塞 I/O ==========
// 默认模式,函数调用会阻塞直到完成
//
// 阻塞场景:
//   accept():    等待客户端连接
//   recv():      等待数据到达
//   connect():   等待握手完成
//   send():      等待缓冲区有空闲
//
// 问题: 一个阻塞调用会卡住整个线程
// 解决方案: 多线程 / 非阻塞 / 异步

// ========== 非阻塞 I/O ==========
// 调用立即返回,无论操作是否完成
// 设置方式:
//   fcntl(sockfd, F_SETFL, O_NONBLOCK)
// 或:
//   socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0)
//
// 返回 -1, errno = EAGAIN/EWOULDBLOCK
// 表示"现在不行,稍后再试"

// ========== I/O 多路复用 ==========
// select(), poll(), epoll() (Linux)
// kqueue (macOS/BSD)
// IOCP (Windows)
//
// 事件驱动,一个线程管理数千连接
// Node.js 的 libuv 就是基于 epoll/kqueue/IOCP

// ========== I/O 模型对比 ==========
// 阻塞 I/O:         简单,但一个连接一个线程
// 非阻塞 I/O:       轮询,浪费 CPU
// I/O 多路复用:     epoll,高并发
// 信号驱动 I/O:     SIGIO,不常用
// 异步 I/O (AIO):   内核完成操作再通知
```
> **Note**: ⚡ epoll 是 Linux 高性能网络编程的核心——Node.js、Nginx、Redis 都依赖它实现单线程处理万级并发连接。
