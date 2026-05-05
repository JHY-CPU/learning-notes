# 端口与套接字

## 🔌 端口与套接字

端口作用与分类、套接字 Socket 概念、Socket 编程模型、TCP/UDP Socket 差异。

## 端口 (Port)

端口是传输层用于区分不同应用程序的逻辑标识，范围 0-65535。
```
// ========== 端口的作用 ==========
// 一台服务器运行多个网络服务:
//   ┌──────────────────────────────┐
//   │  服务器 (IP: 203.0.113.10)   │
//   │                              │
//   │  Web 服务器    :80           │
//   │  SSH 服务器    :22           │
//   │  数据库服务器  :3306         │
//   │  Redis 缓存    :6379         │
//   └──────────────────────────────┘
//
// 连接标识:
//   (客户端 IP, 客户端端口, 服务器 IP, 服务器端口, 协议)
//   例: (192.168.1.5, 52001, 203.0.113.10, 443, TCP)

// ========== 端口分配 ==========
// 知名端口 (0-1023):
//   需要 root/管理员权限绑定
//   80 (HTTP), 443 (HTTPS), 22 (SSH)
//
// 注册端口 (1024-49151):
//   可由普通用户绑定
//   3306 (MySQL), 5432 (PostgreSQL)
//   27017 (MongoDB), 6379 (Redis)
//
// 动态端口 (49152-65535):
//   客户端临时使用
//   由操作系统自动分配

// ========== 查看端口占用 ==========
// Linux/macOS:
//   $ netstat -tlnp       # TCP 监听端口
//   $ ss -tlnp            # 新版替代 netstat
//   $ lsof -i :3306       # 查看特定端口
//
// Windows:
//   > netstat -ano | findstr :3306
//   > tasklist | findstr
```
## Socket (套接字)

Socket 是应用层与传输层之间的编程接口，封装了网络通信的端点。
```
// ========== Socket 是什么 ==========
// Socket 是操作系统提供的一个抽象层
// 让开发者无需处理底层网络细节
// 通过文件描述符 (FD) 进行操作
//
// 在 Linux 中, Socket 也是"一切皆文件"的体现
// $ ls -la /proc//fd/ | grep socket

// ========== TCP Socket 编程流程 ==========
//
// 服务器端:                      客户端:
//   socket() 创建socket          socket() 创建socket
//      ↓                              ↓
//   bind() 绑定端口               connect() 连接服务器
//      ↓                              ↓
//   listen() 监听                (三次握手)
//      ↓                              ↓
//   accept() 接受连接            send()/recv()
//      ↓                              数据传输
//   recv()/send()                close()
//      ↓
//   close()
//
// ========== UDP Socket 编程流程 ==========
// 服务器端:                      客户端:
//   socket() 创建socket          socket() 创建socket
//      ↓                              ↓
//   bind() 绑定端口               sendto() 发送数据
//      ↓                              ↓
//   recvfrom() 接收              recvfrom() 接收
//      ↓                              ↓
//   sendto() 发送                close()
//      ↓
//   close()
//
// UDP 不需要 listen() 和 accept()
// UDP 不需要建立连接,直接发数据
```
## TCP vs UDP Socket 差异
```
// ========== TCP Socket ==========
// socket(AF_INET, SOCK_STREAM, 0)
//    ↑           ↑
//   IPv4       面向连接
//
// 特点:
//   - 可靠,按序,无丢失
//   - 流量控制 (滑动窗口)
//   - 拥塞控制 (慢启动/CUBIC)
//   - 数据流 (无边界,需应用层分包)
//
// 适用: HTTP, FTP, SMTP, SSH
//
// ========== UDP Socket ==========
// socket(AF_INET, SOCK_DGRAM, 0)
//    ↑           ↑
//   IPv4       面向消息
//
// 特点:
//   - 不可靠,可能丢失/乱序
//   - 无流量/拥塞控制
//   - 低延迟,低开销
//   - 消息边界保留 (每个 sendto 对应一个 recvfrom)
//
// 适用: DNS, DHCP, 视频直播, 在线游戏

// ========== 代码示例 (Node.js) ==========
// TCP 服务器:
//   const net = require('net');
//   const server = net.createServer(sock => {
//     sock.on('data', data => { /* ... */ });
//   });
//   server.listen(3000);
//
// UDP 服务器:
//   const dgram = require('dgram');
//   const server = dgram.createSocket('udp4');
//   server.on('message', (msg, rinfo) => { /* ... */ });
//   server.bind(3000);
```
## Socket 选项与常见问题
```
// ========== 重要 Socket 选项 ==========
// SO_REUSEADDR:
//   允许重用 TIME-WAIT 状态的端口
//   服务器重启后快速恢复
//
// SO_KEEPALIVE:
//   定时发送探测包检测连接存活
//   默认 2 小时无活动后开始探测
//
// TCP_NODELAY:
//   禁用 Nagle 算法
//   减少小数据包延迟
//   适合实时应用
//
// SO_RCVBUF / SO_SNDBUF:
//   设置接收/发送缓冲区大小

// ========== 常见问题 ==========
// Address already in use:
//   端口被占用,使用 SO_REUSEADDR
//
// Connection refused:
//   目标端口没有服务监听
//
// Too many open files:
//   文件描述符耗尽
//   ulimit -n 查看/修改限制

// ========== 最大连接数 ==========
// 理论 TCP 连接数:
//   客户端端口 65535 个, 但实际受限于:
//   - 文件描述符限制 (ulimit)
//   - 内存 (每个连接 ~3KB 内核缓冲区)
//   - 网络带宽
//
// 一个服务器可支持百万级并发连接
// C10K → C10M 问题是网络编程经典挑战
```
> **Note**: 💡 Linux 一切皆文件——每个 Socket 连接占用一个文件描述符。ulimit -n 默认 1024,高并发服务器需要调高到 100000+。
