# HTTP/2 与 HTTP/3


## HTTP/2 与 HTTP/3


HTTP/2HTTP/3QUIC


HTTP/2 通过二进制分帧和多路复用解决了 HTTP/1.1 的队头阻塞问题。HTTP/3 基于 QUIC 协议进一步消除了 TCP 层面的队头阻塞。


## HTTP/2 二进制分帧层


```
HTTP/1.1 vs HTTP/2：
  HTTP/1.1：文本协议，基于字符解析
  HTTP/2：二进制协议，基于帧（Frame）解析

帧结构（Frame）：
  ┌─────────────────────────────────────────────┐
  │  +-------------------------------+           │
  │  | Length (24 bits) 帧数据长度   |           │
  │  +-------------------------------+           │
  │  | Type (8 bits) 帧类型         |           │
  │  +-------------------------------+           │
  │  | Flags (8 bits) 标志位        |           │
  │  +-------------------------------+           │
  │  | R (1 bit) 保留位             |           │
  │  | Stream ID (31 bits) 流标识    |           │
  │  +-------------------------------+           │
  │  | Frame Payload (可变长度)      |           │
  │  +-------------------------------+           │
  └─────────────────────────────────────────────┘

帧类型：
  DATA       - 传输请求/响应体
  HEADERS    - 传输头部
  PRIORITY   - 设置流优先级
  RST_STREAM - 终止流
  SETTINGS   - 连接配置
  PUSH_PROMISE - 服务器推送预告
  PING       - 连接保活
  GOAWAY     - 通知关闭连接
  WINDOW_UPDATE - 流量控制
  CONTINUATION - 头部续传

连接升级过程：
  // 客户端发送升级请求（支持HTTP/2）
  GET / HTTP/1.1
  Host: www.example.com
  Connection: Upgrade, HTTP2-Settings
  Upgrade: h2c           // h2c = HTTP/2 cleartext
  HTTP2-Settings: ...

  // ALPN 协商（HTTPS更常用）
  TLS握手中的 Application-Layer Protocol Negotiation
  客户端支持：h2, http/1.1
  服务端选择：h2
  // 之后的通信全部使用 HTTP/2

  // 实际上大多数 HTTP/2 都基于 TLS（h2）
  // 纯文本 HTTP/2（h2c）使用较少
```


## 多路复用（Multiplexing）


```
多路复用原理：
  在一个TCP连接上同时发送多个请求和响应
  每个请求/响应是一个独立的流（Stream）
  通过 Stream ID 区分不同的流
  帧可以乱序发送，在接收端按 Stream ID 重组

HTTP/1.1 并发模型：
  浏览器 → 6个TCP连接 → 每个连接串行请求
  连接数有限，队头阻塞

HTTP/2 多路复用：
  浏览器 → 1个TCP连接 → 所有请求并行传输
  ┌─────────────────────────────────────────────┐
  │  TCP Connection                              │
  │  ┌─────────┬─────────┬─────────┬─────────┐  │
  │  │Stream 1 │Stream 3 │Stream 5 │Stream 7 │  │
  │  │HEADERS  │HEADERS  │DATA     │HEADERS  │  │
  │  ├─────────┼─────────┼─────────┼─────────┤  │
  │  │DATA     │DATA     │DATA     │DATA     │  │
  │  ├─────────┼─────────┼─────────┼─────────┤  │
  │  │DATA     │HEADERS  │HEADERS  │DATA     │  │
  │  └─────────┴─────────┴─────────┴─────────┘  │
  │  帧可以交错发送，互不阻塞                    │
  └─────────────────────────────────────────────┘

流优先级：
  每个流可以设置优先级和依赖关系
  客户端告诉服务端哪些资源更重要
  服务端可以按优先级分配带宽

  // 优先级树
  Stream 1 (HTML) — 最高优先级
    ├── Stream 3 (CSS) — 依赖于1
    └── Stream 5 (JS)  — 依赖于3
        └── Stream 7 (Image) — 依赖于5

流量控制（Flow Control）：
  每个流和连接都有流量控制窗口
  接收方通过 WINDOW_UPDATE 帧通知发送方
  防止发送方淹没接收方

  // 连接级流量控制：整个TCP连接的带宽
  // 流级流量控制：单个流的带宽
  // 接收方缓冲区满了就不发 WINDOW_UPDATE
  // 发送方等待窗口恢复后继续发送
```


## 服务器推送（Server Push）与 HPACK


```
服务器推送：
  服务端可以主动推送资源给客户端
  不需要客户端先请求

推送流程：
  1. 客户端请求 index.html
  2. 服务端知道客户端还需要 style.css
  3. 服务端发送 PUSH_PROMISE 帧预告
  4. 服务端在新的 Stream 中推送 style.css
  5. 客户端收到后存入缓存

帧序列：
  Server → Client:
    Stream 1: HEADERS (index.html 响应头)
    Stream 2: PUSH_PROMISE (预告要推 style.css)
    Stream 1: DATA (index.html 内容)
    Stream 2: HEADERS (style.css 响应头)
    Stream 2: DATA (style.css 内容)

  // 客户端可以用 RST_STREAM 拒绝推送
  // 如果客户端已有缓存，会拒绝推送

实际使用限制：
  - 浏览器支持不稳定（Chrome 已弃用 HTTP/2 Push）
  - 难以判断客户端是否需要
  - 可能推送客户端已有的资源
  - 实际项目中用 <link rel="preload"> 替代

HPACK 头部压缩：
  HTTP/1.1 每次请求都携带完整头部
  Cookie、User-Agent 等头部很大且重复

HPACK 原理：
  ┌─────────────────────────────────────────────┐
  │  静态表（61个预定义的头部字段）             │
  │  :authority: www.example.com  → 索引2       │
  │  :method: GET                 → 索引3       │
  │  :path: /                     → 索引4       │
  │                                             │
  │  动态表（连接内维护，LRU淘汰）              │
  │  custom-header: value → 本次连接内复用      │
  │                                             │
  │  编码方式：                                  │
  │  - 索引引用：直接用索引号（1字节）          │
  │  - 字面量：新头部用 Huffman 编码压缩        │
  │  - 从未索引：敏感头部不入动态表             │
  └─────────────────────────────────────────────┘

  // 第一次请求
  :method: GET          → 索引引用 83（1字节）
  :path: /index.html    → 字面量 + Huffman
  :authority: example.com → 索引引用 1（1字节）

  // 第二次请求（同一个连接）
  大部分头部在动态表中 → 直接索引引用
  头部大小从 ~1KB 压缩到 ~100B
```


## HTTP/3 与 QUIC 协议


```
HTTP/2 的遗留问题 — TCP 层队头阻塞：
  HTTP/2 解决了 HTTP 层队头阻塞
  但仍然使用 TCP → TCP 层队头阻塞仍然存在
  ┌─────────────────────────────────────────────┐
  │  一个TCP包丢失 → 整个TCP连接阻塞            │
  │  即使其他流的数据已到达，也要等待重传        │
  │                                             │
  │  Stream 1: [包1] [包2丢失] [包3]            │
  │  Stream 3: [包4] [包5] [包6]                │
  │  Stream 5: [包7] [包8] [包9]                │
  │                                             │
  │  包2丢失 → 所有流都要等待TCP重传包2         │
  └─────────────────────────────────────────────┘

HTTP/3 基于 QUIC：
  QUIC = Quick UDP Internet Connections
  基于 UDP 而非 TCP
  在用户空间实现可靠传输
  消除 TCP 层队头阻塞

QUIC 协议栈：
  ┌─────────────────────────────────────────────┐
  │  HTTP/2                HTTP/3               │
  │  ├── HPACK             ├── QPACK            │
  │  ├── HTTP/2 Framing    ├── HTTP/3 Framing   │
  │  ├── TLS 1.2+          ├── TLS 1.3 内置    │
  │  └── TCP               └── QUIC (UDP)       │
  └─────────────────────────────────────────────┘

QUIC 核心特性：
  1. 消除队头阻塞
     QUIC 流之间独立，一个流丢包不影响其他流
     Stream 1 丢包 → 只阻塞 Stream 1
     Stream 3、5 继续正常传输

  2. 连接建立更快
     TCP + TLS：需要 2-3 个 RTT
     QUIC：首次 1-2 个 RTT（握手+密钥协商合并）
     0-RTT：恢复连接时 0 个 RTT

  3. 连接迁移
     TCP 连接基于四元组（IP+端口）
     IP 变化（WiFi→4G）→ TCP 连接断开
     QUIC 用 Connection ID 标识连接
     IP 变化 → 连接不中断

  4. 前向纠错（FEC）
     发送冗余数据，少量丢包可恢复
     减少重传次数

0-RTT 连接恢复：
  首次连接：
    Client → Server: Initial (ClientHello)
    Server → Client: Handshake (ServerHello + Certificate)
    Client → Server: Finished
    // 1-RTT

  恢复连接（有之前的会话票据）：
    Client → Server: Initial + 0-RTT 数据
    // 0-RTT！第一个包就携带应用数据

  0-RTT 的安全风险：
    重放攻击：攻击者可以重放 0-RTT 数据
    仅适用于幂等请求（GET）
    非幂等请求（POST）不能使用 0-RTT

QPACK 头部压缩：
  HPACK 依赖流内有序传输
  QUIC 流之间乱序 → 需要 QPACK
  QPACK 使用两个单向流：
    - Encoder Stream：同步动态表更新
    - Decoder Stream：发送确认
  解决了 HPACK 在乱序环境下的问题
```


## HTTP 版本对比总结


```
┌──────────┬────────────┬────────────┬──────────────────┐
│ 特性     │ HTTP/1.1   │ HTTP/2     │ HTTP/3           │
├──────────┼────────────┼────────────┼──────────────────┤
│ 协议     │ 文本       │ 二进制     │ 二进制           │
│ 传输层   │ TCP        │ TCP        │ QUIC (UDP)       │
│ 连接数   │ 多个       │ 单个       │ 单个             │
│ 多路复用 │ 不支持     │ 支持       │ 支持（无TCP阻塞）│
│ 头部压缩 │ 无         │ HPACK      │ QPACK            │
│ 服务器推 │ 不支持     │ 支持       │ 支持             │
│ 队头阻塞 │ HTTP+TCP   │ TCP层      │ 无               │
│ 握手     │ TCP 3次    │ TCP+TLS    │ QUIC 0-1RTT      │
│ 连接迁移 │ 不支持     │ 不支持     │ 支持             │
│ 加密     │ 可选       │ 实际必须   │ 必须(TLS 1.3)    │
└──────────┴────────────┴────────────┴──────────────────┘

实际部署建议：
  1. 启用 HTTP/2：Nginx 配置 listen 443 ssl http2
  2. HTTP/3：需要额外配置 QUIC 支持
     // Nginx 1.25+ 原生支持
     listen 443 quic reuseport;
     listen 443 ssl http2;
     add_header Alt-Svc 'h3=":443"; ma=86400';
  3. 证书：必须使用 TLS 证书（Let's Encrypt 免费）
  4. 兼容：同时支持 HTTP/1.1、HTTP/2、HTTP/3

浏览器支持：
  HTTP/2：Chrome 41+, Firefox 36+, Edge 12+
  HTTP/3：Chrome 87+, Firefox 88+, Safari 14+

性能优化关键：
  - HTTP/2：减少请求数不再是首要目标
  - 应关注：关键渲染路径、资源优先级
  - 不再需要合并 CSS/JS 文件
  - 可以细粒度拆分资源
  - 图片使用 srcset 按需加载
```


> **Note:** HTTP/2 通过二进制分帧和多路复用解决了 HTTP 层队头阻塞，HPACK 头部压缩减少了冗余传输。HTTP/3 基于 QUIC 协议进一步消除了 TCP 层队头阻塞，支持 0-RTT 快速连接和连接迁移。实际部署建议同时支持多个版本以确保兼容性。


<!-- Converted from: 02_HTTP2与HTTP3.html -->
