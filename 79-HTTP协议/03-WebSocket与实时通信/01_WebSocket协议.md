# WebSocket 协议


## WebSocket 协议


WebSocket实时通信长连接


WebSocket 是一种全双工通信协议，在单个 TCP 连接上提供持久的双向通信通道，适用于需要实时数据传输的场景。


## WebSocket 握手过程


```
WebSocket 基于 HTTP 升级：
  客户端首先发送 HTTP 请求
  服务端同意升级后切换到 WebSocket 协议

客户端握手请求：
  GET /chat HTTP/1.1
  Host: server.example.com
  Upgrade: websocket                  // 请求协议升级
  Connection: Upgrade                 // 连接升级
  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
  // 随机 Base64 编码的16字节，防止缓存代理
  Sec-WebSocket-Version: 13           // 协议版本
  Sec-WebSocket-Protocol: chat, superchat  // 子协议
  Origin: http://example.com          // 来源验证

服务端握手响应：
  HTTP/1.1 101 Switching Protocols    // 状态码101
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
  // = Base64(SHA-1(client_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
  Sec-WebSocket-Protocol: chat        // 选择的子协议

  // 此后连接切换为 WebSocket 协议

握手中的安全验证：
  GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  // 这是一个固定的魔法字符串
  accept_key = base64(sha1(client_key + GUID))
  // 服务端必须返回正确的 accept_key
  // 防止普通 HTTP 请求意外升级

JavaScript 客户端 API：
  // 建立连接
  const ws = new WebSocket('wss://example.com/ws');

  // 事件监听
  ws.onopen = () => console.log('连接已建立');
  ws.onmessage = (event) => console.log('收到:', event.data);
  ws.onerror = (err) => console.log('错误:', err);
  ws.onclose = (event) => {
    console.log('连接关闭:', event.code, event.reason);
  };

  // 发送数据
  ws.send('Hello Server');
  ws.send(JSON.stringify({type: 'message', content: 'Hi'}));
  ws.send(arrayBuffer);  // 二进制数据

  // 关闭连接
  ws.close(1000, '正常关闭');
```


## WebSocket 数据帧格式


```
帧结构：
  ┌─────────────────────────────────────────────┐
  │  0                   1                   2  │
  │  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0│
  │ +-+-+-+-+-------+-+-------------+---------+ │
  │ |F|R|R|R| opcode|M| Payload len | Extended| │
  │ |I|S|S|S|  (4)  |A|   (7 bits)  | payload| │
  │ |N|V|V|V|       |S|             | length | │
  │ | |1|2|3|       |K|             |        | │
  │ +-+-+-+-+-------+-+-------------+ - - - - + │
  │ |  Extended payload length (16/64 bits)   | │
  │ + - - - - - - - - - - - - - - - - - - - - + │
  │ |  Masking-key (32 bits, if MASK=1)       | │
  │ + - - - - - - - - - - - - - - - - - - - - + │
  │ |  Payload Data                           | │
  │ +-----------------------------------------+ │
  └─────────────────────────────────────────────┘

操作码（Opcode）：
  0x0  - 续传帧（Continuation）
  0x1  - 文本帧（Text）
  0x2  - 二进制帧（Binary）
  0x8  - 关闭帧（Close）
  0x9  - Ping 帧
  0xA  - Pong 帧

FIN 位：
  FIN=1 表示这是消息的最后一帧
  大消息可以分多帧传输

掩码（Masking）：
  客户端→服务端的帧必须掩码（MASK=1）
  服务端→客户端的帧不能掩码（MASK=0）
  防止缓存代理混淆 HTTP 请求

  // 掩码算法
  for i, byte in enumerate(payload):
      masked[i] = byte ^ masking_key[i % 4]

关闭帧：
  // 关闭码
  1000 - 正常关闭
  1001 - 端点离开（页面关闭）
  1002 - 协议错误
  1003 - 不支持的数据类型
  1006 - 异常关闭（无关闭帧）
  1011 - 服务器错误
  1012 - 服务器重启

  // 关闭握手
  客户端 → 服务端: Close 帧 (code=1000, reason="bye")
  服务端 → 客户端: Close 帧 (code=1000)
  双方关闭 TCP 连接
```


## 心跳保活与断线重连


```
心跳机制：
  WebSocket 协议层有 Ping/Pong 帧
  但通常在应用层实现心跳

  // 应用层心跳
  客户端每30秒发送：
  {"type": "ping", "timestamp": 1698765432}

  服务端回复：
  {"type": "pong", "timestamp": 1698765432}

  // 超时检测
  客户端发送 ping 后，60秒内没收到 pong
  → 认为连接已断开 → 触发重连

  // 服务端检测
  90秒没收到客户端消息 → 主动关闭连接

断线重连策略：
  // 指数退避 + 抖动
  function reconnect(attempt) {
    const baseDelay = Math.min(1000 * Math.pow(2, attempt), 30000);
    const jitter = Math.random() * 1000;
    const delay = baseDelay + jitter;

    // 重连间隔：1s, 2s, 4s, 8s, 16s, 30s(max)
    setTimeout(() => {
      connect().catch(() => reconnect(attempt + 1));
    }, delay);
  }

完整重连实现：
  class ReconnectingWebSocket {
    constructor(url) {
      this.url = url;
      this.attempts = 0;
      this.connect();
    }

    connect() {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.attempts = 0;  // 重置重试计数
        this.onopen?.();
        this.resubscribe();  // 重新订阅
      };

      this.ws.onclose = (event) => {
        if (event.code !== 1000) {  // 非正常关闭
          this.reconnect();
        }
      };

      this.ws.onmessage = (event) => {
        this.lastMessageTime = Date.now();
        this.onmessage?.(event);
      };
    }

    reconnect() {
      const delay = Math.min(1000 * 2 ** this.attempts, 30000);
      this.attempts++;
      setTimeout(() => this.connect(), delay);
    }

    send(data) {
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(data);
      } else {
        this.queue.push(data);  // 离线队列
      }
    }
  }

消息可靠性：
  // 客户端给每条消息分配序号
  { "seq": 1, "type": "chat", "content": "Hello" }

  // 服务端 ACK
  { "seq": 1, "type": "ack" }

  // 客户端超时未收到 ACK → 重发
  // 服务端用 seq 去重
```


## WSS（WebSocket Secure）


```
WS vs WSS：
  ws://  - 明文 WebSocket（不安全）
  wss:// - 基于 TLS 的 WebSocket（推荐）

  // WebSocket over TLS
  类似 HTTPS = HTTP over TLS
  WSS = WebSocket over TLS

Nginx 代理配置：
  server {
      listen 443 ssl;
      server_name ws.example.com;

      ssl_certificate /path/to/cert.pem;
      ssl_certificate_key /path/to/key.pem;

      location /ws {
          proxy_pass http://backend:8080;
          proxy_http_version 1.1;

          # 关键配置
          proxy_set_header Upgrade $http_upgrade;
          proxy_set_header Connection "upgrade";
          proxy_set_header Host $host;

          # 超时配置
          proxy_read_timeout 3600s;    # 1小时
          proxy_send_timeout 3600s;

          # 真实IP
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      }
  }

认证方式：
  1. 握手时通过 Cookie/Token 认证
     new WebSocket('wss://example.com/ws?token=xxx')

  2. 连接建立后发送认证消息
     ws.onopen = () => {
       ws.send(JSON.stringify({type: 'auth', token: 'xxx'}));
     };

  3. 子协议中携带认证
     new WebSocket('wss://example.com/ws', ['auth-token-xxx']);

Nginx 层认证：
  // 用 map 指令验证 Token
  map $arg_token $ws_auth {
      default "invalid";
      "valid_token" "valid";
  }

  location /ws {
      if ($ws_auth = "invalid") {
          return 403;
      }
      proxy_pass http://backend:8080;
      ...
  }
```


## 实时通信方案对比


```
┌──────────────┬────────────┬────────────┬────────────┬──────────┐
│ 方案         │ 延迟       │ 开销       │ 复杂度     │ 适用场景 │
├──────────────┼────────────┼────────────┼────────────┼──────────┤
│ 短轮询       │ 高(秒级)   │ 高(频繁请求)│ 低         │ 低频更新 │
│ 长轮询       │ 中(百ms)   │ 中(连接重建)│ 中         │ 中频更新 │
│ SSE          │ 低(ms级)   │ 低(单向)   │ 低         │ 服务推送 │
│ WebSocket    │ 低(ms级)   │ 低(长连接) │ 中         │ 双向实时 │
└──────────────┴────────────┴────────────┴────────────┴──────────┘

短轮询（Polling）：
  setInterval(() => {
    fetch('/api/messages').then(handleResponse);
  }, 3000);
  // 每3秒请求一次
  // 问题：延迟高、浪费带宽、服务器压力大

长轮询（Long Polling）：
  function longPoll() {
    fetch('/api/messages?timeout=30')
      .then(handleResponse)
      .finally(longPoll);  // 收到响应后立即再次请求
  }
  // 服务端挂起请求直到有新数据或超时
  // 减少了无效请求，但仍有连接开销

SSE（Server-Sent Events）：
  const es = new EventSource('/api/stream');
  es.onmessage = (event) => console.log(event.data);
  // 服务端单向推送，自动重连
  // 适合：股票行情、新闻推送、通知

WebSocket：
  const ws = new WebSocket('wss://example.com/ws');
  ws.onmessage = (event) => console.log(event.data);
  ws.send('Hello');
  // 双向通信，最低延迟
  // 适合：聊天、游戏、协同编辑

选择建议：
  只需服务端推送 → SSE
  需要双向通信 → WebSocket
  低频更新 → 短轮询
  不支持 WebSocket 的环境 → 长轮询
  需要兼容 HTTP 基础设施 → SSE
```


> **Note:** WebSocket 通过 HTTP 升级建立全双工通信，握手后切换为帧传输。实际项目中需要实现心跳保活和指数退避重连。WSS（WebSocket Secure）基于 TLS 加密，生产环境必须使用。Nginx 代理需要正确配置 Upgrade 和 Connection 头部以及较长的超时时间。


<!-- Converted from: 01_WebSocket协议.html -->
