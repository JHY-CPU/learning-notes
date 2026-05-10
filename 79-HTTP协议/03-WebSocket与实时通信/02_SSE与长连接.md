# SSE 与长连接


## SSE 与长连接技术


SSE长轮询流式响应


Server-Sent Events（SSE）是一种服务端向客户端推送事件的标准，基于简单的 HTTP 连接，比 WebSocket 更轻量，适合单向数据推送场景。


## Server-Sent Events（SSE）协议


```
SSE 协议格式：
  // 基于 HTTP 长连接
  // Content-Type: text/event-stream
  // 每条事件以空行分隔

  // 服务端响应
  HTTP/1.1 200 OK
  Content-Type: text/event-stream
  Cache-Control: no-cache
  Connection: keep-alive
  Access-Control-Allow-Origin: *

  // 事件格式
  event: message
  id: 1
  data: {"type": "stock", "symbol": "AAPL", "price": 150.25}

  event: notification
  id: 2
  data: 你有一条新消息

  // 注释行（心跳）
  : heartbeat

  // 重连时间（毫秒）
  retry: 5000

字段说明：
  event  - 事件类型（可选，默认 "message"）
  id     - 事件ID，用于断线重连（Last-Event-ID）
  data   - 事件数据（多行用多个 data:）
  retry  - 重连间隔（毫秒）
  : xxx  - 注释行（用于保活）

多行数据：
  data: 第一行
  data: 第二行
  data: 第三行

  // 客户端收到 "第一行\n第二行\n第三行"

客户端 API（JavaScript）：
  // 建立连接
  const es = new EventSource('/api/events');

  // 监听默认事件
  es.onmessage = (event) => {
    console.log('收到:', event.data);
    console.log('ID:', event.lastEventId);
  };

  // 监听自定义事件
  es.addEventListener('stock', (event) => {
    const data = JSON.parse(event.data);
    updateStockPrice(data);
  });

  es.addEventListener('notification', (event) => {
    showNotification(event.data);
  });

  // 连接状态
  es.onopen = () => console.log('SSE 连接已建立');
  es.onerror = (err) => {
    if (es.readyState === EventSource.CONNECTING) {
      console.log('正在重连...');
    }
  };

  // 关闭连接
  es.close();

带认证的 SSE：
  // 方法1：URL 参数（简单但不安全）
  const es = new EventSource('/events?token=xxx');

  // 方法2：Fetch API + ReadableStream（推荐）
  const response = await fetch('/events', {
    headers: { 'Authorization': 'Bearer xxx' }
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    parseSSE(text);  // 解析 SSE 格式
  }
```


## SSE 服务端实现


```
Node.js (Express)：
  app.get('/events', (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // 发送初始消息
    res.write('data: 连接成功\n\n');

    // 每5秒发送一次数据
    const interval = setInterval(() => {
      const data = JSON.stringify({ time: Date.now() });
      res.write(`data: ${data}\n\n`);
    }, 5000);

    // 心跳（每30秒）
    const heartbeat = setInterval(() => {
      res.write(': heartbeat\n\n');
    }, 30000);

    // 客户端断开连接时清理
    req.on('close', () => {
      clearInterval(interval);
      clearInterval(heartbeat);
      res.end();
    });
  });

Python (Flask)：
  from flask import Flask, Response
  import json, time

  @app.route('/events')
  def events():
      def generate():
          while True:
              data = json.dumps({'time': time.time()})
              yield f"data: {data}\n\n"
              time.sleep(5)

      return Response(
          generate(),
          mimetype='text/event-stream',
          headers={'Cache-Control': 'no-cache'}
  )

Java (Spring)：
  @GetMapping(path = "/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
  public Flux<ServerSentEvent<String>> streamEvents() {
      return Flux.interval(Duration.ofSeconds(5))
          .map(seq -> ServerSentEvent.<String>builder()
              .id(String.valueOf(seq))
              .event("message")
              .data("Event #" + seq)
              .build());
  }

Nginx 代理配置：
  location /events {
      proxy_pass http://backend:8080;
      proxy_http_version 1.1;
      proxy_set_header Connection '';
      proxy_buffering off;           // 关键：关闭缓冲
      proxy_cache off;
      chunked_transfer_encoding on;
      tcp_nodelay on;
      proxy_read_timeout 3600s;      // 长超时
  }

断线重连（Last-Event-ID）：
  // 客户端断线后自动重连
  // 浏览器自动在请求头中加入
  GET /events HTTP/1.1
  Last-Event-ID: 42
  // 服务端从 ID 42 之后开始推送

  // 服务端处理
  app.get('/events', (req, res) => {
    const lastId = req.headers['last-event-id'] || 0;
    const missedEvents = getEventsAfter(lastId);
    missedEvents.forEach(e => {
      res.write(`id: ${e.id}\ndata: ${e.data}\n\n`);
    });
    // 继续推送新事件...
  });
```


## 长轮询（Long Polling）


```
长轮询原理：
  客户端发送请求 → 服务端挂起等待
  有新数据时立即返回，或超时后返回空响应
  客户端收到响应后立即发起下一次请求

  ┌─────────────────────────────────────────────┐
  │  客户端              服务端                 │
  │    │── 请求 ────────→│                      │
  │    │                  │ (等待数据...)        │
  │    │                  │ (有新数据!)          │
  │    │←── 响应 ────────│                      │
  │    │── 请求 ────────→│                      │
  │    │                  │ (等待数据...)        │
  │    │←── 超时响应 ────│                      │
  │    │── 请求 ────────→│                      │
  └─────────────────────────────────────────────┘

客户端实现：
  async function longPoll() {
    while (true) {
      try {
        const response = await fetch('/api/poll', {
          signal: AbortSignal.timeout(30000)  // 30秒超时
        });
        const data = await response.json();
        if (data.events.length > 0) {
          handleEvents(data.events);
        }
      } catch (err) {
        await sleep(1000);  // 错误后等待
      }
      // 立即发起下一次请求
    }
  }

服务端实现（Node.js）：
  // 内存中的事件队列
  const waitingClients = [];

  app.get('/api/poll', (req, res) => {
    const timeout = setTimeout(() => {
      // 超时返回空
      res.json({ events: [] });
    }, 25000);

    // 将客户端加入等待列表
    waitingClients.push({ res, timeout });
  });

  // 有新事件时通知所有等待的客户端
  function broadcast(event) {
    while (waitingClients.length > 0) {
      const client = waitingClients.shift();
      clearTimeout(client.timeout);
      client.res.json({ events: [event] });
    }
  }

长轮询优化：
  1. 设置合理超时（25-30秒）
  2. 客户端随机延迟重试（防同时重连）
  3. 服务端限制等待队列大小
  4. 支持 Last-Event-ID 增量获取
  5. 监控连接数和内存使用

分块传输响应：
  // 另一种长连接方式
  // 使用 Transfer-Encoding: chunked

  app.get('/stream', (req, res) => {
    res.setHeader('Content-Type', 'text/plain');
    res.setHeader('Transfer-Encoding', 'chunked');

    const interval = setInterval(() => {
      res.write(JSON.stringify({ time: Date.now() }) + '\n');
    }, 1000);

    req.on('close', () => clearInterval(interval));
  });

  // 客户端逐行读取
  const response = await fetch('/stream');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value);
    const lines = buffer.split('\n');
    buffer = lines.pop();  // 保留不完整的行
    lines.forEach(line => {
      if (line) handleData(JSON.parse(line));
    });
  }
```


## 实时通信方案选择指南


```
方案特性对比：
┌──────────────────┬────────┬────────┬────────┬──────────┐
│ 特性             │ 短轮询 │ 长轮询 │ SSE    │ WebSocket│
├──────────────────┼────────┼────────┼────────┼──────────┤
│ 通信方向         │ 客户→服│ 客户→服│ 服→客户│ 双向     │
│ 协议             │ HTTP   │ HTTP   │ HTTP   │ WS/WSS   │
│ 浏览器支持       │ 全部   │ 全部   │ 现代   │ 现代     │
│ 自动重连         │ 无     │ 无     │ 内置   │ 需实现   │
│ 数据格式         │ JSON等 │ JSON等 │ 文本   │ 文本+二进│
│ 连接开销         │ 高     │ 中     │ 低     │ 低       │
│ 代理兼容性       │ 好     │ 好     │ 好     │ 需配置   │
│ 复杂度           │ 低     │ 中     │ 低     │ 中高     │
└──────────────────┴────────┴────────┴────────┴──────────┘

使用场景推荐：

SSE 最佳场景：
  ✓ 股票/加密货币行情推送
  ✓ 新闻/社交媒体 Feed 更新
  ✓ AI 流式输出（ChatGPT 打字效果）
  ✓ 系统监控仪表板
  ✓ 通知中心
  ✓ 服务器日志实时查看

WebSocket 最佳场景：
  ✓ 即时通讯（微信/钉钉）
  ✓ 在线多人游戏
  ✓ 实时协同编辑（Google Docs）
  ✓ 视频会议信令
  ✓ 实时白板/画板
  ✓ 需要客户端频繁发送数据的场景

长轮询最佳场景：
  ✓ 需要兼容老旧浏览器
  ✓ 企业内网（防火墙限制 WebSocket）
  ✓ 简单的消息通知系统
  ✓ 短时间的实时需求

ChatGPT 流式输出示例：
  // SSE 非常适合 AI 流式输出
  // 服务端逐 token 推送

  // 服务端（Python）
  def chat_stream(prompt):
      for token in llm.generate(prompt):
          yield f"data: {json.dumps({'token': token})}\n\n"

  @app.route('/chat', methods=['POST'])
  def chat():
      prompt = request.json['prompt']
      return Response(chat_stream(prompt), mimetype='text/event-stream')

  // 客户端
  const es = new EventSource('/chat?prompt=' + encodeURIComponent(msg));
  es.onmessage = (e) => {
    const { token } = JSON.parse(e.data);
    document.getElementById('output').textContent += token;
  };

SSE 限制：
  - HTTP/1.1 浏览器限制同域6个连接（SSE + 其他请求共享）
  - HTTP/2 下限制变为同域100个流
  - 只支持文本数据（二进制需 Base64 编码）
  - IE/Edge Legacy 不支持（可用 polyfill）
  - 某些代理/CDN 可能缓冲响应（需配置 proxy_buffering off）
```


> **Note:** SSE 基于 HTTP 长连接实现服务端单向推送，比 WebSocket 更轻量且兼容 HTTP 基础设施。非常适合 AI 流式输出、行情推送等场景。长轮询作为备选方案兼容性最好。选择原则：单向推送用 SSE，双向通信用 WebSocket，兼容优先用长轮询。


<!-- Converted from: 02_SSE与长连接.html -->
