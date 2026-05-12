# WebSocket支持

## 一、概念说明

Rust 框架支持 WebSocket 实现实时双向通信。

```rust
use axum::{extract::ws::{WebSocket, WebSocketUpgrade}, response::IntoResponse};

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    while let Some(msg) = socket.recv().await {
        if let Ok(msg) = msg {
            socket.send(msg).await.unwrap();
        }
    }
}
```

## 二、具体用法

### 2.1 广播消息

```rust
use tokio::sync::broadcast;

struct AppState {
    tx: broadcast::Sender<String>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state.tx))
}

async fn handle_socket(mut socket: WebSocket, tx: broadcast::Sender<String>) {
    let mut rx = tx.subscribe();
    // 接收和广播消息
}
```

### 2.3 聊天室完整示例

```rust
use axum::{extract::ws::{WebSocket, WebSocketUpgrade}, extract::State, response::IntoResponse, Router};
use tokio::sync::broadcast;
use std::sync::Arc;

struct AppState {
    tx: broadcast::Sender<String>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // 接收消息并广播
    let tx = state.tx.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let axum::extract::ws::Message::Text(text) = msg {
                let _ = tx.send(text);
            }
        }
    });

    // 发送广播消息给客户端
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(msg.into()).await.is_err() {
                break;
            }
        }
    });

    tokio::select! {
        _ = recv_task => {},
        _ = send_task => {},
    }
}
```

### 2.4 消息协议设计

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
enum WsMessage {
    Text { content: String },
    Binary { data: Vec<u8> },
    Ping,
    Pong,
    Close { reason: String },
}

// 使用 JSON 序列化
fn serialize_message(msg: WsMessage) -> String {
    serde_json::to_string(&msg).unwrap()
}

fn deserialize_message(data: &str) -> Result<WsMessage, serde_json::Error> {
    serde_json::from_str(data)
}
```

### 2.5 心跳检测

```rust
async fn heartbeat(socket: &mut WebSocket) {
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        if socket.send(axum::extract::ws::Message::Ping(vec![])).await.is_err() {
            // 连接已断开
            break;
        }
    }
}
```

## 四、WebSocket vs SSE vs gRPC

| 协议 | 双向通信 | 复杂度 | 适用场景 |
|------|----------|--------|---------|
| WebSocket | 完全双向 | 中等 | 实时聊天、游戏 |
| SSE | 服务器->客户端 | 低 | 通知、实时更新 |
| gRPC | 双向流 | 高 | 微服务通信 |

## 五、注意事项与常见陷阱

1. **连接管理**：跟踪活跃连接，及时清理断开的连接
2. **消息格式**：使用 JSON 等结构化格式，定义清晰的消息协议
3. **心跳机制**：实现心跳检测断线，避免僵尸连接
4. **安全考虑**：验证 WebSocket 连接来源，使用 WSS（加密）
5. **性能**：大量连接时考虑使用专门的 WebSocket 服务器
6. **背压处理**：使用有界通道防止消息堆积
7. **跨域支持**：配置正确的 CORS 和 WebSocket 升级策略
