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

## 三、注意事项与常见陷阱

1. **连接管理**：跟踪活跃连接
2. **消息格式**：使用 JSON 等结构化格式
3. **心跳机制**：实现心跳检测断线
4. **安全考虑**：验证 WebSocket 连接来源
5. **性能**：大量连接时考虑使用专门的 WebSocket 服务器
