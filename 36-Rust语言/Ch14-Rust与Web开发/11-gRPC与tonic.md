# gRPC与tonic

## 一、概念说明

Tonic 是 Rust 的 gRPC 实现，基于 HTTP/2 和 Protocol Buffers。

```rust
// proto/service.proto
service Greeter {
    rpc SayHello (HelloRequest) returns (HelloResponse);
}

// 生成代码后使用 tonic::include_proto!("service");
```

## 二、具体用法

### 2.1 服务端实现

```rust
use tonic::{Request, Response, Status};

#[derive(Default)]
pub struct MyGreeter {}

#[tonic::async_trait]
impl Greeter for MyGreeter {
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloResponse>, Status> {
        let reply = HelloResponse {
            message: format!("你好，{}！", request.into_inner().name),
        };
        Ok(Response::new(reply))
    }
}
```

### 2.2 客户端调用

```rust
use tonic::transport::Channel;

async fn call_service() -> Result<(), Box<dyn std::error::Error>> {
    let channel = Channel::from_static("http://[::1]:50051")
        .connect()
        .await?;

    let mut client = GreeterClient::new(channel);
    let request = tonic::Request::new(HelloRequest { name: "张三".into() });
    let response = client.say_hello(request).await?;
    println!("响应: {:?}", response);
    Ok(())
}
```

### 2.3 流式 RPC

```rust
// 服务端流式
async fn server_streaming(
    &self,
    request: Request<HelloRequest>,
) -> Result<Response<Self::ServerStreamingStream>, Status> {
    let (tx, rx) = tokio::sync::mpsc::channel(10);

    tokio::spawn(async move {
        for i in 0..10 {
            tx.send(Ok(HelloResponse {
                message: format!("消息 {}", i),
            })).await.unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    Ok(Response::new(ReceiverStream::new(rx)))
}

// 客户端流式
async fn client_streaming(
    &self,
    request: Request<Streaming<HelloRequest>>,
) -> Result<Response<HelloResponse>, Status> {
    let mut stream = request.into_inner();
    let mut messages = Vec::new();

    while let Some(req) = stream.next().await {
        messages.push(req?.name);
    }

    Ok(Response::new(HelloResponse {
        message: format!("收到 {} 条消息", messages.len()),
    }))
}
```

### 2.4 Proto 文件示例

```protobuf
// proto/greeter.proto
syntax = "proto3";
package greeter;

service Greeter {
    rpc SayHello (HelloRequest) returns (HelloResponse);
    rpc SayHelloStream (HelloRequest) returns (stream HelloResponse);
    rpc SayHelloClientStream (stream HelloRequest) returns (HelloResponse);
    rpc SayHelloBidiStream (stream HelloRequest) returns (stream HelloResponse);
}

message HelloRequest {
    string name = 1;
}

message HelloResponse {
    string message = 1;
}
```

### 2.5 build.rs 自动生成

```rust
// build.rs
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&["proto/greeter.proto"], &["proto"])?;
    Ok(())
}
```

## 四、gRPC vs REST 对比

| 特性 | gRPC | REST |
|------|------|------|
| 协议 | HTTP/2 | HTTP/1.1 |
| 数据格式 | Protocol Buffers | JSON |
| 类型安全 | 强类型 | 弱类型 |
| 流式支持 | 原生支持 | 需要 WebSocket |
| 浏览器支持 | 需要 gRPC-Web | 原生支持 |
| 性能 | 更高 | 一般 |

## 五、注意事项与常见陷阱

1. **Proto 管理**：使用 build.rs 自动生成代码，版本控制 proto 文件
2. **版本兼容**：注意 Proto 的向后兼容性，不要删除或重编号字段
3. **流式处理**：支持流式 RPC，适用于大数据传输
4. **TLS 配置**：生产环境配置 TLS，确保通信安全
5. **超时重试**：配置客户端超时和重试，避免长时间等待
6. **错误码**：使用正确的 gRPC 状态码
7. **负载均衡**：配置客户端负载均衡
