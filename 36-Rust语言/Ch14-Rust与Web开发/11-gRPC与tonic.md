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

## 三、注意事项与常见陷阱

1. **Proto 管理**：使用 build.rs 自动生成代码
2. **版本兼容**：注意 Proto 的向后兼容性
3. **流式处理**：支持流式 RPC
4. **TLS 配置**：生产环境配置 TLS
5. **超时重试**：配置客户端超时和重试
