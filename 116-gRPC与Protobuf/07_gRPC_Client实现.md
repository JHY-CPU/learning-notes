# gRPC Client 实现

## Python gRPC 客户端

### 基本客户端调用

```python
# client/client.py
import grpc
import greeter_pb2
import greeter_pb2_grpc


def run():
    """基本的一元 RPC 调用"""
    # 创建不安全的通道（开发环境）
    with grpc.insecure_channel("localhost:50051") as channel:
        # 创建客户端存根
        stub = greeter_pb2_grpc.GreeterServiceStub(channel)

        # 调用一元 RPC
        request = greeter_pb2.HelloRequest(
            name="张三",
            language="zh"
        )
        response = stub.SayHello(request)
        print(f"服务端响应: {response.greeting}")
        print(f"时间戳: {response.timestamp}")


if __name__ == "__main__":
    run()
```

### 带超时和重试的客户端

```python
# client/client_with_retry.py
import grpc
import time
import greeter_pb2
import greeter_pb2_grpc
from grpc import RpcError


def call_with_retry(stub, request, max_retries=3, timeout=5):
    """带重试的 RPC 调用"""
    for attempt in range(max_retries):
        try:
            response = stub.SayHello(request, timeout=timeout)
            return response
        except RpcError as e:
            code = e.code()
            # 只对可重试的错误进行重试
            if code in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"调用失败，{wait_time}秒后重试... (第{attempt+1}次)")
                    time.sleep(wait_time)
                    continue
            raise


def run():
    channel = grpc.insecure_channel(
        "localhost:50051",
        options=[
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
        ]
    )
    stub = greeter_pb2_grpc.GreeterServiceStub(channel)

    request = greeter_pb2.HelloRequest(name="李四", language="zh")

    try:
        response = call_with_retry(stub, request)
        print(f"响应: {response.greeting}")
    except RpcError as e:
        print(f"最终失败: {e.code()} - {e.details()}")

    channel.close()


if __name__ == "__main__":
    run()
```

### 带元数据的客户端

```python
# client/client_with_metadata.py
import grpc
import greeter_pb2
import greeter_pb2_grpc


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = greeter_pb2_grpc.GreeterServiceStub(channel)

    # 添加元数据（HTTP 头部）
    metadata = [
        ("authorization", "Bearer valid-token-1"),
        ("x-request-id", "req-12345"),
        ("x-trace-id", "trace-67890"),
    ]

    request = greeter_pb2.HelloRequest(name="王五", language="zh")

    # 通过 metadata 参数传递元数据
    response = stub.SayHello(
        request,
        metadata=metadata,
        timeout=10
    )
    print(f"响应: {response.greeting}")

    channel.close()


if __name__ == "__main__":
    run()
```

### 安全通道客户端

```python
# client/client_secure.py
import grpc
import greeter_pb2
import greeter_pb2_grpc


def run_with_tls():
    """使用 TLS 加密的客户端"""
    # 加载 CA 证书
    with open("certs/ca.crt", "rb") as f:
        ca_cert = f.read()

    # 创建 SSL 凭证
    credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
    )

    # 创建安全通道
    with grpc.secure_channel("localhost:50051", credentials) as channel:
        stub = greeter_pb2_grpc.GreeterServiceStub(channel)
        request = greeter_pb2.HelloRequest(name="安全用户", language="zh")
        response = stub.SayHello(request, timeout=10)
        print(f"安全响应: {response.greeting}")


if __name__ == "__main__":
    run_with_tls()
```

## Go gRPC 客户端

### 基本客户端调用

```go
// client/main.go
package main

import (
    "context"
    "log"
    "time"

    pb "example.com/project/gen"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

func main() {
    // 建立连接
    conn, err := grpc.Dial(
        "localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
    if err != nil {
        log.Fatalf("无法连接: %v", err)
    }
    defer conn.Close()

    // 创建客户端
    client := pb.NewGreeterServiceClient(conn)

    // 设置上下文超时
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    // 调用一元 RPC
    resp, err := client.SayHello(ctx, &pb.HelloRequest{
        Name:     "张三",
        Language: "zh",
    })
    if err != nil {
        log.Fatalf("调用失败: %v", err)
    }

    log.Printf("响应: %s", resp.Greeting)
}
```

### 带元数据和拦截器的客户端

```go
// client/client_with_interceptor.go
package main

import (
    "context"
    "log"
    "time"

    pb "example.com/project/gen"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/grpc/metadata"
)

// 元数据拦截器
func metadataInterceptor(
    ctx context.Context,
    method string,
    req, reply interface{},
    cc *grpc.ClientConn,
    invoker grpc.UnaryInvoker,
    opts ...grpc.CallOption,
) error {
    // 添加元数据
    md := metadata.Pairs(
        "authorization", "Bearer valid-token-1",
        "x-request-id", "req-12345",
    )
    ctx = metadata.NewOutgoingContext(ctx, md)

    log.Printf("调用方法: %s", method)
    start := time.Now()

    err := invoker(ctx, method, req, reply, cc, opts...)

    log.Printf("方法 %s 耗时: %v", method, time.Since(start))
    return err
}

func main() {
    conn, err := grpc.Dial(
        "localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithUnaryInterceptor(metadataInterceptor),
    )
    if err != nil {
        log.Fatalf("无法连接: %v", err)
    }
    defer conn.Close()

    client := pb.NewGreeterServiceClient(conn)

    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    resp, err := client.SayHello(ctx, &pb.HelloRequest{
        Name:     "李四",
        Language: "zh",
    })
    if err != nil {
        log.Fatalf("调用失败: %v", err)
    }

    log.Printf("响应: %s", resp.Greeting)
}
```

## 通道选项（Channel Options）

```go
// Go 中的常用通道选项
opts := []grpc.DialOption{
    // 消息大小限制
    grpc.WithDefaultCallOptions(
        grpc.MaxCallRecvMsgSize(50 * 1024 * 1024),
        grpc.MaxCallSendMsgSize(50 * 1024 * 1024),
    ),

    // Keepalive 配置
    grpc.WithKeepaliveParams(keepalive.ClientParameters{
        Time:                10 * time.Second,
        Timeout:             5 * time.Second,
        PermitWithoutStream: true,
    }),

    // 连接池配置
    grpc.WithInitialWindowSize(1 << 20),
    grpc.WithInitialConnWindowSize(1 << 20),
}

conn, err := grpc.Dial("localhost:50051", opts...)
```

```python
# Python 中的通道选项
options = [
    ("grpc.max_send_message_length", 50 * 1024 * 1024),
    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.initial_reconnect_backoff_ms", 1000),
]

channel = grpc.insecure_channel("localhost:50051", options=options)
```

## 上下文与截止时间

```go
// Go 中的上下文使用
import (
    "context"
    "time"
    "google.golang.org/grpc/status"
    "google.golang.org/grpc/codes"
)

func callWithDeadline(client pb.GreeterServiceClient) {
    // 设置 5 秒截止时间
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    resp, err := client.SayHello(ctx, &pb.HelloRequest{
        Name: "张三",
    })
    if err != nil {
        // 检查是否超时
        st, ok := status.FromError(err)
        if ok {
            switch st.Code() {
            case codes.DeadlineExceeded:
                log.Println("请求超时")
            case codes.Canceled:
                log.Println("请求被取消")
            default:
                log.Printf("其他错误: %s", st.Message())
            }
        }
        return
    }

    log.Printf("响应: %s", resp.Greeting)
}
```

```python
# Python 中的截止时间
import grpc

# 方式一：在调用时指定超时（秒）
response = stub.SayHello(request, timeout=5)

# 方式二：使用 deadline
import datetime
deadline = datetime.datetime.now() + datetime.timedelta(seconds=5)
response = stub.SayHello(request, deadline=deadline)
```

## 连接管理

```go
// Go 中的连接池和复用
import (
    "sync"
    "google.golang.org/grpc"
)

type ClientManager struct {
    mu      sync.RWMutex
    conns   map[string]*grpc.ClientConn
    address string
}

func NewClientManager(address string) *ClientManager {
    return &ClientManager{
        conns:   make(map[string]*grpc.ClientConn),
        address: address,
    }
}

func (m *ClientManager) GetConnection() (*grpc.ClientConn, error) {
    m.mu.RLock()
    conn, ok := m.conns[m.address]
    m.mu.RUnlock()

    if ok && conn.GetState() != grpc.Shutdown {
        return conn, nil
    }

    m.mu.Lock()
    defer m.mu.Unlock()

    conn, err := grpc.Dial(
        m.address,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
    if err != nil {
        return nil, err
    }

    m.conns[m.address] = conn
    return conn, nil
}

func (m *ClientManager) Close() {
    m.mu.Lock()
    defer m.mu.Unlock()

    for _, conn := range m.conns {
        conn.Close()
    }
}
```

## 小结

gRPC 客户端实现需要创建通道（Channel）和存根（Stub），通过存根调用远程方法。Python 和 Go 都支持元数据、超时控制、重试机制和安全通道。合理使用通道选项和连接管理可以优化性能和可靠性。拦截器机制允许在调用前后插入横切关注点逻辑。
