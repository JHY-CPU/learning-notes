# gRPC Server 实现

## Python gRPC 服务端

### 项目结构

```
project/
├── protos/
│   └── greeter.proto
├── server/
│   ├── __init__.py
│   ├── greeter_servicer.py
│   └── main.py
└── gen/
    ├── greeter_pb2.py
    └── greeter_pb2_grpc.py
```

### 完整的 Python 服务端实现

```protobuf
// protos/greeter.proto
syntax = "proto3";

package greeter;

message HelloRequest {
  string name = 1;
  string language = 2;
}

message HelloResponse {
  string greeting = 1;
  int64 timestamp = 2;
}

message ListGreetingsRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message ListGreetingsResponse {
  repeated HelloResponse greetings = 1;
  string next_page_token = 2;
}

service GreeterService {
  rpc SayHello(HelloRequest) returns (HelloResponse);
  rpc ListGreetings(ListGreetingsRequest) returns (ListGreetingsResponse);
}
```

```python
# server/greeter_servicer.py
import time
import grpc
from concurrent import futures
import greeter_pb2
import greeter_pb2_grpc

class GreeterServiceServicer(greeter_pb2_grpc.GreeterServiceServicer):
    """GreeterService 的实现"""

    def __init__(self):
        self.greetings_store = []
        self._init_sample_data()

    def _init_sample_data(self):
        """初始化示例数据"""
        languages = {
            "zh": "你好",
            "en": "Hello",
            "ja": "こんにちは",
            "ko": "안녕하세요",
        }
        for lang, greeting in languages.items():
            self.greetings_store.append(
                greeter_pb2.HelloResponse(
                    greeting=f"{greeting}!",
                    timestamp=int(time.time())
                )
            )

    def SayHello(self, request, context):
        """一元 RPC 实现"""
        print(f"收到请求: name={request.name}, language={request.language}")

        greetings_map = {
            "zh": f"你好, {request.name}!",
            "en": f"Hello, {request.name}!",
            "ja": f"こんにちは, {request.name}!",
            "ko": f"안녕하세요, {request.name}!",
        }

        greeting = greetings_map.get(request.language,
                                     f"Hello, {request.name}!")

        return greeter_pb2.HelloResponse(
            greeting=greeting,
            timestamp=int(time.time())
        )

    def ListGreetings(self, request, context):
        """分页列表 RPC"""
        page_size = request.page_size or 10
        # 简单分页逻辑
        start = 0
        end = min(start + page_size, len(self.greetings_store))

        return greeter_pb2.ListGreetingsResponse(
            greetings=self.greetings_store[start:end],
            next_page_token=""
        )
```

```python
# server/main.py
import grpc
from concurrent import futures
import greeter_pb2_grpc
from greeter_servicer import GreeterServiceServicer

def serve():
    """启动 gRPC 服务"""
    # 创建线程池，最大 10 个工作线程
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ]
    )

    # 注册服务实现
    greeter_pb2_grpc.add_GreeterServiceServicer_to_server(
        GreeterServiceServicer(), server
    )

    # 绑定端口
    server_address = "[::]:50051"
    server.add_insecure_port(server_address)

    # 启动服务
    server.start()
    print(f"gRPC 服务已启动，监听地址: {server_address}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("正在关闭服务...")
        server.stop(grace=5)  # 5 秒优雅关闭

if __name__ == "__main__":
    serve()
```

## Go gRPC 服务端

### 项目结构

```
project/
├── protos/
│   └── greeter.proto
├── gen/
│   ├── greeter.pb.go
│   └── greeter_grpc.pb.go
├── server/
│   └── main.go
└── go.mod
```

```go
// server/main.go
package main

import (
    "context"
    "fmt"
    "log"
    "net"
    "sync"
    "time"

    pb "example.com/project/gen"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// GreeterServer 实现 GreeterService
type GreeterServer struct {
    pb.UnimplementedGreeterServiceServer
    mu              sync.RWMutex
    greetingsStore  []*pb.HelloResponse
}

// NewGreeterServer 创建服务实例
func NewGreeterServer() *GreeterServer {
    s := &GreeterServer{}
    s.initSampleData()
    return s
}

func (s *GreeterServer) initSampleData() {
    languages := map[string]string{
        "zh": "你好",
        "en": "Hello",
        "ja": "こんにちは",
        "ko": "안녕하세요",
    }
    for _, greeting := range languages {
        s.greetingsStore = append(s.greetingsStore, &pb.HelloResponse{
            Greeting:  fmt.Sprintf("%s!", greeting),
            Timestamp: time.Now().Unix(),
        })
    }
}

// SayHello 实现一元 RPC
func (s *GreeterServer) SayHello(
    ctx context.Context,
    req *pb.HelloRequest,
) (*pb.HelloResponse, error) {
    log.Printf("收到请求: name=%s, language=%s", req.Name, req.Language)

    // 检查上下文是否已取消
    if err := ctx.Err(); err != nil {
        return nil, status.Errorf(codes.Canceled, "请求已取消: %v", err)
    }

    greetings := map[string]string{
        "zh": "你好",
        "en": "Hello",
        "ja": "こんにちは",
        "ko": "안녕하세요",
    }

    prefix, ok := greetings[req.Language]
    if !ok {
        prefix = "Hello"
    }

    return &pb.HelloResponse{
        Greeting:  fmt.Sprintf("%s, %s!", prefix, req.Name),
        Timestamp: time.Now().Unix(),
    }, nil
}

// ListGreetings 实现分页列表
func (s *GreeterServer) ListGreetings(
    ctx context.Context,
    req *pb.ListGreetingsRequest,
) (*pb.ListGreetingsResponse, error) {
    s.mu.RLock()
    defer s.mu.RUnlock()

    pageSize := int(req.PageSize)
    if pageSize <= 0 {
        pageSize = 10
    }

    start := 0
    end := start + pageSize
    if end > len(s.greetingsStore) {
        end = len(s.greetingsStore)
    }

    return &pb.ListGreetingsResponse{
        Greetings: s.greetingsStore[start:end],
    }, nil
}

func main() {
    // 监听 TCP 端口
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("无法监听端口: %v", err)
    }

    // 创建 gRPC 服务器
    grpcServer := grpc.NewServer(
        grpc.MaxRecvMsgSize(50*1024*1024),
        grpc.MaxSendMsgSize(50*1024*1024),
    )

    // 注册服务
    pb.RegisterGreeterServiceServer(grpcServer, NewGreeterServer())

    log.Println("gRPC 服务已启动，监听端口 :50051")

    // 启动服务
    if err := grpcServer.Serve(lis); err != nil {
        log.Fatalf("服务启动失败: %v", err)
    }
}
```

## 拦截器（Interceptor）

拦截器允许在 RPC 方法执行前后插入自定义逻辑，类似于 HTTP 中间件。

### Python 拦截器

```python
# server/interceptors.py
import grpc
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimingInterceptor(grpc.ServerInterceptor):
    """计时拦截器"""

    def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method
        start_time = time.time()

        handler = continuation(handler_call_details)
        if handler is None:
            return None

        # 包装 RPC 处理函数
        if handler.unary_unary:
            original = handler.unary_unary

            def timed_handler(request, context):
                result = original(request, context)
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"[{method_name}] 耗时: {elapsed:.2f}ms")
                return result

            return grpc.unary_unary_rpc_method_handler(
                timed_handler,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        return handler


class AuthInterceptor(grpc.ServerInterceptor):
    """认证拦截器"""

    def __init__(self, valid_tokens):
        self.valid_tokens = valid_tokens

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        token = metadata.get("authorization", "")

        if token not in self.valid_tokens:
            # 返回拒绝处理的 handler
            def reject(request, context):
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "无效的认证令牌"
                )
            return grpc.unary_unary_rpc_method_handler(reject)

        return continuation(handler_call_details)
```

```python
# 使用拦截器启动服务
import grpc
from concurrent import futures
import greeter_pb2_grpc
from greeter_servicer import GreeterServiceServicer
from interceptors import TimingInterceptor, AuthInterceptor

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[
            TimingInterceptor(),
            AuthInterceptor({"valid-token-1", "valid-token-2"}),
        ]
    )

    greeter_pb2_grpc.add_GreeterServiceServicer_to_server(
        GreeterServiceServicer(), server
    )

    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
```

### Go 拦截器

```go
// server/interceptors.go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/status"
)

// UnaryTimingInterceptor 计时拦截器
func UnaryTimingInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()

    resp, err := handler(ctx, req)

    elapsed := time.Since(start)
    log.Printf("[%s] 耗时: %v", info.FullMethod, elapsed)

    return resp, err
}

// UnaryAuthInterceptor 认证拦截器
func UnaryAuthInterceptor(
    validTokens map[string]bool,
) grpc.UnaryServerInterceptor {
    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (interface{}, error) {
        // 从 metadata 中获取 token
        md, ok := metadata.FromIncomingContext(ctx)
        if !ok {
            return nil, status.Error(codes.Unauthenticated, "缺少 metadata")
        }

        tokens := md.Get("authorization")
        if len(tokens) == 0 {
            return nil, status.Error(codes.Unauthenticated, "缺少认证令牌")
        }

        if !validTokens[tokens[0]] {
            return nil, status.Error(codes.Unauthenticated, "无效的认证令牌")
        }

        return handler(ctx, req)
    }
}
```

```go
// 在 main.go 中使用拦截器
func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("无法监听端口: %v", err)
    }

    validTokens := map[string]bool{
        "valid-token-1": true,
        "valid-token-2": true,
    }

    grpcServer := grpc.NewServer(
        grpc.UnaryInterceptor(UnaryTimingInterceptor),
        grpc.ChainUnaryInterceptor(
            UnaryAuthInterceptor(validTokens),
            UnaryTimingInterceptor,
        ),
    )

    pb.RegisterGreeterServiceServer(grpcServer, NewGreeterServer())

    log.Println("gRPC 服务已启动，监听端口 :50051")
    if err := grpcServer.Serve(lis); err != nil {
        log.Fatalf("服务启动失败: %v", err)
    }
}
```

## 优雅关闭

```python
# Python 优雅关闭
import signal
import grpc
from concurrent import futures

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 注册服务...
    server.add_insecure_port("[::]:50051")
    server.start()

    def graceful_shutdown(signum, frame):
        print("收到关闭信号，正在优雅关闭...")
        server.stop(grace=10)  # 等待 10 秒完成正在处理的请求

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    server.wait_for_termination()
```

```go
// Go 优雅关闭
import (
    "os"
    "os/signal"
    "syscall"
    "time"

    "google.golang.org/grpc"
)

func main() {
    grpcServer := grpc.NewServer()
    // 注册服务...

    lis, _ := net.Listen("tcp", ":50051")

    go func() {
        log.Println("gRPC 服务已启动")
        grpcServer.Serve(lis)
    }()

    // 等待关闭信号
    quit := make(chan os.Signal, 1)
    signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
    <-quit

    log.Println("正在关闭服务...")
    grpcServer.GracefulStop() // 优雅关闭
    log.Println("服务已关闭")
}
```

## 小结

gRPC 服务端实现需要创建 Servicer 类/结构体来实现 `.proto` 中定义的接口方法。Python 使用 `grpcio` 库和线程池服务器，Go 使用 `google.golang.org/grpc` 包。拦截器是实现横切关注点（认证、日志、监控）的关键机制。优雅关闭确保在服务停止时完成正在处理的请求。
