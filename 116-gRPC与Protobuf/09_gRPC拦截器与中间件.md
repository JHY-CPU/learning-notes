# gRPC 拦截器与中间件

## 拦截器概述

拦截器（Interceptor）是 gRPC 中实现横切关注点的机制，类似于 HTTP 中间件。拦截器可以在 RPC 方法执行前后插入自定义逻辑，用于认证、日志记录、指标收集、错误处理等场景。

gRPC 提供了四种拦截器类型：

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| 服务端一元拦截器 | 拦截一元 RPC | 认证、日志、限流 |
| 服务端流拦截器 | 拦截流式 RPC | 流级认证、监控 |
| 客户端一元拦截器 | 拦截客户端调用 | 重试、指标 |
| 客户端流拦截器 | 拦截客户端流调用 | 流重试、追踪 |

## Python 拦截器实现

### 服务端一元拦截器

```python
# server/interceptors.py
import grpc
import time
import logging
import functools
from typing import Callable, Any

logger = logging.getLogger(__name__)


class LoggingInterceptor(grpc.ServerInterceptor):
    """日志拦截器：记录每个 RPC 调用的详细信息"""

    def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method
        metadata = dict(handler_call_details.invocation_metadata or [])

        logger.info(f"RPC 调用开始: {method_name}")
        logger.info(f"元数据: {metadata}")

        start_time = time.time()

        handler = continuation(handler_call_details)
        if handler is None:
            return None

        if handler.unary_unary:
            original = handler.unary_unary

            @functools.wraps(original)
            def logged_handler(request, context):
                try:
                    response = original(request, context)
                    elapsed = (time.time() - start_time) * 1000
                    logger.info(
                        f"RPC 调用完成: {method_name}, "
                        f"耗时: {elapsed:.2f}ms"
                    )
                    return response
                except Exception as e:
                    elapsed = (time.time() - start_time) * 1000
                    logger.error(
                        f"RPC 调用失败: {method_name}, "
                        f"耗时: {elapsed:.2f}ms, 错误: {e}"
                    )
                    raise

            return grpc.unary_unary_rpc_method_handler(
                logged_handler,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        return handler


class AuthInterceptor(grpc.ServerInterceptor):
    """认证拦截器：验证请求中的令牌"""

    def __init__(self, token_validator: Callable[[str], bool]):
        self.token_validator = token_validator
        # 白名单方法：不需要认证
        self.exempt_methods = {"/grpc.health.v1.Health/Check"}

    def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method

        # 跳过白名单方法
        if method_name in self.exempt_methods:
            return continuation(handler_call_details)

        metadata = dict(handler_call_details.invocation_metadata or [])
        token = metadata.get("authorization", "").removeprefix("Bearer ")

        if not token or not self.token_validator(token):
            def reject(request, context):
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "无效或缺失的认证令牌"
                )
            return grpc.unary_unary_rpc_method_handler(reject)

        return continuation(handler_call_details)


class RateLimitInterceptor(grpc.ServerInterceptor):
    """限流拦截器：限制每个客户端的请求频率"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # client_id -> [timestamps]
        self.lock = __import__("threading").Lock()

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        client_id = metadata.get("x-client-id", "unknown")

        with self.lock:
            now = time.time()
            if client_id not in self.requests:
                self.requests[client_id] = []

            # 清理过期记录
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < self.window_seconds
            ]

            if len(self.requests[client_id]) >= self.max_requests:
                def reject(request, context):
                    context.abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        f"请求频率超过限制: {self.max_requests}/{self.window_seconds}s"
                    )
                return grpc.unary_unary_rpc_method_handler(reject)

            self.requests[client_id].append(now)

        return continuation(handler_call_details)
```

### 使用拦截器

```python
# server/main.py
import grpc
from concurrent import futures
import greeter_pb2_grpc
from interceptors import LoggingInterceptor, AuthInterceptor, RateLimitInterceptor


def validate_token(token: str) -> bool:
    """验证令牌"""
    valid_tokens = {"secret-token-1", "secret-token-2"}
    return token in valid_tokens


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[
            LoggingInterceptor(),
            RateLimitInterceptor(max_requests=100, window_seconds=60),
            AuthInterceptor(validate_token),
        ]
    )

    greeter_pb2_grpc.add_GreeterServiceServicer_to_server(
        GreeterServicer(), server
    )

    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()
```

## Go 拦截器实现

### 服务端一元拦截器

```go
// server/interceptors.go
package main

import (
    "context"
    "log"
    "strings"
    "sync"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/metadata"
    "google.golang.org/grpc/peer"
    "google.golang.org/grpc/status"
)

// LoggingInterceptor 日志拦截器
func LoggingInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    start := time.Now()

    // 获取客户端信息
    p, _ := peer.FromContext(ctx)
    clientAddr := "unknown"
    if p != nil {
        clientAddr = p.Addr.String()
    }

    log.Printf("[请求] 方法: %s, 客户端: %s", info.FullMethod, clientAddr)

    resp, err := handler(ctx, req)

    elapsed := time.Since(start)
    if err != nil {
        st, _ := status.FromError(err)
        log.Printf(
            "[失败] 方法: %s, 耗时: %v, 错误: %s",
            info.FullMethod, elapsed, st.Message(),
        )
    } else {
        log.Printf("[完成] 方法: %s, 耗时: %v", info.FullMethod, elapsed)
    }

    return resp, err
}

// AuthInterceptor 认证拦截器
func AuthInterceptor(
    validTokens map[string]bool,
) grpc.UnaryServerInterceptor {
    // 白名单方法
    exemptMethods := map[string]bool{
        "/grpc.health.v1.Health/Check": true,
    }

    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (interface{}, error) {
        // 检查是否在白名单中
        if exemptMethods[info.FullMethod] {
            return handler(ctx, req)
        }

        // 从 metadata 获取 token
        md, ok := metadata.FromIncomingContext(ctx)
        if !ok {
            return nil, status.Error(codes.Unauthenticated, "缺少 metadata")
        }

        authValues := md.Get("authorization")
        if len(authValues) == 0 {
            return nil, status.Error(codes.Unauthenticated, "缺少认证令牌")
        }

        token := strings.TrimPrefix(authValues[0], "Bearer ")
        if !validTokens[token] {
            return nil, status.Error(codes.Unauthenticated, "无效的认证令牌")
        }

        return handler(ctx, req)
    }
}

// RateLimitInterceptor 限流拦截器
func RateLimitInterceptor(
    maxRequests int,
    window time.Duration,
) grpc.UnaryServerInterceptor {
    var mu sync.Mutex
    requests := make(map[string][]time.Time)

    return func(
        ctx context.Context,
        req interface{},
        info *grpc.UnaryServerInfo,
        handler grpc.UnaryHandler,
    ) (interface{}, error) {
        // 获取客户端标识
        md, _ := metadata.FromIncomingContext(ctx)
        clientIDs := md.Get("x-client-id")
        clientID := "unknown"
        if len(clientIDs) > 0 {
            clientID = clientIDs[0]
        }

        mu.Lock()
        now := time.Now()
        if _, ok := requests[clientID]; !ok {
            requests[clientID] = []time.Time{}
        }

        // 清理过期记录
        valid := []time.Time{}
        for _, t := range requests[clientID] {
            if now.Sub(t) < window {
                valid = append(valid, t)
            }
        }
        requests[clientID] = valid

        if len(requests[clientID]) >= maxRequests {
            mu.Unlock()
            return nil, status.Errorf(
                codes.ResourceExhausted,
                "请求频率超过限制: %d/%v",
                maxRequests, window,
            )
        }

        requests[clientID] = append(requests[clientID], now)
        mu.Unlock()

        return handler(ctx, req)
    }
}
```

### 链式拦截器

```go
// 使用 ChainUnaryInterceptor 链接多个拦截器
func main() {
    validTokens := map[string]bool{
        "secret-token-1": true,
        "secret-token-2": true,
    }

    server := grpc.NewServer(
        grpc.ChainUnaryInterceptor(
            LoggingInterceptor,
            RateLimitInterceptor(100, time.Minute),
            AuthInterceptor(validTokens),
        ),
    )

    // 注册服务...
}
```

## 客户端拦截器

### Python 客户端拦截器

```python
# client/interceptors.py
import grpc
import time
import logging

logger = logging.getLogger(__name__)


class ClientAuthInterceptor(grpc.UnaryUnaryClientInterceptor):
    """客户端认证拦截器：自动添加令牌"""

    def __init__(self, token: str):
        self.token = token

    def intercept_unary_unary(
        self, continuation, client_call_details, request
    ):
        # 添加认证头
        metadata = list(client_call_details.metadata or [])
        metadata.append(("authorization", f"Bearer {self.token}"))

        new_details = grpc.ClientCallDetails(
            method=client_call_details.method,
            timeout=client_call_details.timeout,
            metadata=metadata,
            credentials=client_call_details.credentials,
            wait_for_ready=client_call_details.wait_for_ready,
            compression=client_call_details.compression,
        )

        return continuation(new_details, request)


class ClientRetryInterceptor(grpc.UnaryUnaryClientInterceptor):
    """客户端重试拦截器"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def intercept_unary_unary(
        self, continuation, client_call_details, request
    ):
        last_error = None
        for attempt in range(self.max_retries):
            response = continuation(client_call_details, request)
            if isinstance(response, grpc.RpcError):
                last_error = response
                if response.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                ):
                    wait = 2 ** attempt
                    logger.warning(
                        f"调用失败，{wait}s 后重试 ({attempt+1}/{self.max_retries})"
                    )
                    time.sleep(wait)
                    continue
            return response
        raise last_error


# 使用客户端拦截器
def create_channel_with_interceptors():
    channel = grpc.insecure_channel("localhost:50051")
    return grpc.intercept_channel(
        channel,
        ClientAuthInterceptor("secret-token-1"),
        ClientRetryInterceptor(max_retries=3),
    )
```

### Go 客户端拦截器

```go
// client/interceptors.go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/metadata"
)

// ClientAuthInterceptor 客户端认证拦截器
func ClientAuthInterceptor(token string) grpc.UnaryClientInterceptor {
    return func(
        ctx context.Context,
        method string,
        req, reply interface{},
        cc *grpc.ClientConn,
        invoker grpc.UnaryInvoker,
        opts ...grpc.CallOption,
    ) error {
        // 添加认证令牌
        md, _ := metadata.FromOutgoingContext(ctx)
        md = md.Copy()
        md.Set("authorization", "Bearer "+token)
        ctx = metadata.NewOutgoingContext(ctx, md)

        return invoker(ctx, method, req, reply, cc, opts...)
    }
}

// ClientLoggingInterceptor 客户端日志拦截器
func ClientLoggingInterceptor(
    ctx context.Context,
    method string,
    req, reply interface{},
    cc *grpc.ClientConn,
    invoker grpc.UnaryInvoker,
    opts ...grpc.CallOption,
) error {
    start := time.Now()
    log.Printf("[客户端请求] 方法: %s", method)

    err := invoker(ctx, method, req, reply, cc, opts...)

    elapsed := time.Since(start)
    if err != nil {
        log.Printf("[客户端失败] 方法: %s, 耗时: %v, 错误: %v",
            method, elapsed, err)
    } else {
        log.Printf("[客户端完成] 方法: %s, 耗时: %v", method, elapsed)
    }

    return err
}

// 创建带拦截器的连接
func newConnection(address string, token string) (*grpc.ClientConn, error) {
    return grpc.Dial(
        address,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithChainUnaryInterceptor(
            ClientLoggingInterceptor,
            ClientAuthInterceptor(token),
        ),
    )
}
```

## 小结

拦截器是 gRPC 中实现横切关注点的核心机制。服务端拦截器可用于认证、日志、限流等功能，客户端拦截器可用于自动添加令牌、重试、指标收集等。Python 使用 `grpc.ServerInterceptor` 接口，Go 使用函数式拦截器并通过 `ChainUnaryInterceptor` 链接。合理组织拦截器顺序可以构建健壮的 RPC 系统。
