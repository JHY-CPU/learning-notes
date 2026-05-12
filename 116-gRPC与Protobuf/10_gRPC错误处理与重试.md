# gRPC 错误处理与重试

## gRPC 状态码

gRPC 定义了一组标准状态码，用于表示 RPC 调用的结果。这些状态码定义在 `google.gRPC.Status` 中。

### 常用状态码

| 状态码 | 名称 | 说明 | 是否可重试 |
|--------|------|------|-----------|
| 0 | OK | 成功 | - |
| 1 | CANCELLED | 客户端取消请求 | 否 |
| 2 | UNKNOWN | 未知错误 | 否 |
| 3 | INVALID_ARGUMENT | 请求参数无效 | 否 |
| 4 | DEADLINE_EXCEEDED | 请求超时 | 是 |
| 5 | NOT_FOUND | 资源未找到 | 否 |
| 6 | ALREADY_EXISTS | 资源已存在 | 否 |
| 7 | PERMISSION_DENIED | 权限不足 | 否 |
| 8 | RESOURCE_EXHAUSTED | 资源耗尽（如限流） | 是 |
| 9 | FAILED_PRECONDITION | 前置条件不满足 | 否 |
| 10 | ABORTED | 操作被中止 | 是 |
| 11 | OUT_OF_RANGE | 超出范围 | 否 |
| 12 | UNIMPLEMENTED | 方法未实现 | 否 |
| 13 | INTERNAL | 服务端内部错误 | 是 |
| 14 | UNAVAILABLE | 服务不可用 | 是 |
| 15 | DATA_LOSS | 数据丢失 | 否 |
| 16 | UNAUTHENTICATED | 未认证 | 否 |

## Python 错误处理

### 服务端返回错误

```python
# server/error_examples.py
import grpc
from concurrent import futures
import greeter_pb2
import greeter_pb2_grpc


class GreeterServicer(greeter_pb2_grpc.GreeterServiceServicer):
    def SayHello(self, request, context):
        name = request.name

        # 参数验证
        if not name:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "name 参数不能为空"
            )

        if len(name) > 100:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "name 参数长度不能超过 100 个字符"
            )

        # 模拟资源未找到
        if name == "不存在":
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"用户 {name} 不存在"
            )

        # 模拟权限不足
        if name == "无权限":
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "您没有权限访问此资源"
            )

        # 模拟服务端错误
        if name == "错误":
            context.abort(
                grpc.StatusCode.INTERNAL,
                "服务器内部错误，请稍后重试"
            )

        # 模拟限流
        if name == "限流":
            context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "请求频率过高，请稍后重试"
            )

        return greeter_pb2.HelloResponse(
            greeting=f"你好, {name}!",
            timestamp=int(time.time())
        )
```

### 客户端处理错误

```python
# client/error_handling.py
import grpc
import greeter_pb2
import greeter_pb2_grpc


def call_with_error_handling(stub, name):
    """带完整错误处理的 RPC 调用"""
    request = greeter_pb2.HelloRequest(name=name)

    try:
        response = stub.SayHello(request, timeout=5)
        return response
    except grpc.RpcError as e:
        # 获取状态码和详情
        code = e.code()
        details = e.details()

        # 根据状态码进行不同处理
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            print(f"参数错误: {details}")
        elif code == grpc.StatusCode.NOT_FOUND:
            print(f"资源未找到: {details}")
        elif code == grpc.StatusCode.PERMISSION_DENIED:
            print(f"权限不足: {details}")
        elif code == grpc.StatusCode.UNAUTHENTICATED:
            print(f"未认证: {details}")
        elif code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            print(f"请求被限流: {details}")
        elif code == grpc.StatusCode.UNAVAILABLE:
            print(f"服务不可用: {details}")
        elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
            print(f"请求超时: {details}")
        elif code == grpc.StatusCode.INTERNAL:
            print(f"服务端内部错误: {details}")
        else:
            print(f"未知错误 [{code}]: {details}")

        # 获取调试信息（trailing metadata）
        trailing = e.trailing_metadata()
        if trailing:
            for key, value in trailing:
                print(f"调试信息 - {key}: {value}")

        return None


def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = greeter_pb2_grpc.GreeterServiceStub(channel)

        # 测试各种错误场景
        for name in ["张三", "", "不存在", "无权限", "错误", "限流"]:
            print(f"\n--- 测试: name='{name}' ---")
            call_with_error_handling(stub, name)


if __name__ == "__main__":
    run()
```

## Go 错误处理

### 服务端返回错误

```go
// server/error_examples.go
package main

import (
    "context"
    "fmt"

    pb "example.com/project/gen"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

func (s *GreeterServer) SayHello(
    ctx context.Context,
    req *pb.HelloRequest,
) (*pb.HelloResponse, error) {
    name := req.Name

    // 参数验证
    if name == "" {
        return nil, status.Error(codes.InvalidArgument, "name 参数不能为空")
    }

    if len(name) > 100 {
        return nil, status.Errorf(
            codes.InvalidArgument,
            "name 参数长度不能超过 100 个字符，当前长度: %d",
            len(name),
        )
    }

    // 模拟资源未找到
    if name == "不存在" {
        return nil, status.Errorf(
            codes.NotFound,
            "用户 %s 不存在",
            name,
        )
    }

    // 使用 WithDetails 添加详细错误信息
    if name == "详细错误" {
        st := status.New(codes.InvalidArgument, "验证失败")
        // 可以附加自定义的错误详情
        return nil, st.Err()
    }

    return &pb.HelloResponse{
        Greeting:  fmt.Sprintf("你好, %s!", name),
        Timestamp: time.Now().Unix(),
    }, nil
}
```

### 客户端处理错误

```go
// client/error_handling.go
package main

import (
    "context"
    "log"
    "time"

    pb "example.com/project/gen"
    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

func callWithErrorHandling(
    client pb.GreeterServiceClient,
    name string,
) (*pb.HelloResponse, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    resp, err := client.SayHello(ctx, &pb.HelloRequest{Name: name})
    if err != nil {
        // 从错误中提取状态
        st, ok := status.FromError(err)
        if !ok {
            log.Printf("无法解析错误状态: %v", err)
            return nil, err
        }

        // 根据状态码处理
        switch st.Code() {
        case codes.InvalidArgument:
            log.Printf("参数错误: %s", st.Message())
        case codes.NotFound:
            log.Printf("资源未找到: %s", st.Message())
        case codes.PermissionDenied:
            log.Printf("权限不足: %s", st.Message())
        case codes.Unauthenticated:
            log.Printf("未认证: %s", st.Message())
        case codes.ResourceExhausted:
            log.Printf("请求被限流: %s", st.Message())
        case codes.Unavailable:
            log.Printf("服务不可用: %s", st.Message())
        case codes.DeadlineExceeded:
            log.Printf("请求超时: %s", st.Message())
        case codes.Internal:
            log.Printf("服务端内部错误: %s", st.Message())
        default:
            log.Printf("未知错误 [%s]: %s", st.Code(), st.Message())
        }

        return nil, err
    }

    return resp, nil
}
```

## 重试策略

### Python 重试实现

```python
# client/retry.py
import grpc
import time
import random
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)
T = TypeVar("T")

# 可重试的状态码
RETRYABLE_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.INTERNAL,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
}


def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
    backoff_multiplier: float = 2.0,
    retryable_codes: set = RETRYABLE_CODES,
):
    """带指数退避的重试装饰器"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            backoff = initial_backoff

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except grpc.RpcError as e:
                    last_exception = e
                    code = e.code()

                    if code not in retryable_codes:
                        logger.error(f"不可重试的错误: {code}")
                        raise

                    if attempt == max_retries:
                        logger.error(f"重试次数耗尽: {max_retries} 次")
                        raise

                    # 添加抖动
                    jitter = random.uniform(0, backoff * 0.1)
                    sleep_time = backoff + jitter

                    logger.warning(
                        f"调用失败 [{code}]，"
                        f"{sleep_time:.2f}s 后重试 "
                        f"({attempt + 1}/{max_retries})"
                    )

                    time.sleep(sleep_time)
                    backoff = min(backoff * backoff_multiplier, max_backoff)

            raise last_exception

        return wrapper
    return decorator


# 使用示例
@retry_with_backoff(max_retries=3, initial_backoff=1.0)
def call_greeter(stub, name):
    request = greeter_pb2.HelloRequest(name=name)
    return stub.SayHello(request, timeout=5)
```

### Go 重试实现

```go
// client/retry.go
package main

import (
    "context"
    "log"
    "math"
    "math/rand"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// RetryConfig 重试配置
type RetryConfig struct {
    MaxRetries      int
    InitialBackoff  time.Duration
    MaxBackoff      time.Duration
    BackoffMultiplier float64
    RetryableCodes  map[codes.Code]bool
}

// DefaultRetryConfig 默认重试配置
func DefaultRetryConfig() *RetryConfig {
    return &RetryConfig{
        MaxRetries:        3,
        InitialBackoff:    time.Second,
        MaxBackoff:        30 * time.Second,
        BackoffMultiplier: 2.0,
        RetryableCodes: map[codes.Code]bool{
            codes.Unavailable:        true,
            codes.DeadlineExceeded:   true,
            codes.Internal:           true,
            codes.ResourceExhausted:  true,
        },
    }
}

// WithRetry 带重试的 RPC 调用
func WithRetry(
    ctx context.Context,
    config *RetryConfig,
    fn func(context.Context) error,
) error {
    var lastErr error
    backoff := config.InitialBackoff

    for attempt := 0; attempt <= config.MaxRetries; attempt++ {
        err := fn(ctx)
        if err == nil {
            return nil
        }
        lastErr = err

        // 检查是否可重试
        st, ok := status.FromError(err)
        if !ok || !config.RetryableCodes[st.Code()] {
            return err
        }

        // 检查是否还有重试次数
        if attempt == config.MaxRetries {
            log.Printf("重试次数耗尽: %d 次", config.MaxRetries)
            return err
        }

        // 计算退避时间
        jitter := time.Duration(rand.Float64() * float64(backoff) * 0.1)
        sleepTime := backoff + jitter

        log.Printf(
            "调用失败 [%s]，%v 后重试 (%d/%d)",
            st.Code(), sleepTime, attempt+1, config.MaxRetries,
        )

        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(sleepTime):
        }

        backoff = time.Duration(
            math.Min(
                float64(backoff)*config.BackoffMultiplier,
                float64(config.MaxBackoff),
            ),
        )
    }

    return lastErr
}

// 使用示例
func callWithRetry(client pb.GreeterServiceClient, name string) error {
    config := DefaultRetryConfig()

    return WithRetry(context.Background(), config, func(ctx context.Context) error {
        ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
        defer cancel()

        _, err := client.SayHello(ctx, &pb.HelloRequest{Name: name})
        return err
    })
}
```

## 健康检查

gRPC 提供了标准的健康检查协议，用于探测服务是否可用。

```protobuf
// 使用标准健康检查服务
// google/grpc/health/v1/health.proto
syntax = "proto3";

package grpc.health.v1;

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
    SERVICE_UNKNOWN = 3;
  }
  ServingStatus status = 1;
}

service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
}
```

```python
# Python 健康检查服务端
import grpc
from grpc_health.v1 import health, health_pb2_grpc

def add_health_check(server):
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # 设置服务状态
    health_servicer.set("greeter.GreeterService",
                        health_pb2.HealthCheckResponse.SERVING)
```

```go
// Go 健康检查
import (
    "google.golang.org/grpc/health"
    healthpb "google.golang.org/grpc/health/grpc_health_v1"
)

func registerHealthCheck(server *grpc.Server) {
    healthServer := health.NewServer()
    healthpb.RegisterHealthServer(server, healthServer)

    // 设置服务状态
    healthServer.SetServingStatus(
        "greeter.GreeterService",
        healthpb.HealthCheckResponse_SERVING,
    )
}
```

## 小结

gRPC 的错误处理基于标准状态码，每个状态码有明确的语义。Python 使用 `context.abort()` 返回错误，Go 使用 `status.Error()` 返回错误。客户端通过捕获 `RpcError` 获取状态码和详情。重试策略应仅针对可重试的状态码（如 UNAVAILABLE、DEADLINE_EXCEEDED），并使用指数退避避免雪崩。健康检查协议提供了标准的服务探活机制。
