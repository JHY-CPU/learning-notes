# gRPC 在微服务中的应用

## 微服务架构中的 gRPC

在微服务架构中，gRPC 常用于服务之间的内部通信。相比 REST，gRPC 提供了更高的性能、强类型接口和原生流式支持。以下是 gRPC 在微服务中的关键应用场景。

## 服务发现

### 基于 DNS 的服务发现

```python
# 使用 DNS 解析服务地址
import grpc
import socket


def resolve_service(service_name: str, port: int = 50051) -> list:
    """通过 DNS 解析服务地址"""
    try:
        results = socket.getaddrinfo(service_name, port)
        addresses = list(set(f"{r[4][0]}:{port}" for r in results))
        return addresses
    except socket.gaierror:
        return []


def create_channel_with_dns(service_name: str):
    """使用 DNS 解析创建通道"""
    # gRPC 原生支持 DNS 解析
    target = f"dns:///{service_name}:50051"
    channel = grpc.insecure_channel(target)
    return channel
```

### 基于 etcd 的服务发现

```python
# server/etcd_register.py
import etcd3
import json
import time
import threading


class EtcdServiceRegistry:
    """基于 etcd 的服务注册与发现"""

    def __init__(self, etcd_host="localhost", etcd_port=2379):
        self.client = etcd3.client(host=etcd_host, port=etcd_port)
        self.lease = None
        self.keep_alive_thread = None

    def register(
        self,
        service_name: str,
        host: str,
        port: int,
        ttl: int = 10
    ):
        """注册服务"""
        # 创建租约
        self.lease = self.client.lease(ttl=ttl)

        # 注册服务信息
        key = f"/services/{service_name}/{host}:{port}"
        value = json.dumps({
            "host": host,
            "port": port,
            "service": service_name,
            "registered_at": time.time(),
        })

        self.client.put(key, value, lease=self.lease)
        print(f"服务已注册: {key}")

        # 保持租约
        self.keep_alive_thread = threading.Thread(
            target=self.lease.refresh,
            daemon=True
        )
        self.keep_alive_thread.start()

    def discover(self, service_name: str) -> list:
        """发现服务实例"""
        prefix = f"/services/{service_name}/"
        results = self.client.get_prefix(prefix)

        instances = []
        for value, metadata in results:
            if value:
                instance = json.loads(value)
                instances.append(instance)

        return instances

    def deregister(self, service_name: str, host: str, port: int):
        """注销服务"""
        key = f"/services/{service_name}/{host}:{port}"
        self.client.delete(key)
        if self.lease:
            self.lease.revoke()
```

### 基于 Consul 的服务发现

```python
# server/consul_discovery.py
import consul
import grpc


class ConsulServiceDiscovery:
    """基于 Consul 的服务发现"""

    def __init__(self, host="localhost", port=8500):
        self.consul = consul.Consul(host=host, port=port)

    def register(self, service_name, host, port, health_check_interval="10s"):
        """注册服务"""
        self.consul.agent.service.register(
            name=service_name,
            service_id=f"{service_name}-{host}-{port}",
            address=host,
            port=port,
            check=consul.Check.tcp(host, port, health_check_interval),
        )

    def discover(self, service_name):
        """发现服务实例"""
        _, services = self.consul.health.service(service_name, passing=True)
        instances = []
        for service in services:
            address = service["Service"]["Address"]
            port = service["Service"]["Port"]
            instances.append(f"{address}:{port}")
        return instances

    def create_channel(self, service_name):
        """为发现的服务创建通道"""
        instances = self.discover(service_name)
        if not instances:
            raise ValueError(f"未找到服务: {service_name}")

        # 简单轮询选择
        target = instances[0]
        return grpc.insecure_channel(target)
```

## 负载均衡

### 客户端负载均衡

```python
# client/load_balancer.py
import grpc
import random
import threading
from typing import List


class RoundRobinBalancer:
    """轮询负载均衡器"""

    def __init__(self, addresses: List[str]):
        self.addresses = addresses
        self.index = 0
        self.lock = threading.Lock()

    def get_next(self) -> str:
        with self.lock:
            address = self.addresses[self.index]
            self.index = (self.index + 1) % len(self.addresses)
            return address


class LoadBalancedChannel:
    """负载均衡通道"""

    def __init__(self, balancer: RoundRobinBalancer):
        self.balancer = balancer
        self.channels = {}

    def get_channel(self) -> grpc.Channel:
        address = self.balancer.get_next()
        if address not in self.channels:
            self.channels[address] = grpc.insecure_channel(address)
        return self.channels[address]
```

```go
// Go 中的客户端负载均衡
import (
    "google.golang.org/grpc"
    "google.golang.org/grpc/balancer/roundrobin"
    "google.golang.org/grpc/resolver"
    "google.golang.org/grpc/resolver/manual"
)

func createBalancedConn(addresses []string) (*grpc.ClientConn, error) {
    // 使用 round-robin 负载均衡
    return grpc.Dial(
        fmt.Sprintf("%s:///%s", "dns", "my-service"),
        grpc.WithDefaultServiceConfig(`{"loadBalancingPolicy":"round_robin"}`),
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
}
```

## 截止时间传播

在微服务调用链中，截止时间（Deadline）需要在服务之间传播，确保整个调用链不会无限等待。

```python
# client/deadline_propagation.py
import grpc
import time


def call_with_deadline_propagation(stub, request, parent_deadline=None):
    """带截止时间传播的调用"""
    # 计算剩余时间
    if parent_deadline:
        remaining = parent_deadline - time.time()
        timeout = max(remaining - 0.1, 0.01)  # 预留 100ms
    else:
        timeout = 10.0  # 默认 10 秒

    try:
        response = stub.SayHello(request, timeout=timeout)
        return response
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print("调用超时（截止时间传播）")
        raise


class DeadlinePropagationInterceptor(grpc.ServerInterceptor):
    """截止时间传播拦截器"""

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or not handler.unary_unary:
            return handler

        original = handler.unary_unary

        def wrapped(request, context):
            # 将截止时间传递到上下文中
            deadline = context.time_remaining()
            if deadline and deadline < 0.5:
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "上游截止时间已过期"
                )
            return original(request, context)

        return grpc.unary_unary_rpc_method_handler(
            wrapped,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )
```

```go
// Go 中的截止时间传播
import (
    "context"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

// DeadlinePropagationInterceptor 截止时间传播
func DeadlinePropagationInterceptor(
    ctx context.Context,
    req interface{},
    info *grpc.UnaryServerInfo,
    handler grpc.UnaryHandler,
) (interface{}, error) {
    // 检查剩余时间
    if deadline, ok := ctx.Deadline(); ok {
        remaining := time.Until(deadline)
        if remaining <= 0 {
            return nil, status.Error(
                codes.DeadlineExceeded,
                "上游截止时间已过期",
            )
        }
        log.Printf("方法 %s 剩余时间: %v", info.FullMethod, remaining)
    }

    return handler(ctx, req)
}
```

## 链路追踪

### OpenTelemetry 集成

```python
# server/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
import grpc


def setup_tracing(service_name: str):
    """设置链路追踪"""
    # 配置 Tracer Provider
    provider = TracerProvider()

    # 配置 Jaeger 导出器
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
    trace.set_tracer_provider(provider)

    # 自动插桩 gRPC 服务端
    grpc_server_instrumentor = GrpcInstrumentorServer()
    grpc_server_instrumentor.instrument()

    return trace.get_tracer(service_name)


def setup_client_tracing():
    """设置客户端追踪"""
    from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient

    grpc_client_instrumentor = GrpcInstrumentorClient()
    grpc_client_instrumentor.instrument()
```

```go
// Go 中的 OpenTelemetry 集成
package main

import (
    "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/exporters/jaeger"
    "go.opentelemetry.io/otel/sdk/trace"
    "google.golang.org/grpc"
)

func setupTracing() (*trace.TracerProvider, error) {
    // 创建 Jaeger 导出器
    exporter, err := jaeger.New(
        jaeger.WithCollectorEndpoint(
            jaeger.WithEndpoint("http://localhost:14268/api/traces"),
        ),
    )
    if err != nil {
        return nil, err
    }

    // 创建 Tracer Provider
    tp := trace.NewTracerProvider(
        trace.WithBatcher(exporter),
        trace.WithResource(resource.NewWithAttributes(
            semconv.SchemaURL,
            semconv.ServiceNameKey.String("my-service"),
        )),
    )

    otel.SetTracerProvider(tp)
    return tp, nil
}

func main() {
    tp, _ := setupTracing()
    defer tp.Shutdown(nil)

    // 创建带追踪的 gRPC 服务器
    server := grpc.NewServer(
        grpc.StatsHandler(otelgrpc.NewServerHandler()),
    )

    // 创建带追踪的客户端连接
    conn, _ := grpc.Dial(
        "localhost:50051",
        grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
    )
    _ = conn
}
```

## 熔断与限流

```python
# client/circuit_breaker.py
import grpc
import time
import threading
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 熔断
    HALF_OPEN = "half_open"  # 半开


class CircuitBreaker:
    """熔断器"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """通过熔断器调用函数"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise grpc.RpcError(
                        grpc.StatusCode.UNAVAILABLE,
                        "熔断器已打开"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except grpc.RpcError as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


# 使用熔断器
def call_with_circuit_breaker(stub, request):
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
    )
    return breaker.call(stub.SayHello, request, timeout=5)
```

## 微服务网关

### gRPC-Gateway（REST 到 gRPC 代理）

```protobuf
// protos/gateway.proto
syntax = "proto3";

import "google/api/annotations.proto";

package gateway;

message HelloRequest {
  string name = 1;
}

message HelloResponse {
  string greeting = 1;
}

service GreeterService {
  rpc SayHello(HelloRequest) returns (HelloResponse) {
    option (google.api.http) = {
      get: "/api/v1/hello/{name}"
    };
  }

  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse) {
    option (google.api.http) = {
      post: "/api/v1/users"
      body: "*"
    };
  }
}
```

```go
// Go gRPC-Gateway 启动
package main

import (
    "context"
    "log"
    "net"
    "net/http"

    "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    gw "example.com/project/gen/gateway"
)

func main() {
    // 启动 gRPC 服务
    go func() {
        lis, _ := net.Listen("tcp", ":50051")
        grpcServer := grpc.NewServer()
        // 注册服务...
        grpcServer.Serve(lis)
    }()

    // 启动 HTTP 网关
    ctx := context.Background()
    mux := runtime.NewServeMux()
    opts := []grpc.DialOption{
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    }

    err := gw.RegisterGreeterServiceHandlerFromEndpoint(
        ctx, mux, "localhost:50051", opts,
    )
    if err != nil {
        log.Fatal(err)
    }

    log.Println("HTTP 网关启动: :8080")
    http.ListenAndServe(":8080", mux)
}
```

## 小结

gRPC 在微服务架构中扮演核心通信角色。服务发现解决了动态实例定位问题，客户端和服务端负载均衡优化了请求分配，截止时间传播确保了调用链的超时控制，链路追踪提供了分布式调用可视化，熔断器提高了系统容错能力。结合 gRPC-Gateway 等工具，可以构建完整的微服务通信体系。
