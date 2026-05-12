# gRPC 安全

## 安全概述

gRPC 提供了多种安全机制来保护通信，包括传输层加密（TLS/SSL）、双向 TLS 认证、基于令牌的身份认证和访问控制。在生产环境中，启用安全机制是基本要求。

## TLS/SSL 加密

### 自签名证书生成

```bash
#!/bin/bash
# generate_certs.sh - 生成测试用证书

# 创建证书目录
mkdir -p certs
cd certs

# 1. 生成 CA 私钥和证书
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 3650 -key ca.key \
  -out ca.crt \
  -subj "/CN=MyCA"

# 2. 生成服务端私钥和证书签名请求（CSR）
openssl genrsa -out server.key 4096
openssl req -new -key server.key \
  -out server.csr \
  -subj "/CN=localhost"

# 创建扩展文件（SAN）
cat > server.ext << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.example.com
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# 3. 使用 CA 签署服务端证书
openssl x509 -req -days 365 \
  -in server.csr -CA ca.crt -CAkey ca.key \
  -set_serial 01 -out server.crt \
  -extfile server.ext

# 4. 生成客户端私钥和证书
openssl genrsa -out client.key 4096
openssl req -new -key client.key \
  -out client.csr \
  -subj "/CN=client1"

openssl x509 -req -days 365 \
  -in client.csr -CA ca.crt -CAkey ca.key \
  -set_serial 02 -out client.crt

# 清理临时文件
rm -f server.csr client.csr server.ext

echo "证书生成完成！"
echo "  ca.crt     - CA 证书"
echo "  ca.key     - CA 私钥"
echo "  server.crt - 服务端证书"
echo "  server.key - 服务端私钥"
echo "  client.crt - 客户端证书"
echo "  client.key - 客户端私钥"
```

### 单向 TLS（服务端认证）

客户端验证服务端身份，是最常见的 TLS 使用方式。

**Python 实现：**

```python
# server/tls_server.py
import grpc
from concurrent import futures
import greeter_pb2_grpc


def serve_with_tls():
    # 加载服务端证书和私钥
    with open("certs/server.crt", "rb") as f:
        server_cert = f.read()
    with open("certs/server.key", "rb") as f:
        server_key = f.read()

    # 创建服务端凭证
    server_credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)]
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    greeter_pb2_grpc.add_GreeterServiceServicer_to_server(
        GreeterServicer(), server
    )

    # 使用 TLS 监听
    server.add_secure_port("[::]:50051", server_credentials)
    server.start()
    print("安全 gRPC 服务已启动")
    server.wait_for_termination()
```

```python
# client/tls_client.py
import grpc
import greeter_pb2
import greeter_pb2_grpc


def call_with_tls():
    # 加载 CA 证书
    with open("certs/ca.crt", "rb") as f:
        ca_cert = f.read()

    # 创建 SSL 凭证
    credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
    )

    # 使用安全通道
    with grpc.secure_channel("localhost:50051", credentials) as channel:
        stub = greeter_pb2_grpc.GreeterServiceStub(channel)
        response = stub.SayHello(
            greeter_pb2.HelloRequest(name="安全用户"),
            timeout=5
        )
        print(f"响应: {response.greeting}")
```

**Go 实现：**

```go
// server/tls_server.go
package main

import (
    "crypto/tls"
    "log"
    "net"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials"
)

func serveWithTLS() {
    // 加载服务端证书
    cert, err := tls.LoadX509KeyPair("certs/server.crt", "certs/server.key")
    if err != nil {
        log.Fatalf("无法加载证书: %v", err)
    }

    // 创建 TLS 凭证
    creds := credentials.NewTLS(&tls.Config{
        Certificates: []tls.Certificate{cert},
    })

    // 创建 gRPC 服务器
    server := grpc.NewServer(grpc.Creds(creds))

    // 注册服务...

    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("无法监听: %v", err)
    }

    log.Println("安全 gRPC 服务已启动")
    server.Serve(lis)
}
```

```go
// client/tls_client.go
package main

import (
    "crypto/tls"
    "crypto/x509"
    "log"
    "os"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials"
)

func dialWithTLS() (*grpc.ClientConn, error) {
    // 加载 CA 证书
    caCert, err := os.ReadFile("certs/ca.crt")
    if err != nil {
        return nil, err
    }

    certPool := x509.NewCertPool()
    certPool.AppendCertsFromPEM(caCert)

    // 创建 TLS 凭证
    creds := credentials.NewTLS(&tls.Config{
        RootCAs: certPool,
    })

    return grpc.Dial("localhost:50051", grpc.WithTransportCredentials(creds))
}
```

### 双向 TLS（Mutual TLS）

客户端和服务端互相验证身份，提供更强的安全保障。

**Python 实现：**

```python
# server/mtls_server.py
def serve_with_mtls():
    # 加载服务端证书和私钥
    with open("certs/server.crt", "rb") as f:
        server_cert = f.read()
    with open("certs/server.key", "rb") as f:
        server_key = f.read()

    # 加载 CA 证书（用于验证客户端）
    with open("certs/ca.crt", "rb") as f:
        ca_cert = f.read()

    # 创建双向 TLS 凭证
    server_credentials = grpc.ssl_server_credentials(
        private_key_certificate_chain_pairs=[(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=True,  # 要求客户端认证
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    greeter_pb2_grpc.add_GreeterServiceServicer_to_server(
        GreeterServicer(), server
    )
    server.add_secure_port("[::]:50051", server_credentials)
    server.start()
    server.wait_for_termination()
```

```python
# client/mtls_client.py
def call_with_mtls():
    # 加载 CA 证书
    with open("certs/ca.crt", "rb") as f:
        ca_cert = f.read()

    # 加载客户端证书和私钥
    with open("certs/client.crt", "rb") as f:
        client_cert = f.read()
    with open("certs/client.key", "rb") as f:
        client_key = f.read()

    # 创建双向 TLS 凭证
    credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )

    with grpc.secure_channel("localhost:50051", credentials) as channel:
        stub = greeter_pb2_grpc.GreeterServiceStub(channel)
        response = stub.SayHello(
            greeter_pb2.HelloRequest(name="mTLS 用户"),
            timeout=5
        )
        print(f"响应: {response.greeting}")
```

**Go 实现：**

```go
// Go 双向 TLS
func serveWithMTLS() {
    // 加载服务端证书
    cert, _ := tls.LoadX509KeyPair("certs/server.crt", "certs/server.key")

    // 加载 CA 证书
    caCert, _ := os.ReadFile("certs/ca.crt")
    certPool := x509.NewCertPool()
    certPool.AppendCertsFromPEM(caCert)

    // 配置双向 TLS
    tlsConfig := &tls.Config{
        Certificates: []tls.Certificate{cert},
        ClientCAs:    certPool,
        ClientAuth:   tls.RequireAndVerifyClientCert,
    }

    creds := credentials.NewTLS(tlsConfig)
    server := grpc.NewServer(grpc.Creds(creds))

    lis, _ := net.Listen("tcp", ":50051")
    server.Serve(lis)
}
```

## 基于令牌的认证

### JWT 令牌认证

```python
# server/jwt_auth.py
import grpc
import jwt
import time
from typing import Optional


JWT_SECRET = "your-secret-key"
JWT_ALGORITHM = "HS256"


def generate_token(user_id: str, expires_in: int = 3600) -> str:
    """生成 JWT 令牌"""
    payload = {
        "user_id": user_id,
        "exp": time.time() + expires_in,
        "iat": time.time(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def validate_token(token: str) -> Optional[dict]:
    """验证 JWT 令牌"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


class JWTAuthInterceptor(grpc.ServerInterceptor):
    """JWT 认证拦截器"""

    def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata or [])
        auth_header = metadata.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            def reject(request, context):
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "需要 Bearer 令牌"
                )
            return grpc.unary_unary_rpc_method_handler(reject)

        token = auth_header[7:]
        payload = validate_token(token)

        if payload is None:
            def reject(request, context):
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "无效或过期的令牌"
                )
            return grpc.unary_unary_rpc_method_handler(reject)

        return continuation(handler_call_details)
```

```python
# 客户端使用 JWT
def call_with_jwt():
    token = generate_token("user-001")

    with grpc.insecure_channel("localhost:50051") as channel:
        stub = greeter_pb2_grpc.GreeterServiceStub(channel)

        # 在 metadata 中传递 JWT
        metadata = [("authorization", f"Bearer {token}")]
        response = stub.SayHello(
            greeter_pb2.HelloRequest(name="JWT 用户"),
            metadata=metadata,
            timeout=5
        )
        print(f"响应: {response.greeting}")
```

## RBAC（基于角色的访问控制）

```python
# server/rbac.py
import grpc
from functools import wraps

# 角色权限映射
ROLE_PERMISSIONS = {
    "admin": {"GetUser", "CreateUser", "DeleteUser", "ListUsers"},
    "editor": {"GetUser", "CreateUser", "ListUsers"},
    "viewer": {"GetUser", "ListUsers"},
}


class RBACInterceptor(grpc.ServerInterceptor):
    """RBAC 拦截器"""

    def __init__(self, role_permissions: dict = None):
        self.role_permissions = role_permissions or ROLE_PERMISSIONS

    def intercept_service(self, continuation, handler_call_details):
        method_name = handler_call_details.method.split("/")[-1]
        metadata = dict(handler_call_details.invocation_metadata or [])

        # 从 metadata 获取用户角色
        user_role = metadata.get("x-user-role", "viewer")

        # 检查权限
        allowed_methods = self.role_permissions.get(user_role, set())
        if method_name not in allowed_methods:
            def reject(request, context):
                context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    f"角色 {user_role} 无权访问 {method_name}"
                )
            return grpc.unary_unary_rpc_method_handler(reject)

        return continuation(handler_call_details)
```

## 完整的安全服务示例

```python
# server/secure_server.py
import grpc
from concurrent import futures
import greeter_pb2_grpc


def serve():
    # 加载证书
    with open("certs/server.crt", "rb") as f:
        server_cert = f.read()
    with open("certs/server.key", "rb") as f:
        server_key = f.read()
    with open("certs/ca.crt", "rb") as f:
        ca_cert = f.read()

    # TLS 凭证
    server_credentials = grpc.ssl_server_credentials(
        private_key_certificate_chain_pairs=[(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=True,
    )

    # 创建带拦截器的服务器
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[
            TimingInterceptor(),
            JWTAuthInterceptor(),
            RBACInterceptor(),
        ]
    )

    # 注册服务
    greeter_pb2_grpc.add_GreeterServiceServicer_to_server(
        GreeterServicer(), server
    )

    # 启动安全服务
    server.add_secure_port("[::]:50051", server_credentials)
    server.start()
    print("安全 gRPC 服务已启动（mTLS + JWT + RBAC）")
    server.wait_for_termination()
```

## 小结

gRPC 提供了完善的安全机制。TLS/SSL 加密保护传输安全，双向 TLS 提供客户端和服务端的互相认证，JWT 令牌实现无状态认证，RBAC 实现细粒度的访问控制。在生产环境中，应至少启用单向 TLS，并结合令牌认证和 RBAC 构建完整的安全体系。
