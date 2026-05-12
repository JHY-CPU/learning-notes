# Protobuf 代码生成

## protoc 编译器

`protoc` 是 Protocol Buffers 的官方编译器，负责将 `.proto` 文件编译成目标语言的代码。

### 安装 protoc

```bash
# macOS
brew install protobuf

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y protobuf-compiler

# CentOS / RHEL
sudo yum install -y protobuf-compiler

# Windows - 从 GitHub Releases 下载
# https://github.com/protocolbuffers/protobuf/releases

# 验证安装
protoc --version
# libprotoc 25.1
```

### 基本用法

```bash
# 基本语法
protoc [选项] .proto 文件路径

# 常用选项
# -I 或 --proto_path：指定 .proto 文件的搜索路径
# --<lang>_out：指定目标语言和输出目录

# 示例目录结构
# project/
# ├── protos/
# │   ├── common.proto
# │   └── user.proto
# └── gen/
#     ├── python/
#     └── go/

# 生成 Python 代码
protoc -I=protos --python_out=gen/python protos/user.proto

# 生成 Go 代码
protoc -I=protos --go_out=gen/go --go-grpc_out=gen/go protos/user.proto

# 同时生成多种语言
protoc -I=protos \
  --python_out=gen/python \
  --go_out=gen/go \
  --go-grpc_out=gen/go \
  protos/user.proto
```

## Python 代码生成

### 安装 Python 工具

```bash
# 安装 grpcio-tools（包含 protoc 和 Python 插件）
pip install grpcio grpcio-tools protobuf

# 验证安装
python -m grpc_tools.protoc --version
```

### 生成 Python 代码

```bash
# 使用 grpc_tools 生成（推荐，不需要系统安装 protoc）
python -m grpc_tools.protoc \
  -I./protos \
  --python_out=./gen \
  --grpc_python_out=./gen \
  ./protos/user.proto

# 生成的文件：
# gen/user_pb2.py        - 消息类
# gen/user_pb2_grpc.py   - gRPC 存根和处理器
```

```protobuf
// protos/user.proto
syntax = "proto3";

package user;

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
}

message GetUserRequest {
  string user_id = 1;
}

message GetUserResponse {
  User user = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
}

service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
}
```

### 使用生成的 Python 代码

```python
# 使用消息类
from gen import user_pb2

# 创建消息
user = user_pb2.User()
user.id = "u001"
user.name = "张三"
user.email = "zhangsan@example.com"
user.age = 28

# 序列化
data = user.SerializeToString()
print(f"序列化大小: {len(data)} 字节")

# 反序列化
user2 = user_pb2.User()
user2.ParseFromString(data)
print(f"反序列化结果: {user2.name}, {user2.age}")

# 使用关键字参数创建
user3 = user_pb2.User(
    id="u002",
    name="李四",
    email="lisi@example.com",
    age=32
)

# 将消息转换为字典
from google.protobuf.json_format import MessageToDict
user_dict = MessageToDict(user3)
print(user_dict)
# {'id': 'u002', 'name': '李四', 'email': 'lisi@example.com', 'age': 32}

# 从字典创建消息
from google.protobuf.json_format import ParseDict
user4 = ParseDict(
    {"id": "u003", "name": "王五", "email": "wangwu@example.com"},
    user_pb2.User()
)
```

## Go 代码生成

### 安装 Go 插件

```bash
# 安装 protoc-gen-go 和 protoc-gen-go-grpc
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 确保 $GOPATH/bin 在 PATH 中
export PATH="$PATH:$(go env GOPATH)/bin"
```

### 生成 Go 代码

```bash
# 生成 Go 代码
protoc \
  -I./protos \
  --go_out=./gen \
  --go_opt=paths=source_relative \
  --go-grpc_out=./gen \
  --go-grpc_opt=paths=source_relative \
  ./protos/user.proto

# 生成的文件：
# gen/user.pb.go       - 消息结构体
# gen/user_grpc.pb.go  - gRPC 客户端和服务端接口
```

### Go Proto 文件配置

```protobuf
// protos/user.proto - Go 专用配置
syntax = "proto3";

package user;

// 指定 Go 包路径
option go_package = "example.com/project/gen/user";

message User {
  string id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
}
```

### 使用生成的 Go 代码

```go
package main

import (
    "fmt"
    "log"

    pb "example.com/project/gen/user"
    "google.golang.org/protobuf/proto"
    "google.golang.org/protobuf/encoding/protojson"
)

func main() {
    // 创建消息
    user := &pb.User{
        Id:    "u001",
        Name:  "张三",
        Email: "zhangsan@example.com",
        Age:   28,
    }

    // 序列化为二进制
    data, err := proto.Marshal(user)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("二进制大小: %d 字节\n", len(data))

    // 反序列化
    user2 := &pb.User{}
    err = proto.Unmarshal(data, user2)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("反序列化: %s, %d\n", user2.Name, user2.Age)

    // JSON 序列化
    jsonBytes, err := protojson.MarshalOptions{
        Indent:        "  ",
        UseProtoNames: true,
    }.Marshal(user)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("JSON 输出:\n%s\n", string(jsonBytes))

    // JSON 反序列化
    user3 := &pb.User{}
    err = protojson.Unmarshal(jsonBytes, user3)
    if err != nil {
        log.Fatal(err)
    }
}
```

## 描述符文件

描述符文件包含了 `.proto` 文件的元数据信息，可用于运行时反射。

```bash
# 生成描述符文件
protoc \
  -I./protos \
  --descriptor_set_out=./protos/user.desc \
  --include_imports \
  ./protos/user.proto
```

```go
// 在 Go 中加载描述符
package main

import (
    "fmt"
    "os"

    "google.golang.org/protobuf/proto"
    "google.golang.org/protobuf/reflect/protodesc"
    "google.golang.org/protobuf/types/descriptorpb"
)

func main() {
    // 读取描述符文件
    descBytes, err := os.ReadFile("protos/user.desc")
    if err != nil {
        panic(err)
    }

    // 解析描述符
    fileSet := &descriptorpb.FileDescriptorSet{}
    err = proto.Unmarshal(descBytes, fileSet)
    if err != nil {
        panic(err)
    }

    // 创建文件描述符
    files, err := protodesc.NewFiles(fileSet)
    if err != nil {
        panic(err)
    }

    // 查找特定消息的描述符
    fd, err := files.FindDescriptorByName("user.User")
    if err != nil {
        panic(err)
    }
    fmt.Printf("消息描述符: %s\n", fd.FullName())
}
```

## 批量生成脚本

```bash
#!/bin/bash
# generate.sh - 批量生成多语言代码

PROTO_DIR="./protos"
GEN_DIR="./gen"

# 清理旧的生成文件
rm -rf ${GEN_DIR}
mkdir -p ${GEN_DIR}/python
mkdir -p ${GEN_DIR}/go

# 查找所有 .proto 文件
PROTO_FILES=$(find ${PROTO_DIR} -name "*.proto")

# 生成 Python 代码
python -m grpc_tools.protoc \
  -I${PROTO_DIR} \
  --python_out=${GEN_DIR}/python \
  --grpc_python_out=${GEN_DIR}/python \
  ${PROTO_FILES}

# 生成 Go 代码
protoc \
  -I${PROTO_DIR} \
  --go_out=${GEN_DIR}/go \
  --go_opt=paths=source_relative \
  --go-grpc_out=${GEN_DIR}/go \
  --go-grpc_opt=paths=source_relative \
  ${PROTO_FILES}

echo "代码生成完成！"
echo "Python 输出目录: ${GEN_DIR}/python"
echo "Go 输出目录: ${GEN_DIR}/go"
```

## Makefile 集成

```makefile
# Makefile
PROTO_DIR := protos
GEN_DIR := gen
PYTHON_DIR := $(GEN_DIR)/python
GO_DIR := $(GEN_DIR)/go

.PHONY: proto clean proto-python proto-go

proto: proto-python proto-go

proto-python:
	@mkdir -p $(PYTHON_DIR)
	python -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(PYTHON_DIR) \
		--grpc_python_out=$(PYTHON_DIR) \
		$(PROTO_DIR)/*.proto

proto-go:
	@mkdir -p $(GO_DIR)
	protoc \
		-I$(PROTO_DIR) \
		--go_out=$(GO_DIR) \
		--go_opt=paths=source_relative \
		--go-grpc_out=$(GO_DIR) \
		--go-grpc_opt=paths=source_relative \
		$(PROTO_DIR)/*.proto

clean:
	rm -rf $(GEN_DIR)
```

## 小结

protoc 编译器是 Protobuf 生态的核心工具，配合语言特定的插件可以生成高质量的代码。Python 使用 `grpcio-tools` 包更加便捷，Go 需要安装 `protoc-gen-go` 插件。描述符文件提供了运行时反射能力，批量脚本和 Makefile 可以简化代码生成流程。
