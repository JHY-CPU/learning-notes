# Protobuf 语法与数据类型

## Proto3 基本语法

Proto3 是 Protocol Buffers 的第三个主要版本，语法更加简洁。每个 `.proto` 文件以 `syntax` 声明开头。

```protobuf
// 指定使用 proto3 语法
syntax = "proto3";

// 可选的包声明，防止命名冲突
package tutorial;

// 可选：指定生成代码的 Go 包路径
option go_package = "example.com/project/tutorial";

// 导入其他 .proto 文件
import "google/protobuf/timestamp.proto";
```

## 标量数据类型

### 完整类型映射表

| Protobuf 类型 | Go 类型 | Python 类型 | Java 类型 | 说明 |
|---------------|---------|-------------|-----------|------|
| `double` | float64 | float | double | 64 位浮点数 |
| `float` | float32 | float | float | 32 位浮点数 |
| `int32` | int32 | int | int | 32 位有符号整数 |
| `int64` | int64 | int | long | 64 位有符号整数 |
| `uint32` | uint32 | int | int | 32 位无符号整数 |
| `uint64` | uint64 | int | long | 64 位无符号整数 |
| `sint32` | int32 | int | int | 32 位有符号整数（ZigZag） |
| `sint64` | int64 | int | long | 64 位有符号整数（ZigZag） |
| `fixed32` | uint32 | int | int | 32 位定长无符号 |
| `fixed64` | uint64 | int | long | 64 位定长无符号 |
| `sfixed32` | int32 | int | int | 32 位定长有符号 |
| `sfixed64` | int64 | int | long | 64 位定长有符号 |
| `bool` | bool | bool | boolean | 布尔值 |
| `string` | string | str | String | UTF-8 字符串 |
| `bytes` | []byte | bytes | ByteString | 字节序列 |

### 整数类型选择指南

```protobuf
// 一般情况使用 int32/int64
message NormalNumbers {
  int32 count = 1;
  int64 big_count = 2;
}

// 负数较多时使用 sint32/sint64（ZigZag 编码更高效）
message SignedNumbers {
  sint32 temperature = -20;  // 负数场景
  sint64 delta = -1000;
}

// 已知为正数且需要固定大小时使用 fixed 类型
message FixedNumbers {
  fixed32 id = 1;    // 适合 ID、计数器
  fixed64 timestamp = 2;
}
```

## 枚举类型（Enum）

```protobuf
// 枚举定义
enum Status {
  STATUS_UNSPECIFIED = 0;  // proto3 要求第一个值必须是 0
  STATUS_ACTIVE = 1;
  STATUS_INACTIVE = 2;
  STATUS_DELETED = 3;
}

// 使用枚举
message User {
  string name = 1;
  Status status = 2;
  Role role = 3;
}

// 枚举可以定义在消息内部（作用域限定）
message Order {
  enum State {
    STATE_UNSPECIFIED = 0;
    STATE_PENDING = 1;
    STATE_PROCESSING = 2;
    STATE_COMPLETED = 3;
    STATE_CANCELLED = 4;
  }

  string order_id = 1;
  State state = 2;
  double total_amount = 3;
}

// 允许别名：不同的枚举值对应相同的数字
enum Color {
  option allow_alias = true;
  COLOR_UNSPECIFIED = 0;
  COLOR_RED = 1;
  COLOR_CRIMSON = 1;  // 别名，与 RED 相同
  COLOR_GREEN = 2;
  COLOR_BLUE = 3;
}
```

## 嵌套消息

```protobuf
// 消息可以嵌套定义
message Address {
  string street = 1;
  string city = 2;
  string country = 3;
  int32 zip_code = 4;
}

message Person {
  string name = 1;
  int32 age = 2;

  // 使用嵌套消息
  Address home_address = 3;
  Address work_address = 4;

  // 嵌套定义的消息
  message PhoneNumber {
    string number = 1;
    PhoneType type = 2;
  }

  enum PhoneType {
    PHONE_UNSPECIFIED = 0;
    PHONE_MOBILE = 1;
    PHONE_HOME = 2;
    PHONE_WORK = 3;
  }

  repeated PhoneNumber phones = 5;
}

// 引用嵌套消息类型时使用点号
// Person.PhoneNumber
// Person.PhoneType
```

## repeated 字段

`repeated` 表示该字段可以有零个或多个值，类似于数组或列表。

```protobuf
message Team {
  string name = 1;

  // 基本类型的列表
  repeated string member_names = 2;

  // 消息类型的列表
  repeated Person members = 3;

  // 数字列表
  repeated int32 scores = 4;

  // 默认情况下，repeated 字段是打包（packed）的
  // 对于标量数字类型，可以更高效地编码
  repeated int32 packed_numbers = 5 [packed = true];
}
```

## oneof 字段

`oneof` 表示一组字段中同时只能有一个被设置，类似于联合体（union）。

```protobuf
// oneof 示例
message SearchResult {
  string query = 1;

  oneof result {
    string article = 2;
    string video = 3;
    string image = 4;
  }
}

// 在 Go 中使用 oneof
// result := &SearchResult{Query: "hello"}
// result.Result = &SearchResult_Article{Article: "article content"}
// result.Result = &SearchResult_Video{Video: "video url"}

// 更实用的示例：支付方式
message Payment {
  string order_id = 1;
  double amount = 2;

  oneof payment_method {
    CreditCard credit_card = 3;
    BankTransfer bank_transfer = 4;
    Alipay alipay = 5;
    WechatPay wechat_pay = 6;
  }
}

message CreditCard {
  string card_number = 1;
  string expiry_date = 2;
  string cvv = 3;
}

message BankTransfer {
  string bank_name = 1;
  string account_number = 2;
}

message Alipay {
  string account = 1;
}

message WechatPay {
  string open_id = 1;
}
```

## map 类型

`map` 表示键值对映射，键可以是整数或字符串类型，值可以是任意类型（除 map 外）。

```protobuf
// map 类型定义
message Config {
  // 字符串到字符串的映射
  map<string, string> settings = 1;

  // 字符串到整数的映射
  map<string, int32> counters = 2;

  // 字符串到消息的映射
  map<string, Person> contacts = 3;

  // 整数到字符串的映射
  map<int32, string> error_messages = 4;
}

// 实际使用示例
message ApiGateway {
  string name = 1;
  // 路由表：路径 -> 后端服务地址
  map<string, string> routes = 2;
  // 限流配置：路径 -> 每秒请求数
  map<string, int32> rate_limits = 3;
}
```

## 默认值

Proto3 中所有字段都有默认值：

```protobuf
// 各类型的默认值
message DefaultValues {
  // 数字类型：0
  int32 int_val = 1;        // 默认 0
  double double_val = 2;    // 默认 0.0
  float float_val = 3;      // 默认 0.0

  // 布尔：false
  bool bool_val = 4;        // 默认 false

  // 字符串：空字符串
  string string_val = 5;    // 默认 ""

  // 字节：空字节数组
  bytes bytes_val = 6;      // 默认 []

  // 枚举：第一个值（必须是 0）
  // enum Status { UNKNOWN = 0; ACTIVE = 1; }
  // Status status = 7;     // 默认 UNKNOWN

  // 消息：nil（未设置状态）
  // SubMessage msg = 8;    // 默认 nil

  // repeated 和 map：空列表/空映射
  repeated string items = 9;  // 默认 []
  map<string, int32> data = 10;  // 默认 {}
}
```

## 保留字段

```protobuf
message User {
  string name = 1;

  // 保留已删除的字段编号，防止将来误用
  reserved 2, 15, 9 to 11;

  // 保留已删除的字段名
  reserved "old_field", "deprecated_name";

  string email = 3;
  int32 age = 4;
}
```

## any 类型

```protobuf
import "google/protobuf/any.proto";

// Any 可以包含任意消息类型
message Wrapper {
  string id = 1;
  google.protobuf.Any payload = 2;
}

// Go 中使用 Any
// import "google.golang.org/protobuf/types/known/anypb"
// anyPayload, _ := anypb.New(someMessage)
// wrapper := &Wrapper{Id: "123", Payload: anyPayload}
```

## 小结

Proto3 语法简洁明了，支持标量类型、枚举、嵌套消息、oneof、map 等丰富的数据结构。合理选择数据类型（如 sint32 vs int32）可以优化编码效率，使用 reserved 可以安全地演进 Schema。在下一章中，我们将学习如何使用 protoc 编译器生成代码。
