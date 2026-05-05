# 73_SSL-TLS概述

## 核心概念

- **SSL（Secure Sockets Layer）**：安全套接层
- **TLS（Transport Layer Security）**：传输层安全协议
- SSL 3.0 → TLS 1.0 → TLS 1.2 → TLS 1.3
- SSL/TLS位于**应用层和传输层之间**
- SSL/TLS提供：**机密性、完整性、身份认证**
- **408考试重点**：SSL/TLS的作用、握手过程、与HTTPS的关系

## 原理分析

### SSL/TLS协议栈

| 层次 | 协议 | 作用 |
|------|------|------|
| 应用层 | HTTP、FTP、SMTP | 应用数据 |
| 安全层 | SSL/TLS | 加密、认证 |
| 传输层 | TCP | 可靠传输 |
| 网络层 | IP | 路由 |

### SSL/TLS握手过程

```
Client                          Server
  |------- ClientHello --------->|
  |  (支持的算法、随机数)          |
  |                               |
  |<------ ServerHello ----------|
  |  (选择的算法、随机数)          |
  |<------ Certificate ----------|
  |  (服务器证书)                 |
  |<------ ServerHelloDone ------|
  |                               |
  |------- ClientKeyExchange --->|
  |  (预主密钥，用服务器公钥加密)    |
  |------- ChangeCipherSpec ---->|
  |------- Finished ------------>|
  |                               |
  |<------ ChangeCipherSpec -----|
  |<------ Finished -------------|
```

### 密钥交换过程

1. **ClientHello**：客户端发送支持的算法列表和随机数
2. **ServerHello**：服务器选择算法，发送随机数
3. **Certificate**：服务器发送数字证书
4. **ClientKeyExchange**：客户端生成预主密钥，用服务器公钥加密
5. **双方计算会话密钥**：使用两个随机数和预主密钥
6. **ChangeCipherSpec**：切换到加密通信
7. **Finished**：握手完成验证

### SSL/TLS记录协议

- **作用**：加密应用数据
- **过程**：
  1. 分片：将数据分成小块
  2. 压缩：可选压缩
  3. 计算MAC：完整性验证
  4. 加密：使用会话密钥加密
  5. 添加头部：添加SSL/TLS记录头部

### 数字证书

- **CA（Certificate Authority）**：证书颁发机构
- **证书内容**：
  - 服务器公钥
  - 服务器域名
  - CA签名
  - 有效期
- **验证**：客户端使用CA公钥验证证书

## 直观理解

**SSL/TLS握手就像秘密接头**：
1. 你："我有这些暗号支持"（ClientHello）
2. 对方："我们用这个暗号"（ServerHello）
3. 对方："这是我的身份证"（Certificate）
4. 你："验证通过，这是密钥"（ClientKeyExchange）
5. 双方："开始用密语交流"（ChangeCipherSpec）

**记忆技巧**：
- SSL/TLS = "应用层和传输层之间的安全层"
- SSL/TLS提供：机密性 + 完整性 + 认证
- HTTPS = HTTP + SSL/TLS
- 握手 = 协商算法 + 交换密钥 + 验证证书

## 协议关联

- **SSL/TLS与HTTPS**：HTTPS = HTTP + SSL/TLS
- **SSL/TLS与TCP**：SSL/TLS在TCP之上
- **SSL/TLS与数字证书**：数字证书验证服务器身份
- **408考点**：SSL/TLS的作用、位置、与HTTPS的关系
