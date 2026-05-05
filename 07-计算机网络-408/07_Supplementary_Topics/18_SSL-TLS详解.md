# 18_SSL-TLS详解

## 核心概念

- **SSL/TLS定义**：在传输层为应用层提供安全通信的协议
  - SSL（Secure Sockets Layer）：由Netscape开发，已废弃（SSL 2.0/3.0不安全）
  - TLS（Transport Layer Security）：SSL的标准化继任者，当前版本TLS 1.2/1.3
- **TLS提供的安全服务**：
  - 机密性（加密通信内容）
  - 完整性（MAC/HMAC验证）
  - 认证性（数字证书验证身份）
- **TLS记录协议（Record Protocol）**：负责数据分片、压缩、加密、MAC
- **TLS握手协议（Handshake Protocol）**：协商算法、交换密钥、身份认证
- **408重点**：TLS握手过程、密钥交换机制

## 原理分析

### TLS 1.2 握手过程（完整版）

```
Client                              Server
  |                                    |
  |  -------- ClientHello -------->   |
  |  (支持的TLS版本、密码套件列表、     |
  |   客户端随机数Client_random)       |
  |                                    |
  |  <------- ServerHello ---------   |
  |  (选定的TLS版本、密码套件、         |
  |   服务器随机数Server_random)       |
  |                                    |
  |  <--- Certificate ------------    |
  |  (服务器数字证书)                  |
  |                                    |
  |  <--- ServerKeyExchange ------    |
  |  (服务器DH参数/RSA公钥)            |
  |                                    |
  |  <--- ServerHelloDone ---------   |
  |                                    |
  |  --- ClientKeyExchange ------->   |
  |  (客户端DH公钥/预主密钥加密)        |
  |                                    |
  |  --- ChangeCipherSpec -------->   |
  |  (切换到协商的加密算法)             |
  |                                    |
  |  --- Finished --------------->    |
  |  (握手消息的MAC)                   |
  |                                    |
  |  <--- ChangeCipherSpec --------   |
  |  <--- Finished ---------------    |
  |                                    |
  |  <=== 加密的应用数据 =========>   |
```

### 密钥派生过程

1. 生成预主密钥（Pre-Master Secret）：
   - RSA方式：客户端生成，用服务器公钥加密后发送
   - DH方式：双方DH交换生成共享密钥
2. 生成主密钥（Master Secret）：
   $$MS = PRF(PMS, "master secret", Client\_random + Server\_random)$$
3. 生成会话密钥：
   $$Key\_Block = PRF(MS, "key expansion", Server\_random + Client\_random)$$
   从中派生：客户端MAC密钥、服务器MAC密钥、客户端加密密钥、服务器加密密钥

### 密码套件（Cipher Suite）示例

`TLS_RSA_WITH_AES_128_CBC_SHA256` 含义：
- 密钥交换：RSA
- 加密算法：AES-128-CBC
- MAC算法：SHA-256

## 直观理解

- **握手类比**：像两个人初次见面
  1. 你说"你好，我会说中文和英文"（ClientHello）
  2. 对方说"你好，我们说中文吧"（ServerHello）
  3. 对方出示身份证（Certificate）
  4. 你验证身份证（验证证书链）
  5. 双方约定一个暗号（密钥交换）
  6. 之后所有对话都用暗号加密（加密通信）
- **记忆口诀**："Hello选算法，Certificate验身份，KeyExchange传密钥，CipherSpec切加密"
- **TLS vs IPsec**：
  - TLS在传输层，保护特定应用（HTTP→HTTPS）
  - IPsec在网络层，保护所有IP包

## 协议关联

- **与HTTPS**：HTTPS = HTTP + TLS，端口443
- **与TCP**：TLS在TCP之上，必须先建立TCP连接
- **与数字证书**：TLS握手需要服务器提供数字证书
- **与TCP的关联**：TLS记录协议对数据分片后交给TCP传输
- **408常见陷阱**：
  - "TLS工作在应用层"——错，TLS在传输层和应用层之间
  - "HTTPS是安全的HTTP协议"——HTTPS是HTTP over TLS
  - "TLS可以不使用证书"——服务器认证时一般需要证书（匿名TLS不安全）
