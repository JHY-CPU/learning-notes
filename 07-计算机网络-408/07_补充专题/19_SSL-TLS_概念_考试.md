# 19_SSL-TLS_概念_考试

## 核心概念

- **408考纲要点**：SSL/TLS的基本概念、握手过程、与HTTPS的关系
- **必背知识点**：
  - SSL已废弃，当前使用TLS（1.2/1.3）
  - TLS在传输层和应用层之间
  - TLS提供机密性、完整性、认证性
  - HTTPS = HTTP + TLS，端口443
- **TLS握手的目的**：
  1. 协商密码套件（加密算法、MAC算法）
  2. 身份认证（服务器证书验证）
  3. 密钥交换（生成共享会话密钥）

## 原理分析

### 典型真题：TLS握手顺序

**题目**：TLS握手过程的正确顺序是（）
①服务器发送证书 ②客户端发送ClientHello ③客户端验证证书 ④密钥交换 ⑤开始加密通信

A. ②①③④⑤ B. ①②③④⑤ C. ②③①④⑤ D. ②①④③⑤

**解析**：答案A。正确顺序：ClientHello→ServerHello+Certificate→客户端验证证书→密钥交换→ChangeCipherSpec→加密通信。

### 典型真题：TLS位置

**题目**：TLS协议工作在OSI模型的哪一层？（）
A. 应用层 B. 传输层 C. 网络层 D. 表示层

**解析**：严格来说TLS在传输层之上、应用层之下（有时称为"表示层"或"4.5层"）。408考试中，通常认为TLS在传输层和应用层之间。

### 典型真题：HTTPS

**题目**：HTTPS的默认端口号是（）
A. 80 B. 443 C. 8080 D. 8443

**解析**：答案B。HTTP默认80，HTTPS默认443。

### 典型真题：TLS vs IPsec

**题目**：以下关于TLS和IPsec的比较，正确的是（）
A. TLS工作在网络层，IPsec工作在传输层
B. TLS保护特定应用，IPsec保护所有IP流量
C. IPsec不需要数字证书
D. TLS不能保护UDP应用

**解析**：答案B。
- A错：TLS在传输层，IPsec在网络层
- B对：TLS基于应用（如HTTPS），IPsec保护所有IP包
- C错：IPsec也可用数字证书认证（IKE）
- D对：传统TLS只保护TCP，但DTLS可保护UDP

## 直观理解

- **答题技巧**：
  - 看到"443/HTTPS"→TLS
  - 看到"握手过程"→协商算法→验证身份→交换密钥→开始加密
  - 看到"传输层安全"→TLS
  - 看到"网络层安全"→IPsec
  - 看到"SSL 3.0"→已不安全
- **TLS版本记忆**：SSL 2.0/3.0不安全→TLS 1.0/1.1过时→TLS 1.2主流→TLS 1.3最新

## 协议关联

- **TLS与TCP**：TLS依赖TCP，先建立TCP连接再TLS握手
- **TLS与HTTP**：HTTPS = HTTP + TLS
- **TLS与证书**：握手时服务器发送证书，客户端验证
- **TLS与密钥交换**：RSA或DH/ECDH
- **408考法**：握手顺序、TLS位置、HTTPS端口、TLS与IPsec对比
