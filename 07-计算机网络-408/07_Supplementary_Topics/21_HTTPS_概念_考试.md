# 21_HTTPS_概念_考试

## 核心概念

- **408考纲要点**：HTTPS的概念、端口号、工作原理
- **必背知识点**：
  - HTTPS = HTTP + TLS/SSL
  - 默认端口443（HTTP为80）
  - HTTPS提供机密性、完整性、认证性
  - HTTPS需要CA签发的数字证书
- **HTTPS连接建立过程**：
  1. DNS解析（获取服务器IP）
  2. TCP三次握手（端口443）
  3. TLS握手（算法协商、证书验证、密钥交换）
  4. 加密的HTTP请求/响应

## 原理分析

### 典型真题：HTTPS端口

**题目**：HTTPS使用的默认端口号是（）
A. 80 B. 8080 C. 443 D. 8443

**解析**：答案C。HTTP=80，HTTPS=443，Tomcat管理=8080。

### 典型真题：HTTPS安全服务

**题目**：HTTPS可以提供以下哪些安全服务？（多选）
A. 数据机密性 B. 数据完整性 C. 服务器身份认证 D. 数据压缩

**解析**：答案ABC。TLS提供机密性（加密）、完整性（MAC）、认证性（证书）。压缩是可选的非安全功能。

### 典型真题：HTTPS建立连接

**题目**：建立HTTPS连接时，首先进行的操作是（）
A. TLS握手 B. TCP握手 C. HTTP请求 D. DNS解析

**解析**：答案D。先DNS解析（获取IP），再TCP握手（建立连接），再TLS握手（安全协商），最后HTTP请求。

### 典型真题：HTTPS与HTTP

**题目**：关于HTTPS的描述，错误的是（）
A. HTTPS使用443端口
B. HTTPS在HTTP和TCP之间增加了TLS层
C. HTTPS比HTTP更安全但性能一定更差
D. HTTPS需要数字证书

**解析**：答案C。现代硬件下HTTPS性能开销很小，且有会话复用等优化，"一定更差"过于绝对。

## 直观理解

- **考试技巧**：
  - 问"端口号"：HTTP=80，HTTPS=443
  - 问"安全服务"：TLS三件套（机密性+完整性+认证）
  - 问"建立顺序"：DNS→TCP→TLS→HTTP
  - 问"证书验证"：TLS握手阶段
- **易错点**：
  - HTTPS的"S"是Secure不是Server
  - HTTPS不能隐藏域名（DNS仍是明文的）
  - HTTPS不能防所有攻击（如XSS、CSRF仍可能发生）

## 协议关联

- **HTTPS = HTTP + TLS + TCP + IP**（完整的协议栈）
- **与DNS**：DNS查询通常是明文的，需要用DNS over HTTPS (DoH)解决
- **与HTTP/2**：HTTP/2虽然不要求HTTPS，但浏览器实现中通常要求HTTPS
- **408常见考法**：HTTPS端口、HTTPS提供哪些安全服务、HTTPS建立过程
