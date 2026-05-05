# 43_HTTPS_考试

## 核心概念

- **408常考题型**：选择题为主，涉及HTTPS端口、SSL/TLS作用
- **关键考点**：
  - HTTPS端口443
  - HTTPS = HTTP + SSL/TLS
  - SSL/TLS提供机密性、完整性、认证
  - HTTPS使用混合加密
- **易混淆点**：
  - HTTPS端口443 vs HTTP端口80
  - SSL/TLS位于应用层和传输层之间
  - HTTPS比HTTP慢，但更安全

## 原理分析

### 典型考题1：HTTPS端口

**题目**：HTTPS使用的端口号是（  ）
A. 80
B. 8080
C. 443
D. 21

**答案**：C

**解析**：
- HTTP端口80
- HTTPS端口443
- 8080是备用HTTP端口
- 21是FTP控制端口

### 典型考题2：SSL/TLS作用

**题目**：SSL/TLS协议提供的功能包括（  ）（多选）
A. 数据加密
B. 数据完整性验证
C. 身份认证
D. 数据压缩

**答案**：A, B, C

**解析**：
- SSL/TLS提供机密性（加密）
- SSL/TLS提供完整性（MAC验证）
- SSL/TLS提供认证（数字证书）
- SSL/TLS不提供压缩

### 典型考题3：HTTPS与HTTP

**题目**：关于HTTPS，以下说法正确的是（  ）
A. HTTPS使用UDP协议
B. HTTPS端口号是80
C. HTTPS = HTTP + SSL/TLS
D. HTTPS比HTTP更快

**答案**：C

**解析**：
- HTTPS使用TCP（A错误）
- HTTPS端口443（B错误）
- HTTPS = HTTP + SSL/TLS（C正确）
- HTTPS因加密开销比HTTP慢（D错误）

### 典型考题4：加密方式

**题目**：HTTPS使用混合加密方式，即（  ）
A. 非对称加密传输数据
B. 对称加密交换密钥
C. 非对称加密交换密钥，对称加密传输数据
D. 只使用对称加密

**答案**：C

## 直观理解

**做题技巧**：
- HTTPS = HTTP + SSL/TLS，端口443
- SSL/TLS = 机密性 + 完整性 + 认证
- 混合加密 = 非对称交换密钥 + 对称传输数据
- HTTPS比HTTP慢但安全

**常见错误**：
- 混淆HTTP端口80和HTTPS端口443
- 误认为HTTPS比HTTP快
- 忘记SSL/TLS在应用层和传输层之间

## 协议关联

- **HTTPS与HTTP**：HTTPS是HTTP的安全版本
- **HTTPS与TCP**：HTTPS使用TCP，端口443
- **HTTPS与数字证书**：数字证书验证服务器身份
- **408常见组合**：HTTPS端口 + SSL/TLS功能 + 加密方式 = 安全综合题
