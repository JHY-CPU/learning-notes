# 74_SSL-TLS_考试

## 核心概念

- **408常考题型**：选择题为主，涉及SSL/TLS作用和位置
- **关键考点**：
  - SSL/TLS在应用层和传输层之间
  - SSL/TLS提供机密性、完整性、认证
  - HTTPS = HTTP + SSL/TLS
  - SSL/TLS使用数字证书验证身份
- **易混淆点**：
  - SSL/TLS不是传输层协议，是安全层
  - SSL已废弃，现在使用TLS

## 原理分析

### 典型考题1：SSL/TLS位置

**题目**：SSL/TLS协议位于（  ）
A. 应用层
B. 传输层
C. 应用层和传输层之间
D. 网络层

**答案**：C

**解析**：
- SSL/TLS在应用层和传输层之间
- 为应用层提供安全服务

### 典型考题2：SSL/TLS功能

**题目**：SSL/TLS提供的功能包括（  ）（多选）
A. 数据加密
B. 数据完整性验证
C. 身份认证
D. 数据压缩

**答案**：A, B, C

**解析**：
- SSL/TLS提供机密性（加密）
- SSL/TLS提供完整性（MAC）
- SSL/TLS提供认证（数字证书）
- SSL/TLS不提供压缩

### 典型考题3：HTTPS

**题目**：HTTPS是（  ）
A. HTTP + TCP
B. HTTP + SSL/TLS
C. HTTP + UDP
D. HTTP + IP

**答案**：B

**解析**：
- HTTPS = HTTP + SSL/TLS
- SSL/TLS提供安全服务

## 直观理解

**做题技巧**：
- SSL/TLS在应用层和传输层之间
- SSL/TLS = 机密性 + 完整性 + 认证
- HTTPS = HTTP + SSL/TLS
- 数字证书验证身份

## 协议关联

- **SSL/TLS与HTTPS**：HTTPS使用SSL/TLS
- **SSL/TLS与TCP**：SSL/TLS在TCP之上
- **408常见组合**：SSL/TLS + HTTPS = 安全综合题
