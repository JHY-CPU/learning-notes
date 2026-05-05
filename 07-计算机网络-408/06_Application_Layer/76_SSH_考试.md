# 76_SSH_考试

## 核心概念

- **408常考题型**：选择题为主，涉及SSH端口号和与TELNET对比
- **关键考点**：
  - SSH端口22
  - SSH使用TCP
  - SSH是TELNET的安全替代
  - SSH提供加密、认证、完整性
- **易混淆点**：
  - SSH端口22 vs TELNET端口23
  - SSH使用TCP，不使用UDP

## 原理分析

### 典型考题1：SSH端口

**题目**：SSH使用的端口号是（  ）
A. 21
B. 22
C. 23
D. 80

**答案**：B

**解析**：
- SSH端口：22
- TELNET端口：23
- FTP控制：21
- HTTP：80

### 典型考题2：SSH与TELNET

**题目**：与TELNET相比，SSH的优势是（  ）
A. 使用UDP协议
B. 提供加密传输
C. 使用端口23
D. 不需要认证

**答案**：B

**解析**：
- SSH提供加密传输（B正确）
- SSH使用TCP（A错误）
- SSH端口22（C错误）
- SSH需要认证（D错误）

### 典型考题3：SSH功能

**题目**：SSH提供的功能包括（  ）（多选）
A. 数据加密
B. 身份认证
C. 数据完整性
D. 文件传输

**答案**：A, B, C, D

**解析**：
- SSH提供加密、认证、完整性
- SSH支持SCP/SFTP文件传输

## 直观理解

**做题技巧**：
- SSH = 端口22，安全远程登录
- TELNET = 端口23，不安全
- "SSH 22安全，TELNET 23不安全"
- SSH使用TCP

## 协议关联

- **SSH与TELNET**：SSH是TELNET的安全替代
- **SSH与TCP**：SSH使用TCP，端口22
- **SSH与SSL/TLS**：都提供安全服务
- **408常见组合**：SSH + TELNET = 远程登录对比题
