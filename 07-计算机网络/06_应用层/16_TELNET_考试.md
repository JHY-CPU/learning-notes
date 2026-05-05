# 17_TELNET_考试

## 核心概念

- **408常考题型**：选择题为主，涉及TELNET端口号、NVT概念、与SSH对比
- **关键考点**：
  - TELNET使用TCP，端口号23
  - NVT（网络虚拟终端）的作用
  - TELNET是明文传输，不安全
  - SSH是TELNET的安全替代品
- **易混淆点**：
  - TELNET端口23 vs SSH端口22
  - NVT不是一种物理终端，而是一种标准格式
  - TELNET使用TCP而非UDP

## 原理分析

### 典型考题1：TELNET基本信息

**题目**：下列关于TELNET的说法正确的是（  ）
A. TELNET使用UDP协议
B. TELNET端口号是22
C. TELNET使用明文传输
D. TELNET是安全的远程登录协议

**答案**：C

**解析**：
- A错误：TELNET使用TCP
- B错误：TELNET端口23，SSH端口22
- C正确：TELNET明文传输，不安全
- D错误：TELNET不安全，SSH才是安全的

### 典型考题2：NVT的作用

**题目**：TELNET中NVT的作用是（  ）
A. 加密传输数据
B. 提供文件传输功能
C. 解决不同终端的兼容性问题
D. 管理用户认证

**答案**：C

**解析**：
- NVT（网络虚拟终端）定义了统一的终端标准格式
- 不同系统的终端类型和控制码不同，NVT作为"通用语言"解决兼容性问题
- 客户端和服务器各自将本地格式转换为NVT格式

### 典型考题3：TELNET与SSH对比

**题目**：与TELNET相比，SSH的优势是（  ）
A. 使用UDP协议，更快
B. 提供加密传输，更安全
C. 使用端口23
D. 不需要认证

**答案**：B

**解析**：
- SSH提供加密、认证、完整性保护
- TELNET明文传输密码，容易被截获
- SSH使用端口22，TELNET使用端口23

## 直观理解

**做题技巧**：
- 看到"TELNET"立即想到"端口23、TCP、明文"
- 看到"远程登录+安全"，答案是SSH
- 看到"NVT"，答案是"解决终端兼容性"
- TELNET和SSH的端口号经常出选择题

**记忆口诀**：
- "TELNET 23不安全，SSH 22有保障"
- NVT = "网络虚拟终端" = "通用翻译"

## 协议关联

- **TELNET与TCP**：TELNET依赖TCP的可靠传输，端口23
- **TELNET与SSH**：SSH（端口22）是TELNET（端口23）的安全替代品
- **TELNET与FTP**：FTP控制连接使用NVT ASCII格式
- **408常见陷阱**：
  - TELNET端口不是22（那是SSH）
  - TELNET不安全，SSH安全
  - TELNET使用TCP，不使用UDP
