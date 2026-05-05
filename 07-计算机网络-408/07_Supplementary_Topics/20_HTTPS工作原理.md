# 20_HTTPS工作原理

## 核心概念

- **HTTPS定义**：HTTP Secure = HTTP over TLS/SSL，默认端口443
- **HTTPS工作流程**：
  1. TCP三次握手（端口443）
  2. TLS握手（协商算法、验证证书、交换密钥）
  3. HTTP请求/响应（在TLS加密通道中传输）
- **HTTP vs HTTPS对比**：

| 特性 | HTTP | HTTPS |
|------|------|-------|
| 端口 | 80 | 443 |
| 加密 | 无 | TLS加密 |
| 证书 | 不需要 | 需要CA签发的证书 |
| 速度 | 较快 | 略慢（握手开销） |
| URL | http:// | https:// |

- **HTTPS的性能优化**：
  - 会话复用（Session Resumption）：跳过完整握手
  - TLS False Start：减少握手往返
  - OCSP Stapling：服务器提供证书状态，减少客户端查询
- **408重点**：HTTPS工作流程、与HTTP的区别

## 原理分析

### HTTPS完整交互流程

```
浏览器                          Web服务器
  |                                |
  |  ---- TCP SYN (443) ----->    |  ① TCP三次握手
  |  <--- TCP SYN+ACK ---------   |
  |  ---- TCP ACK ------------>   |
  |                                |
  |  ---- ClientHello --------->  |  ② TLS握手开始
  |  <--- ServerHello -----------  |
  |  <--- Certificate -----------  |
  |  <--- ServerHelloDone --------  |
  |                                |
  |  ---- ClientKeyExchange --->  |  ③ 密钥交换
  |  ---- ChangeCipherSpec ---->  |  ④ 切换加密
  |  ---- Finished ------------>  |
  |  <--- ChangeCipherSpec ------  |
  |  <--- Finished ---------------  |
  |                                |
  |  ==== 加密的HTTP请求 ======>  |  ⑤ 加密通信
  |  <=== 加密的HTTP响应 --------  |
```

### TLS会话复用

**Session ID方式**：
- 首次握手：服务器生成Session ID，客户端缓存
- 再次连接：ClientHello携带Session ID
- 服务器找到匹配的会话参数，跳过完整握手

**Session Ticket方式**：
- 服务器用密钥加密会话状态，作为Ticket发给客户端
- 再次连接：ClientHello携带Ticket
- 服务器解密Ticket恢复会话状态

**效果**：从2-RTT握手减少到1-RTT

### HTTPS对性能的影响

- **握手延迟**：额外1-2个RTT（首次连接）
- **CPU开销**：加解密消耗CPU
- **内存开销**：维护TLS会话状态
- **实际影响**：现代硬件下，HTTPS性能开销很小（<1%CPU），是值得的

## 直观理解

- **HTTPS类比**：
  - HTTP = 明信片（谁都能看）
  - HTTPS = 加密封条的信（只有收件人能拆开看）
  - TLS握手 = 检查收件人身份证+约定暗号
- **为什么需要HTTPS**：
  - 防窃听：密码、Cookie不被截获
  - 防篡改：运营商不能注入广告
  - 防冒充：证书验证网站身份
- **记忆口诀**："先握手（TCP+TLS），后通信（HTTP加密传）"

## 协议关联

- **HTTPS与HTTP**：HTTPS封装HTTP，HTTP数据在TLS加密通道中传输
- **HTTPS与TCP**：必须先TCP握手（端口443），再TLS握手
- **HTTPS与DNS**：DNS解析仍为明文（可通过DNS over HTTPS解决）
- **HTTPS与证书**：证书验证是TLS握手的核心环节
- **408常见考法**：HTTPS端口号、HTTPS工作流程、HTTPS与HTTP的区别
