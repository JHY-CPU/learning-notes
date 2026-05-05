# 68_SIP与VoIP

## 核心概念

- **SIP（Session Initiation Protocol）**：会话发起协议
  - 用于建立、修改和终止多媒体会话
  - 使用**UDP/TCP**，端口**5060**（明文）/5061（TLS）
  - 类似HTTP的文本协议
- **VoIP（Voice over IP）**：基于IP的语音传输
  - 使用RTP传输实时语音/视频
  - 使用SIP或H.323建立会话
- **408考试重点**：SIP的基本概念、VoIP的协议栈

## 原理分析

### SIP工作过程

1. **注册**：
   - 用户代理向SIP注册服务器注册
   - 注册用户的位置信息

2. **邀请**：
   - 主叫方发送INVITE请求
   - 包含会话描述（SDP）

3. **响应**：
   - 被叫方回复180 Ringing（振铃）
   - 被叫方回复200 OK（接听）

4. **确认**：
   - 主叫方发送ACK确认

5. **媒体传输**：
   - 使用RTP传输媒体流
   - 使用RTCP控制媒体质量

6. **结束**：
   - 任一方发送BYE
   - 对方回复200 OK

### VoIP协议栈

| 层次 | 协议 | 作用 |
|------|------|------|
| 应用层 | SIP/H.323 | 会话控制 |
| 应用层 | SDP | 会话描述 |
| 应用层 | RTP | 媒体传输 |
| 应用层 | RTCP | 传输控制 |
| 传输层 | UDP | 传输 |
| 网络层 | IP | 路由 |

### SIP消息格式

**请求示例**：
```
INVITE sip:bob@example.com SIP/2.0
Via: SIP/2.0/UDP pc.example.com
From: Alice <sip:alice@example.com>
To: Bob <sip:bob@example.com>
Call-ID: 1234@pc.example.com
CSeq: 1 INVITE
Content-Type: application/sdp
```

**响应示例**：
```
SIP/2.0 200 OK
Via: SIP/2.0/UDP pc.example.com
From: Alice <sip:alice@example.com>
To: Bob <sip:bob@example.com>
```

## 直观理解

**SIP就像打电话的信号系统**：
- 注册 = 告诉电话公司你的号码
- INVITE = 拨号
- 180 Ringing = 对方电话响了
- 200 OK = 对方接听
- RTP = 实际通话内容
- BYE = 挂断

## 协议关联

- **SIP与UDP/TCP**：SIP可以使用UDP或TCP
- **SIP与RTP**：SIP建立会话，RTP传输媒体
- **SIP与SDP**：SDP描述会话参数
- **408考点**：SIP的基本概念、VoIP协议栈
