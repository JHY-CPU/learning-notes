# 19_SMTP协议

## 核心概念

- **SMTP（Simple Mail Transfer Protocol）**：简单邮件传输协议
- SMTP使用**TCP**协议，端口号**25**
- SMTP是**C/S模式**的应用协议
- SMTP是"**推（Push）**"协议：将邮件从客户端推送到服务器
- SMTP使用**7位ASCII码**传输邮件内容
- **408考试重点**：SMTP的端口号、工作过程、与HTTP的对比

## 原理分析

### SMTP工作过程

1. **建立连接**：
   - 客户端（发送方邮件服务器）连接服务器（接收方邮件服务器）的25端口
   - 服务器响应：`220 Service ready`

2. **SMTP握手**：
   - `HELO sender.com`：客户端标识自己
   - `MAIL FROM:<alice@sender.com>`：指定发件人
   - `RCPT TO:<bob@receiver.com>`：指定收件人
   - `DATA`：开始传输邮件内容

3. **邮件传输**：
   - 客户端发送邮件首部和正文
   - 以`.`（单独一行）表示结束

4. **关闭连接**：
   - `QUIT`：客户端请求关闭连接
   - 服务器响应：`221 Bye`

### SMTP命令和响应

| 命令 | 说明 |
|------|------|
| HELO | 问候，标识发送方 |
| MAIL FROM | 指定发件人地址 |
| RCPT TO | 指定收件人地址 |
| DATA | 开始传输邮件内容 |
| QUIT | 关闭连接 |

**响应码格式**：三位数字
- 2xx：成功
- 3xx：中间响应
- 4xx：暂时失败
- 5xx：永久失败

### SMTP的特点

- **简单**：协议设计简单，易于实现
- **可靠**：基于TCP，保证传输可靠性
- **局限性**：
  - 只能传输7位ASCII文本
  - 不能传输二进制文件（需要MIME扩展）
  - 邮件长度有限制
  - 服务器之间直接通信，不经过中间存储

## 直观理解

**SMTP就像快递员送信**：
- 你（发件人）把信交给快递员（SMTP）
- 快递员按地址（MX记录）找到收件人的邮局（邮件服务器）
- 快递员直接把信交给对方邮局（推模式）
- 如果对方邮局暂时不开门（4xx），快递员会重试

**记忆技巧**：
- SMTP = "推"（Push），端口25
- "SM25发"（S-Simple, M-Mail, 25端口发邮件）
- SMTP只负责发送，不负责接收
- SMTP用7位ASCII，不能发二进制（需要MIME）

## 代码示例

### 使用 Python smtplib 发送邮件

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(sender, receiver, subject, body, password):
    """通过SMTP发送邮件"""
    # 构造邮件
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # 连接SMTP服务器并发送
    with smtplib.SMTP('smtp.example.com', 25) as server:
        server.ehlo()                           # HELO/EHLO 握手
        server.starttls()                       # 启用TLS加密
        server.login(sender, password)          # 登录认证
        server.sendmail(sender, [receiver], msg.as_string())
        print("邮件发送成功!")

# 使用示例
send_email(
    sender='alice@example.com',
    receiver='bob@example.com',
    subject='测试邮件',
    body='这是一封通过SMTP协议发送的测试邮件。',
    password='your_password'
)
```

### 使用 telnet 手动体验 SMTP 交互过程

```bash
# 连接SMTP服务器（手动输入SMTP命令）
telnet smtp.example.com 25

# 服务器响应: 220 smtp.example.com Service ready

EHLO client.example.com       # 标识自己
MAIL FROM:<alice@example.com> # 指定发件人
RCPT TO:<bob@example.com>     # 指定收件人
DATA                          # 开始传输邮件内容
From: alice@example.com
To: bob@example.com
Subject: Hello

这是一封通过telnet发送的邮件。
.                             # 单独一行的点号表示结束
QUIT                          # 关闭连接
```

### 使用 Python smtplib 直接发送 SMTP 命令

```python
import smtplib

def send_raw_smtp():
    """直接使用SMTP命令发送邮件（理解协议过程）"""
    # 连接 SMTP 服务器的 25 端口
    server = smtplib.SMTP('localhost', 25)
    server.set_debuglevel(1)  # 开启调试，可以看到SMTP交互过程

    # SMTP 握手过程
    server.ehlo('client.example.com')           # HELO/EHLO
    server.sendmail(
        'alice@localhost',
        'bob@localhost',
        'Subject: SMTP测试\r\n'
        '\r\n'
        '直接使用SMTP命令发送邮件。'
    )
    server.quit()                                # QUIT 关闭连接

send_raw_smtp()
```

## 协议关联

- **SMTP与TCP**：SMTP使用TCP的可靠传输，端口25
- **SMTP与DNS**：发送邮件前查MX记录找到目标邮件服务器
- **SMTP与MIME**：MIME扩展SMTP，支持多媒体邮件
- **SMTP与HTTP**：
  - HTTP是拉协议（客户端从服务器获取），SMTP是推协议
  - HTTP每个对象用独立连接，SMTP所有对象用一个连接
  - HTTP用多种格式，SMTP只用7位ASCII
- **408考点**：
  - SMTP端口25
  - SMTP是推协议
  - SMTP只能传输ASCII文本
  - SMTP与POP3/IMAP的区别
- **陷阱**：SMTP是服务器间传输协议，不是用户发送邮件的协议（用户用UA通过SMTP发到本地服务器）
