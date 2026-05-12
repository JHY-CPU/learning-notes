# 21_IMAP协议

## 核心概念

- **IMAP（Internet Message Access Protocol）**：互联网邮件访问协议
- IMAP使用**TCP**协议，端口号**143**
- IMAP是**C/S模式**的应用协议
- IMAP是"**拉（Pull）**"协议：客户端从服务器获取邮件
- IMAP邮件**保留在服务器**上，支持多设备同步
- IMAP支持**在线和离线**两种操作模式
- **408考试重点**：IMAP端口号、与POP3的区别、服务器保留邮件的特点

## 原理分析

### IMAP工作过程

1. **建立连接**：
   - 客户端连接服务器143端口
   - 服务器响应：`* OK IMAP4 server ready`

2. **登录认证**：
   - `LOGIN username password`：登录
   - 服务器验证：`OK LOGIN completed`

3. **邮件操作**：
   - `LIST`：列出邮箱文件夹
   - `SELECT INBOX`：选择收件箱
   - `FETCH n`：获取第n封邮件
   - `SEARCH`：搜索邮件
   - `COPY`：复制邮件
   - `MOVE`：移动邮件
   - `DELETE`：删除邮件

4. **关闭连接**：
   - `LOGOUT`：退出登录
   - 关闭TCP连接

### IMAP的特点

1. **服务器端存储**：
   - 邮件保留在服务器上
   - 支持多设备访问同一邮箱
   - 邮件状态在所有设备间同步

2. **丰富的邮件管理**：
   - 支持文件夹管理
   - 支持邮件搜索
   - 支持邮件标记（已读、未读、星标等）
   - 支持邮件复制、移动

3. **在线和离线模式**：
   - 在线模式：直接在服务器上操作
   - 离线模式：下载邮件到本地操作
   - 同步：重新连接后同步状态

4. **部分下载**：
   - 可以只下载邮件首部
   - 可以只下载邮件正文
   - 大附件可以按需下载

### POP3 vs IMAP对比

| 特性 | POP3 | IMAP |
|------|------|------|
| 端口号 | 110 | 143 |
| 邮件存储 | 下载后删除 | 服务器保留 |
| 多设备同步 | 不支持 | 支持 |
| 文件夹管理 | 不支持 | 支持 |
| 邮件搜索 | 不支持 | 支持 |
| 协议复杂度 | 简单 | 复杂 |
| 适用场景 | 单设备 | 多设备 |

## 直观理解

**IMAP就像云盘**：
- 你的文件（邮件）存在云端（服务器）
- 你可以在电脑、手机、平板等多个设备访问同一个邮箱
- 在一个设备上删除邮件，其他设备也同步删除
- 所有操作都在云端同步

**POP3 vs IMAP**：
- POP3 = "取信后邮局扔掉原件" → 单设备
- IMAP = "信件存放在邮局，你在任何地方都能看" → 多设备

**记忆技巧**：
- IMAP = "互联网邮件访问协议"，端口143
- "IMAP看143"（从143端口查看邮件）
- IMAP保留服务器副本 = "云端邮件"
- 现代邮箱（Gmail、Outlook）多用IMAP

## 代码示例

### 使用 Python imaplib 收取和管理邮件

```python
import imaplib
from email.parser import Parser

def imap_demo(host, username, password):
    """通过IMAP协议操作邮件（服务器端管理）"""
    # 连接IMAP服务器的143端口
    server = imaplib.IMAP4(host, 143)

    # 登录认证
    server.login(username, password)        # LOGIN username password
    print("登录成功!")

    # LIST - 列出邮箱文件夹
    status, folders = server.list()
    print(f"邮箱文件夹: {folders[0].decode()}")

    # SELECT - 选择收件箱
    status, data = server.select('INBOX')
    print(f"收件箱邮件数: {data[0].decode()}")

    # SEARCH - 搜索邮件（IMAP独有功能）
    status, data = server.search(None, 'ALL')
    email_ids = data[0].split()
    print(f"找到 {len(email_ids)} 封邮件")

    # FETCH - 获取最新3封邮件
    for eid in email_ids[-3:]:
        status, data = server.fetch(eid, '(RFC822)')
        msg = Parser().parsestr(data[0][1].decode('utf-8', errors='ignore'))
        print(f"  主题: {msg.get('Subject', '无')}")

    # 搜索未读邮件（IMAP特色功能）
    status, data = server.search(None, 'UNSEEN')
    unread = data[0].split()
    print(f"未读邮件数: {len(unread)}")

    # LOGOUT - 退出
    server.logout()

# 使用示例
imap_demo('imap.example.com', 'user@example.com', 'password')
```

### 使用 telnet 手动体验 IMAP 交互过程

```bash
# 连接IMAP服务器
telnet imap.example.com 143
# 服务器响应: * OK IMAP4 server ready

# 登录
a001 LOGIN user@example.com password    # a001是命令标签

# 列出文件夹
a002 LIST "" "*"

# 选择收件箱
a003 SELECT INBOX

# 搜索所有邮件
a004 SEARCH ALL

# 获取第1封邮件的首部
a005 FETCH 1 (BODY[HEADER])

# 获取第1封邮件的正文
a006 FETCH 1 (BODY[TEXT])

# 退出
a007 LOGOUT
```

## 协议关联

- **IMAP与TCP**：IMAP使用TCP的可靠传输，端口143
- **IMAP与SMTP**：SMTP发送，IMAP接收
- **IMAP与POP3**：都是邮件接收协议，IMAP功能更强
- **IMAP与Webmail**：Webmail（网页版邮箱）提供类似IMAP的功能
- **408考点**：
  - IMAP端口143
  - IMAP服务器保留邮件
  - IMAP支持多设备同步
  - POP3与IMAP的区别
- **陷阱**：IMAP是接收协议，发送仍用SMTP；IMAP复杂度高但功能强大
