# 12_FTP文件传输协议

## 核心概念

- **FTP（File Transfer Protocol）**：文件传输协议，用于在网络上进行文件传输
- FTP使用**两个TCP连接**：
  - **控制连接（Control Connection）**：端口**21**，传输命令和响应
  - **数据连接（Data Connection）**：端口**20**（主动模式），传输实际数据
- FTP是**有状态（Stateful）**的协议：服务器维护用户的当前目录等状态
- FTP使用C/S模式
- **408考试重点**：FTP的两个端口、控制连接与数据连接的关系、主动/被动模式

## 原理分析

### FTP工作原理

1. **建立控制连接**：
   - 客户端主动连接服务器的21端口
   - 使用TCP三次握手
   - 控制连接在整个会话期间保持

2. **用户认证**：
   - 客户端发送用户名（USER命令）
   - 客户端发送密码（PASS命令）
   - 服务器验证并返回状态码

3. **文件传输**：
   - 客户端通过控制连接发送命令（如RETR、STOR）
   - 服务器为每个传输建立新的数据连接
   - 数据传输完成后关闭数据连接

4. **关闭连接**：
   - 客户端发送QUIT命令
   - 关闭控制连接

### FTP命令和响应

**常见命令**：
- `USER username`：指定用户名
- `PASS password`：指定密码
- `LIST`：列出文件目录
- `RETR filename`：下载文件（Retrieve）
- `STOR filename`：上传文件（Store）
- `QUIT`：退出

**响应码格式**：xyz（三位数字）
- 1xx：肯定预备应答
- 2xx：肯定完成应答
- 3xx：肯定中间应答
- 4xx：暂时否定完成应答
- 5xx：永久否定完成应答

### FTP特点

- 使用两条独立的TCP连接（控制+数据）
- 控制连接使用NVT ASCII格式
- 数据连接可以传输文本或二进制文件
- FTP是有状态的协议（记住当前工作目录）
- 支持断点续传

## 直观理解

**FTP就像银行办理业务**：
- **控制连接** = 排队叫号系统（传递"请到几号窗口"等信息）
- **数据连接** = 实际办理业务的窗口（传递钱和文件）
- 你可以一直在叫号系统里等待，但每次业务到不同窗口办理
- 银行知道你是谁、你上次办了什么业务（有状态）

**记忆技巧**：
- 端口21 = 控制（1+1=2，2+1=3，容易记）
- 端口20 = 数据（0代表传输数据的"空管道"）
- 两个连接：控制通道发命令，数据通道传文件
- "21管说话，20管干活"

## 代码示例

### 使用 Python ftplib 传输文件

```python
from ftplib import FTP

def ftp_demo(host, username, password):
    """通过FTP协议传输文件"""
    # 连接FTP服务器的21端口（控制连接）
    ftp = FTP(host)
    ftp.login(username, password)       # USER + PASS
    print(f"欢迎消息: {ftp.getwelcome()}")

    # LIST - 列出当前目录文件
    print("\n目录列表:")
    ftp.dir()                           # LIST 命令

    # PWD - 查看当前目录
    print(f"\n当前目录: {ftp.pwd()}")

    # CWD - 切换目录
    ftp.cwd('/pub/documents')           # CWD 命令

    # RETR - 下载文件（数据连接传输）
    with open('downloaded.txt', 'wb') as f:
        ftp.retrbinary('RETR readme.txt', f.write)  # 二进制下载
    print("文件下载完成!")

    # STOR - 上传文件
    with open('local_file.txt', 'rb') as f:
        ftp.storbinary('STOR upload.txt', f.write)  # 二进制上传
    print("文件上传完成!")

    # QUIT - 关闭连接
    ftp.quit()

# 使用示例
ftp_demo('ftp.example.com', 'anonymous', '')
```

### 使用 ftp 命令行工具

```bash
# 连接FTP服务器
ftp ftp.example.com
# 输入用户名和密码

# 常用命令
ftp> dir               # 列出文件（LIST）
ftp> cd /pub           # 切换目录（CWD）
ftp> get file.txt      # 下载文件（RETR）
ftp> put local.txt     # 上传文件（STOR）
ftp> binary            # 切换二进制传输模式
ftp> ascii             # 切换ASCII传输模式
ftp> quit              # 退出（QUIT）

# 也可以使用 curl 传输FTP文件
curl -u username:password ftp://ftp.example.com/pub/file.txt -o local.txt
curl -u username:password -T local.txt ftp://ftp.example.com/pub/upload.txt
```

### 使用 telnet 手动体验 FTP 控制连接

```bash
# 连接FTP控制端口21
telnet ftp.example.com 21
# 服务器响应: 220 FTP Server ready

USER anonymous          # 331 Password required
PASS guest@             # 230 User logged in
PWD                     # 257 "/" is current directory
LIST                    # 150 Opening data connection (建立数据连接)
QUIT                    # 221 Goodbye
```

## 协议关联

- **FTP与TCP**：FTP依赖TCP的可靠传输，端口20和21
- **FTP与Telnet**：FTP控制连接使用与Telnet类似的NVT ASCII格式
- **FTP与HTTP**：
  - HTTP使用单个TCP连接（80端口），FTP使用两个连接
  - HTTP是无状态的，FTP是有状态的
  - HTTP主要用于Web浏览，FTP主要用于文件传输
- **408常考**：为什么FTP需要两个TCP连接？
  - 控制连接在整个会话期间保持
  - 数据连接按需建立和关闭
  - 避免命令和数据的混淆
- **陷阱题**：FTP的20端口仅用于主动模式的数据连接，被动模式下数据连接使用随机端口
