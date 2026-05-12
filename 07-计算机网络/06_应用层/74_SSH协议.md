# 75_SSH协议

## 核心概念

- **SSH（Secure Shell）**：安全外壳协议
- SSH使用**TCP**协议，端口号**22**
- SSH是**TELNET的安全替代品**
- SSH提供：
  - **加密**：数据加密传输
  - **认证**：密码或密钥认证
  - **完整性**：数据不被篡改
  - **端口转发**：隧道功能
- **408考试重点**：SSH端口号、与TELNET对比

## 原理分析

### SSH工作过程

1. **版本协商**：
   - 客户端和服务器交换版本号
   - 协商使用SSH-2.0

2. **密钥交换**：
   - 使用Diffie-Hellman算法
   - 生成会话密钥

3. **认证**：
   - 密码认证：用户提供密码
   - 公钥认证：用户使用密钥对

4. **数据传输**：
   - 使用会话密钥加密数据
   - 支持多种加密算法

5. **连接关闭**：
   - 发送disconnect消息
   - 关闭TCP连接

### SSH协议组成

| 子协议 | 作用 |
|--------|------|
| SSH-TRANS | 传输层协议（密钥交换） |
| SSH-USERAUTH | 用户认证协议 |
| SSH-CONNECT | 连接协议（多路复用） |

### SSH vs TELNET

| 特性 | SSH | TELNET |
|------|-----|--------|
| 端口 | 22 | 23 |
| 加密 | 有 | 无 |
| 认证 | 密码/密钥 | 明文密码 |
| 安全性 | 高 | 低 |
| 端口转发 | 支持 | 不支持 |

### SSH应用场景

1. **远程登录**：安全的远程终端
2. **文件传输**：SCP、SFTP
3. **端口转发**：隧道
4. **密钥管理**：SSH密钥对

## 直观理解

**SSH就像带保镖的远程控制**：
- TELNET = 裸奔远程控制（明文传输）
- SSH = 带保镖的远程控制（加密传输）
- 保镖验证身份（认证）
- 保镖加密通信（加密）
- 保镖保护数据不被篡改（完整性）

**记忆技巧**：
- SSH = "安全外壳"，端口22
- SSH替代TELNET（端口23）
- SSH提供：加密 + 认证 + 完整性
- "SSH 22安全，TELNET 23不安全"

## 代码示例

### 使用 Python paramiko 进行 SSH 远程操作

```python
# pip install paramiko
import paramiko

def ssh_execute(host, username, password, command):
    """通过SSH执行远程命令"""
    # 创建SSH客户端
    client = paramiko.SSHClient()

    # 自动添加未知主机密钥（生产环境应验证）
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接SSH服务器（TCP 22端口 + 密钥交换 + 认证）
    client.connect(host, username=username, password=password)
    print(f"SSH连接成功: {host}")

    # 执行远程命令
    stdin, stdout, stderr = client.exec_command(command)
    print(f"输出: {stdout.read().decode()}")
    print(f"错误: {stderr.read().decode()}")

    # 关闭连接
    client.close()

# 使用示例
ssh_execute('192.168.1.100', 'root', 'password', 'uname -a')
```

### 使用 paramiko 进行 SSH 文件传输（SFTP）

```python
import paramiko

def sftp_transfer(host, username, password):
    """通过SFTP传输文件（基于SSH的安全文件传输）"""
    transport = paramiko.Transport((host, 22))
    transport.connect(username=username, password=password)

    # 创建SFTP客户端
    sftp = paramiko.SFTPClient.from_transport(transport)

    # 上传文件
    sftp.put('local_file.txt', '/remote/path/file.txt')
    print("文件上传成功!")

    # 下载文件
    sftp.get('/remote/path/file.txt', 'downloaded.txt')
    print("文件下载成功!")

    sftp.close()
    transport.close()

# 使用示例
sftp_transfer('192.168.1.100', 'root', 'password')
```

### 使用 ssh 命令行工具

```bash
# SSH远程登录（密码认证）
ssh user@192.168.1.100 -p 22

# SSH执行单条命令
ssh user@192.168.1.100 'ls -la /var/log'

# SSH密钥认证（更安全）
ssh-keygen -t rsa -b 4096          # 生成密钥对
ssh-copy-id user@192.168.1.100     # 将公钥复制到服务器
ssh user@192.168.1.100             # 免密码登录

# SCP文件传输（基于SSH）
scp local.txt user@192.168.1.100:/remote/path/    # 上传
scp user@192.168.1.100:/remote/file.txt ./        # 下载

# SSH端口转发（隧道）
ssh -L 8080:localhost:80 user@192.168.1.100  # 本地端口转发
# 将本地8080端口转发到远程服务器的80端口

# 对比：telnet是明文传输（端口23），SSH是加密传输（端口22）
telnet 192.168.1.100 23  # 不安全，密码明文传输
ssh user@192.168.1.100   # 安全，所有数据加密
```

## 协议关联

- **SSH与TCP**：SSH使用TCP，端口22
- **SSH与TELNET**：SSH是TELNET的安全替代
- **SSH与SSL/TLS**：都提供安全服务，但SSH用于远程登录，SSL/TLS用于通用安全
- **408考点**：
  - SSH端口22
  - SSH使用TCP
  - SSH是TELNET的安全替代
  - SSH提供加密、认证、完整性
- **陷阱**：SSH端口22，不是23（那是TELNET）
