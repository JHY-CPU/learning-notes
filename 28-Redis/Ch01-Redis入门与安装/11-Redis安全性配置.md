# Redis安全性配置

## 一、概念说明

Redis安全性配置是生产环境的重要环节。默认情况下Redis没有密码保护，且监听所有接口，这在生产环境中是极大的安全隐患。

## 二、具体用法

### 密码认证

```bash
# redis.conf 配置密码
requirepass "yourStrongPassword123!"

# 客户端连接时认证
redis-cli -a yourStrongPassword123!

# 或者连接后使用AUTH命令
redis-cli
AUTH yourStrongPassword123!
# 输出: OK

# 修改密码
CONFIG SET requirepass "newPassword"
```

### 重命名危险命令

```bash
# redis.conf 禁用危险命令
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG "CONFIG_a1b2c3d4"
rename-command DEBUG ""
rename-command KEYS ""

# 重命名后的使用
redis-cli
CONFIG_a1b2c3d4 GET maxmemory
# 输出: "1073741824"
```

### 网络安全

```bash
# 绑定内网IP（不要绑定0.0.0.0）
bind 192.168.1.100 127.0.0.1

# 保护模式（默认开启）
protected-mode yes

# 修改默认端口
port 6380

# 启用SSL/TLS（Redis 6.0+）
tls-port 6379
port 0
tls-cert-file /path/to/redis.crt
tls-key-file /path/to/redis.key
tls-ca-cert-file /path/to/ca.crt
```

### ACL访问控制（Redis 6.0+）

```bash
# 创建用户并设置权限
ACL SETUSER readonly on >password ~* +@read
ACL SETUSER writer on >password ~* +@write +@read

# 查看用户权限
ACL LIST
ACL GETUSER readonly

# 删除用户
ACL DELUSER writer

# 查看当前用户
ACL WHOAMI
# 输出: "default"
```

### 文件系统安全

```bash
# 限制Redis可访问的目录
# redis.conf
dir /var/lib/redis
dbfilename dump.rdb

# 设置文件权限
chmod 600 /etc/redis/redis.conf
chown redis:redis /etc/redis/redis.conf

# 禁用CONFIG命令的危险操作
rename-command CONFIG ""
```

## 三、注意事项与常见陷阱

1. **不要使用默认端口**：修改为非标准端口可减少扫描攻击
2. **密码强度**：使用强密码，至少16位，包含大小写字母和特殊字符
3. **不要在公网暴露Redis**：使用防火墙规则限制访问
4. **ACL细粒度控制**：为不同应用创建不同用户，遵循最小权限原则
5. **定期审计**：定期检查ACL配置和连接日志
6. **SSL/TLS加密**：跨网络传输时务必启用加密
