# Redis安装（Linux）

## 一、概念说明

在Linux系统上安装Redis有多种方式：使用包管理器（apt/yum）安装最简单，编译安装可以获得最新版本。生产环境推荐编译安装以获得更好的性能和最新特性。

### 安装方式对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| apt/yum | 简单快速 | 版本可能较旧 |
| 编译安装 | 最新版本、可定制 | 需要编译环境 |
| Docker | 环境隔离、易管理 | 有额外开销 |

## 二、具体用法

### 方式一：apt安装（Debian/Ubuntu）

```bash
# 更新包索引
sudo apt update

# 安装Redis服务器
sudo apt install redis-server -y

# 查看安装版本
redis-server --version
# 输出: Redis server v=7.0.x sha=00000000:0 malloc=jemalloc-5.3.0

# 启动Redis服务
sudo systemctl start redis-server

# 设置开机自启
sudo systemctl enable redis-server

# 检查服务状态
sudo systemctl status redis-server
# 输出: active (running)

# 测试连接
redis-cli ping
# 输出: PONG
```

### 方式二：yum安装（CentOS/RHEL）

```bash
# 安装EPEL仓库
sudo yum install epel-release -y

# 安装Redis
sudo yum install redis -y

# 启动服务
sudo systemctl start redis
sudo systemctl enable redis

# 检查状态
sudo systemctl status redis
```

### 方式三：编译安装（推荐生产环境）

```bash
# 安装依赖
sudo apt install build-essential tcl -y

# 下载源码（以7.2.x为例）
cd /tmp
wget https://download.redis.io/releases/redis-7.2.4.tar.gz
tar -xzf redis-7.2.4.tar.gz
cd redis-7.2.4

# 编译安装
make
make test    # 可选：运行测试
sudo make install PREFIX=/usr/local/redis

# 创建配置目录
sudo mkdir -p /etc/redis /var/lib/redis

# 复制配置文件
sudo cp redis.conf /etc/redis/

# 创建系统服务
sudo tee /etc/systemd/system/redis.service << 'EOF'
[Unit]
Description=Redis In-Memory Data Store
After=network.target

[Service]
Type=forking
ExecStart=/usr/local/redis/bin/redis-server /etc/redis/redis.conf
ExecStop=/usr/local/redis/bin/redis-cli shutdown
Restart=always
User=redis

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl daemon-reload
sudo systemctl start redis
```

## 三、注意事项与常见陷阱

1. **版本选择**：生产环境建议使用稳定版，不要用rc版
2. **编译依赖**：缺少gcc和tcl会导致编译失败
3. **内存设置**：安装后务必配置`maxmemory`防止OOM
4. **绑定地址**：默认绑定127.0.0.1，远程连接需改为0.0.0.0或指定IP
5. **防火墙**：确保6379端口开放
6. **权限问题**：编译安装后需创建redis用户运行服务
7. **内核参数**：生产环境需设置`vm.overcommit_memory=1`
