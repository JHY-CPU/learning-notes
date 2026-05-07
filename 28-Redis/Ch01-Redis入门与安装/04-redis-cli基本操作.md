# redis-cli基本操作

## 一、概念说明

`redis-cli`是Redis自带的命令行客户端工具，用于连接Redis服务器并执行各种命令。它是学习和管理Redis最常用的工具。

## 二、具体用法

### 连接Redis

```bash
# 基本连接（默认连接本地6379）
redis-cli

# 指定主机和端口
redis-cli -h 192.168.1.100 -p 6379

# 带密码连接
redis-cli -h 127.0.0.1 -p 6379 -a yourpassword

# 连接后选择数据库
redis-cli -n 2
```

### 基本命令测试

```bash
# 测试连接 - PING命令
127.0.0.1:6379> PING
# 输出: PONG

# 带参数的PING
127.0.0.1:6379> PING "hello"
# 输出: "hello"

# 切换数据库（0-15）
127.0.0.1:6379> SELECT 0
# 输出: OK

127.0.0.1:6379> SELECT 3
# 输出: OK

# 查看当前数据库信息
127.0.0.1:6379> DBSIZE
# 输出: (integer) 5

# 查看服务器信息
127.0.0.1:6379> INFO server
# 输出: # Server
# redis_version:7.2.4
# redis_mode:standalone
# ...

# 清空当前数据库
127.0.0.1:6379> FLUSHDB
# 输出: OK

# 退出连接
127.0.0.1:6379> QUIT
```

### 批量执行命令

```bash
# 通过管道批量执行
echo -e "SET name redis\nGET name\nINCR counter" | redis-cli

# 从文件执行命令
cat commands.txt | redis-cli --pipe

# 交互模式下执行多个命令
redis-cli <<EOF
SET key1 value1
SET key2 value2
GET key1
EOF
```

### 连接池模式

```bash
# 使用--pipe模式进行批量导入
cat data.txt | redis-cli --pipe

# 使用CSV导入
redis-cli --csv SET key1 value1

# 监控命令执行
redis-cli MONITOR
# 实时显示所有执行的命令（调试用）
```

## 三、注意事项与常见陷阱

1. **生产环境禁用MONITOR**：MONITOR命令会严重影响性能
2. **密码安全**：避免在命令行历史中留下密码记录
3. **连接超时**：长时间空闲连接会自动断开，设置`timeout`参数
4. **数据库编号**：默认16个数据库（0-15），SELECT切换
5. **批量操作**：大数据量操作使用`--pipe`模式，不要逐条执行
6. **输出格式**：可使用`--raw`参数保持中文等特殊字符原始显示
