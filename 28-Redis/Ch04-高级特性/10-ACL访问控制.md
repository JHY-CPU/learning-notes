# ACL访问控制

## 一、概念说明

Redis ACL（Access Control List）是Redis 6.0引入的细粒度权限控制机制。可以为不同用户设置不同的命令权限和数据访问权限。

## 二、具体用法

### 用户管理

```bash
# 创建用户
ACL SETUSER readonly on >password ~* +@read

# 创建读写用户
ACL SETUSER readwrite on >password ~* +@read +@write

# 创建管理员用户
ACL SETUSER admin on >adminpass ~* +@all

# 查看所有用户
ACL LIST

# 查看用户详情
ACL GETUSER readonly
```

### 权限设置

```bash
# 命令权限
+@read      # 允许所有读命令
+@write     # 允许所有写命令
+@all       # 允许所有命令
-@admin     # 禁止管理命令
+GET -SET   # 允许GET，禁止SET

# 键模式权限
~*          # 所有键
~user:*     # user:开头的键
~cache:*    # cache:开头的键

# 频道权限（Pub/Sub）
&*          # 所有频道
&news:*     # news:开头的频道
```

### 使用ACL

```bash
# 使用新用户登录
redis-cli -u redis://readonly:password@localhost:6379

# 切换用户
AUTH readonly password

# 删除用户
ACL DELUSER readonly

# 查看当前用户
ACL WHOAMI
# 输出: "default"
```

## 三、预定义权限

```bash
# 读命令类
+@read      # GET, MGET, EXISTS, TTL, KEYS, SCAN...

# 写命令类
+@write     # SET, MSET, DEL, EXPIRE, LPUSH...

# 管理类
@admin      # CONFIG, DEBUG, SAVE, SHUTDOWN...

# 危险类
@dangerous  # FLUSHALL, FLUSHDB, KEYS...
```

## 四、注意事项

1. **默认用户**：default用户拥有所有权限
2. **禁用默认用户**：user default off
3. **密码安全**：使用强密码
4. **最小权限原则**：只授予必要的权限
5. **Redis 6.0+**：旧版本不支持ACL
