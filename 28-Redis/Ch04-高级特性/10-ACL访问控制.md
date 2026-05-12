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

## 五、ACL规则详解

```bash
# 规则语法
# +<command>       允许命令
# -<command>       禁止命令
# +@<category>     允许命令类别
# -@<category>     禁止命令类别
# +<command>|<subcommand>  允许子命令
# ~<pattern>       允许访问的Key模式
# &<pattern>       允许访问的频道模式
# >password        设置密码
# <password        从哈希中移除密码
# nopass           无密码
# on               启用用户
# off              禁用用户
# reset            重置用户

# 命令类别
ACL CAT
# 返回所有命令类别
# @read, @write, @admin, @dangerous, @connection, 
# @transaction, @scripting, @pubsub, @geo, @stream 等
```

## 六、生产环境ACL配置

```bash
# 应用用户：只能访问自己的Key空间
ACL SETUSER app_user on >StrongPass123 ~app:* +@read +@write -@dangerous

# 只读用户：用于监控和报表
ACL SETUSER readonly on >ReadOnlyPass ~* +@read -@dangerous

# 缓存用户：只能操作缓存相关的Key
ACL SETUSER cache_user on >CachePass123 ~cache:* +GET +SET +DEL +EXPIRE +TTL

# 队列用户：只能操作消息队列
ACL SETUSER queue_user on >QueuePass123 ~queue:* +LPUSH +RPOP +BRPOP +LLEN

# 管理员：完全权限
ACL SETUSER admin on >AdminPass123 ~* +@all

# 禁用默认用户
ACL SETUSER default off
```

## 七、ACL持久化

```bash
# 方式1：配置文件
# redis.conf
user app_user on >password ~app:* +@read +@write
user readonly on >password ~* +@read

# 方式2：ACL SAVE命令
ACL SAVE
# 将当前ACL配置保存到aclfile

# 方式3：指定ACL文件
aclfile /etc/redis/users.acl

# 查看ACL日志
ACL LOG
# 返回被拒绝的命令日志

# 清空ACL日志
ACL LOG RESET
```

## 八、ACL调试技巧

```bash
# 测试用户权限
ACL WHOAMI
# 返回当前用户名

# 检查特定命令权限
ACL DRYRUN app_user SET test_key test_value
# 返回是否允许执行

# 查看用户详细信息
ACL GETUSER app_user
# 返回flags, passwords, commands, keys, channels

# 模拟用户操作
AUTH app_user password
# 切换到app_user
SET app:data "test"
# 如果权限允许则成功

# 恢复默认用户
AUTH default ""
```
