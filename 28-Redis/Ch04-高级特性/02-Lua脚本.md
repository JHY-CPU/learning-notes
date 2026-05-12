# Lua脚本

## 一、概念说明

Redis支持Lua脚本执行，脚本在Redis服务器端原子性执行。Lua脚本可以组合多个命令，实现复杂的原子操作，避免竞态条件。

## 二、具体用法

### EVAL命令

```bash
# 基本语法
EVAL script numkeys key [key ...] arg [arg ...]

# 简单示例
EVAL "return redis.call('SET',KEYS[1],ARGV[1])" 1 mykey "hello"
# 输出: OK

# 获取值
EVAL "return redis.call('GET',KEYS[1])" 1 mykey
# 输出: "hello"

# 原子性递增并检查
EVAL "
  local current = redis.call('INCR',KEYS[1])
  if current > tonumber(ARGV[1]) then
    redis.call('DEL',KEYS[1])
    return 0
  end
  return current
" 1 counter 10
```

### EVALSHA命令

```bash
# 先加载脚本获取SHA1
SCRIPT LOAD "return redis.call('GET',KEYS[1])"
# 输出: "a]b]c]d]..."（SHA1哈希）

# 使用SHA1执行
EVALSHA "a]b]c]d]..." 1 mykey
# 输出: "hello"

# 检查脚本是否存在
SCRIPT EXISTS "a]b]c]d]..."
# 输出: 1) (integer) 1
```

### 分布式锁实现

```bash
# 加锁脚本
EVAL "
  if redis.call('SET',KEYS[1],ARGV[1],'NX','EX',ARGV[2]) then
    return 1
  else
    return 0
  end
" 1 lock:resource "uuid-value" 30
# 输出: 1（成功）或 0（失败）

# 释放锁脚本
EVAL "
  if redis.call('GET',KEYS[1]) == ARGV[1] then
    return redis.call('DEL',KEYS[1])
  else
    return 0
  end
" 1 lock:resource "uuid-value"
```

## 三、脚本管理

```bash
# 查看脚本缓存
SCRIPT LIST

# 清空脚本缓存
SCRIPT FLUSH

# 杀死正在执行的脚本
SCRIPT KILL
# 只能杀死执行超时的脚本

# 脚本超时配置
lua-time-limit 5000
# 脚本执行超过5000毫秒时可被KILL
```

## 四、注意事项

1. 原子性：整个脚本原子执行，不被其他命令打断
2. 脚本慢会阻塞：长时间脚本阻塞整个Redis
3. 集群限制：脚本中的key必须在同一个槽位
4. 脚本缓存：EVALSHA更高效，避免重复传输脚本
5. 无状态：脚本不能访问外部状态

## 五、常用Lua脚本模式

### 限流脚本

```lua
-- 令牌桶限流
local key = KEYS[1]
local rate = tonumber(ARGV[1])    -- 每秒生成令牌数
local capacity = tonumber(ARGV[2]) -- 桶容量
local now = tonumber(ARGV[3])      -- 当前时间戳(毫秒)
local requested = tonumber(ARGV[4]) -- 请求令牌数

local last = tonumber(redis.call('hget', key, 'last') or 0)
local tokens = tonumber(redis.call('hget', key, 'tokens') or capacity)

-- 计算新令牌
local elapsed = math.max(0, now - last)
tokens = math.min(capacity, tokens + elapsed * rate / 1000)

if tokens >= requested then
    tokens = tokens - requested
    redis.call('hmset', key, 'tokens', tokens, 'last', now)
    redis.call('pexpire', key, math.floor(capacity / rate * 1000 * 2))
    return 1
else
    redis.call('hmset', key, 'tokens', tokens, 'last', now)
    return 0
end
```

### 乐观锁更新

```lua
-- CAS更新：仅当值等于预期值时才更新
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2])
    return 1
else
    return 0
end
```

### 批量删除（前缀匹配）

```lua
-- 批量删除指定前缀的Key
local prefix = ARGV[1]
local count = 0
local cursor = '0'
repeat
    local result = redis.call('SCAN', cursor, 'MATCH', prefix .. '*', 'COUNT', 100)
    cursor = result[1]
    for _, key in ipairs(result[2]) do
        redis.call('DEL', key)
        count = count + 1
    end
until cursor == '0'
return count
```

## 六、Python中使用Lua脚本

```python
import redis

r = redis.Redis()

# 加载脚本
limit_script = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""

# 注册脚本（自动缓存SHA1）
limit = r.register_script(limit_script)

# 使用
count = limit(keys=['rate:user:1001'], args=[60])
if count > 100:
    print("限流")
```

## 七、调试Lua脚本

```bash
# 使用redis-cli --eval
redis-cli --eval script.lua key1 key2 , arg1 arg2

# script.lua内容
local val = redis.call('GET', KEYS[1])
return val

# 日志输出
redis.log(redis.LOG_NOTICE, "Key: " .. KEYS[1])
redis.log(redis.LOG_WARNING, "Value: " .. ARGV[1])
```
