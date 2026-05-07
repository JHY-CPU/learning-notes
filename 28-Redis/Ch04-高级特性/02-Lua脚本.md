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
