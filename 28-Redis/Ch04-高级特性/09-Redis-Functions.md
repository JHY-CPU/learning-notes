# Redis Functions

## 一、概念说明

Redis Functions是Redis 7.0引入的新特性，替代Lua脚本作为服务器端编程方案。Functions支持库管理、持久化和集群复制。

## 二、具体用法

### 创建函数库

```bash
# 加载函数库
FUNCTION LOAD "#!lua name=mylib
  redis.register_function('myfunc', function(keys, args)
    return redis.call('GET', keys[1])
  end)
  
  redis.register_function('setfunc', function(keys, args)
    return redis.call('SET', keys[1], args[1])
  end)
"
# 输出: OK
```

### 调用函数

```bash
# 调用函数
FCALL myfunc 1 mykey
# 输出: 返回mykey的值

FCALL setfunc 1 mykey "newvalue"
# 输出: OK

# 指定库名调用
FCALL mylib.myfunc 1 mykey
```

### 函数管理

```bash
# 列出所有函数
FUNCTION LIST

# 查看函数详情
FUNCTION LIST LIBRARYNAME mylib

# 删除函数库
FUNCTION DELETE mylib

# 清空所有函数
FUNCTION FLUSH

# 导出函数库
FUNCTION DUMP

# 导入函数库
FUNCTION RESTORE <payload>
```

## 三、Functions vs Lua脚本

| 特性 | Functions | Lua脚本 |
|------|-----------|---------|
| 持久化 | 支持 | 不支持 |
| 集群复制 | 自动 | 手动 |
| 库管理 | 支持 | 不支持 |
| 版本控制 | 支持 | 不支持 |
| 引入 | Redis 7.0+ | Redis 2.6+ |

## 四、注意事项

1. **Redis 7.0+才支持**：旧版本不支持
2. **函数不可修改**：需要删除后重新加载
3. **集群自动复制**：函数库自动复制到所有节点
4. **执行原子性**：与Lua脚本一样原子执行
5. **替代Lua脚本**：新项目建议使用Functions

## 五、Functions深入示例

### 注册多个函数

```bash
FUNCTION LOAD "#!lua name=library_v1
  -- 获取用户信息
  redis.register_function('get_user', function(keys, args)
    local key = keys[1]
    return redis.call('HGETALL', key)
  end)

  -- 原子性递增并检查
  redis.register_function('incr_and_check', function(keys, args)
    local key = keys[1]
    local limit = tonumber(args[1])
    local current = redis.call('INCR', key)
    if current == 1 then
      redis.call('EXPIRE', key, 3600)
    end
    if current > limit then
      return 0
    end
    return current
  end)

  -- 批量设置带过期时间
  redis.register_function('batch_set', function(keys, args)
    local ttl = tonumber(args[#args])
    for i, key in ipairs(keys) do
      redis.call('SET', key, args[i], 'EX', ttl)
    end
    return #keys
  end)
"
```

### 调用函数

```bash
# 调用get_user
FCALL get_user 1 user:1001

# 调用incr_and_check（限流）
FCALL incr_and_check 1 rate:user:1001 100

# 调用batch_set（3个key，3个value，1个TTL）
FCALL batch_set 3 key1 key2 key3 val1 val2 val3 3600
```

## 六、Functions管理最佳实践

```bash
# 版本管理
# 函数库名包含版本号
library_v1, library_v2, library_v3

# 灰度发布
# 1. 加载新版本函数库
FUNCTION LOAD "#!lua name=library_v2 ..."
# 2. 部分流量切换到新函数
# 3. 验证无问题后全量切换
# 4. 删除旧版本
FUNCTION DELETE library_v1

# 备份与恢复
# 导出所有函数
FUNCTION DUMP > functions_backup.bin

# 恢复函数
FUNCTION RESTORE < functions_backup.bin

# 集群环境
# Functions自动复制到所有节点
# 无需手动在每个节点加载
```

## 七、迁移Lua脚本到Functions

```bash
# 原Lua脚本
EVAL "return redis.call('GET', KEYS[1])" 1 mykey

# 迁移到Functions
FUNCTION LOAD "#!lua name=migrated
  redis.register_function('get_value', function(keys, args)
    return redis.call('GET', keys[1])
  end)
"

# 调用
FCALL get_value 1 mykey

# 迁移建议
# 1. 逐步迁移，不一次性替换
# 2. 保留Lua脚本作为降级方案
# 3. 测试Functions的性能和正确性
# 4. 更新客户端调用方式
```
