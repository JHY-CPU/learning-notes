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
