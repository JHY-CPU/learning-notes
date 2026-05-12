# Module系统

## 一、概念说明

Redis Module系统允许开发者扩展Redis功能，添加自定义命令和数据类型。官方提供的模块包括RediSearch、RedisJSON、RedisTimeSeries等。

## 二、官方模块

### RediSearch - 全文搜索

```bash
# 加载模块
MODULE LOAD /path/to/redisearch.so

# 创建索引
FT.CREATE idx ON HASH PREFIX 1 doc: SCHEMA title TEXT body TEXT

# 添加文档
HSET doc:1 title "Redis教程" body "Redis是一个内存数据库"
HSET doc:2 title "MySQL指南" body "MySQL是关系型数据库"

# 搜索
FT.SEARCH idx "Redis"
# 返回包含Redis的文档
```

### RedisJSON - JSON数据

```bash
# 加载模块
MODULE LOAD /path/to/rejson.so

# 设置JSON
JSON.SET user:1 $ '{"name":"张三","age":25}'

# 获取JSON字段
JSON.GET user:1 $.name
# 输出: ["张三"]

# 修改JSON
JSON.SET user:1 $.age 26
```

### RedisTimeSeries - 时序数据

```bash
# 创建时序
TS.CREATE temperature RETENTION 86400 LABELS sensor room1

# 添加数据
TS.ADD temperature 1704067200 25.5
TS.ADD temperature 1704067260 26.0

# 查询数据
TS.RANGE temperature - + COUNT 10
```

## 三、模块管理

```bash
# 查看已加载的模块
MODULE LIST

# 加载模块
MODULE LOAD /path/to/module.so

# 卸载模块
MODULE UNLOAD module_name

# 动态加载（redis.conf）
loadmodule /path/to/module.so
```

## 四、注意事项

1. **版本兼容**：模块需要与Redis版本兼容
2. **性能影响**：模块可能影响Redis性能
3. **内存管理**：模块管理自己的内存
4. **安全性**：只加载可信的模块
5. **集群支持**：部分模块不支持集群模式

## 五、开发自定义模块

```c
// 示例：简单的计数器模块
#include "redismodule.h"

// INCRBYFLOAT命令的模块版本
int MyIncrByFloat(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (argc != 3) return RedisModule_WrongArity(ctx);
    
    double increment;
    RedisModule_StringToDouble(argv[2], &increment);
    
    RedisModuleKey *key = RedisModule_OpenKey(ctx, argv[1], 
        REDISMODULE_READ | REDISMODULE_WRITE);
    
    double current = 0;
    RedisModule_StringToDouble(key, &current);
    current += increment;
    
    RedisModule_StringSet(key, current);
    RedisModule_CloseKey(key);
    
    RedisModule_ReplyWithDouble(ctx, current);
    return REDISMODULE_OK;
}

// 模块初始化
int RedisModule_OnLoad(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    if (RedisModule_Init(ctx, "mymodule", 1, REDISMODULE_APIVER_1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    
    if (RedisModule_CreateCommand(ctx, "MYINCRBYFLOAT", 
        MyIncrByFloat, "write", 1, 1, 1) == REDISMODULE_ERR) {
        return REDISMODULE_ERR;
    }
    
    return REDISMODULE_OK;
}

// 编译
// gcc -fPIC -shared -o mymodule.so mymodule.c -I/path/to/redismodule.h

// 使用
// MODULE LOAD /path/to/mymodule.so
// MYINCRBYFLOAT counter 1.5
```

## 六、模块安全考虑

```bash
# 1. 只加载可信模块
# 模块可以访问Redis内存，有安全隐患
# 2. 模块崩溃会导致Redis崩溃
# 3. 模块的内存泄漏会导致Redis OOM
# 4. 模块可能引入新的安全漏洞
# 5. 生产环境使用官方模块更安全

# 安全加载
# 使用绝对路径
loadmodule /opt/redis/modules/redisearch.so

# 限制模块目录
# redis.conf 中指定模块搜索路径
# 避免从不受信任的位置加载
```

## 七、常用官方模块速查

```bash
# RediSearch - 全文搜索
# 支持索引、查询、聚合
FT.CREATE idx ON HASH SCHEMA title TEXT body TEXT
FT.SEARCH idx "Redis" FILTER score 0 100

# RedisJSON - JSON存储
# 支持JSONPath查询
JSON.SET user $ '{"name":"张三"}'
JSON.GET user $.name

# RedisTimeSeries - 时序数据
# 支持自动聚合和降采样
TS.CREATE temp RETENTION 86400
TS.ADD temp * 25.5
TS.RANGE temp - + AGGREGATION avg 60

# RedisBloom - 概率数据结构
# 布隆过滤器、Cuckoo过滤器等
BF.RESERVE filter 0.01 1000000
BF.ADD filter "item"
BF.EXISTS filter "item"

# RedisGraph - 图数据库
# 支持Cypher查询语言
GRAPH.QUERY social "MATCH (n:Person) RETURN n.name"
```
