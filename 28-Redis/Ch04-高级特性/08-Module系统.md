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
