# RedisJSON

## 一、概念说明

RedisJSON是Redis的JSON模块，允许存储、更新和查询JSON文档。支持JSONPath语法，可以高效地访问JSON的嵌套字段。

## 二、具体用法

### 基本操作

```bash
# 设置JSON
JSON.SET user:1 $ '{"name":"张三","age":25,"address":{"city":"北京"}}'

# 获取整个JSON
JSON.GET user:1
# 输出: {"name":"张三","age":25,"address":{"city":"北京"}}

# 获取特定字段
JSON.GET user:1 $.name
# 输出: ["张三"]

JSON.GET user:1 $.address.city
# 输出: ["北京"]
```

### 修改操作

```bash
# 修改字段
JSON.SET user:1 $.age 26
JSON.GET user:1 $.age
# 输出: [26]

# 添加字段
JSON.SET user:1 $.email '"test@example.com"'

# 删除字段
JSON.DEL user:1 $.email
# 输出: (integer) 1

# 数组操作
JSON.SET arr $ '[1,2,3]'
JSON.ARRAPPEND arr $ 4
JSON.GET arr
# 输出: [1,2,3,4]
```

### 数值操作

```bash
# 数值递增
JSON.NUMINCRBY user:1 $.age 1
# 输出: [27]

# 浮点数递增
JSON.NUMINCRBY user:1 $.score 0.5
```

## 三、JSONPath语法

```bash
# $ - 根元素
# . 或 [] - 访问字段
# [*] - 数组所有元素
# [0] - 数组第一个元素
# .. - 递归搜索

JSON.SET data $ '{"users":[{"name":"张三"},{"name":"李四"}]}'
JSON.GET data $.users[*].name
# 输出: ["张三","李四"]
```

## 四、注意事项

1. **需要加载模块**：MODULE LOAD rejson.so
2. **JSONPath支持**：完整的JSONPath查询语法
3. **原子更新**：JSON字段更新是原子的
4. **内存占用**：JSON比单独字段占用更多内存
5. **索引**：可与RediSearch配合建立索引

## 五、复杂JSON操作

```bash
# 嵌套数组操作
JSON.SET data $ '{"users":[{"name":"张三","scores":[90,85,92]},{"name":"李四","scores":[88,91,79]}]}'

# 获取所有用户姓名
JSON.GET data $.users[*].name
# 输出: ["张三","李四"]

# 获取所有分数
JSON.GET data $.users[*].scores[*]
# 输出: [90,85,92,88,91,79]

# 数组追加
JSON.ARRAPPEND data $.users[0].scores 95
# 输出: (integer) 4

# 数组插入
JSON.ARRINSERT data $.users[0].scores 1 88
# 在索引1位置插入88

# 数组删除
JSON.ARRTRIM data $.users[0].scores 0 2
# 只保留前3个元素

# 数组长度
JSON.ARRLEN data $.users[0].scores
# 输出: (integer) 3

# 获取对象键
JSON.OBJKEYS data $
# 输出: ["users"]

# 对象长度
JSON.OBJLEN data $
# 输出: (integer) 1
```

## 六、与RediSearch集成

```bash
# 创建JSON索引
FT.CREATE idx:user ON JSON SCHEMA $.name AS name TEXT $.age AS age NUMERIC

# 添加JSON文档
JSON.SET user:1 $ '{"name":"张三","age":25,"city":"北京"}'
JSON.SET user:2 $ '{"name":"李四","age":30,"city":"上海"}'

# 搜索
FT.SEARCH idx:user "@name:张三"
FT.SEARCH idx:user "@age:[20 30]"

# 聚合
FT.AGGREGATE idx:user "*" GROUPBY 1 @city REDUCE COUNT 0 AS count
```

## 七、性能考量

```bash
# JSON vs Hash对比
# JSON优势：
# 1. 支持嵌套结构
# 2. 支持数组
# 3. 支持JSONPath查询
# 4. 文档化存储

# Hash优势：
# 1. Redis原生支持，无需模块
# 2. 更高效的内存使用
# 3. 更简单的操作
# 4. ziplist编码更紧凑

# 选择建议
# 复杂嵌套结构 → JSON
# 简单KV存储 → Hash
# 需要全文搜索 → JSON + RediSearch
# 内存敏感场景 → Hash

# 内存使用对比
# JSON存储：约 200-500 bytes/文档
# Hash存储：约 50-100 bytes/文档
# 具体取决于文档结构
```
