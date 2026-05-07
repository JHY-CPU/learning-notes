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
