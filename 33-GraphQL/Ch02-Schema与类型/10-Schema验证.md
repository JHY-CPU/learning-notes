# Schema 验证

## 一、语法验证

```graphql
# ✓ 合法 Schema
type User {
  id: ID!
  name: String!
}

type Query {
  user(id: ID!): User
}

# ✗ 非法 Schema - 常见错误
# 1. 循环定义 (自引用需要类型)
type User {
  id: ID!
  friends: [User!]!     # ✓ 合法 - 自引用是允许的
}

# 2. 未定义类型引用
type Order {
  id: ID!
  user: User!           # ✗ 如果 User 未定义则报错
}

# 3. Query 入口重复
type Query {
  user(id: ID!): User
}
type Query {             # ✗ 重复的 Query 类型定义
  users: [User!]!
}
# 正确做法: 用 extend type Query
```

## 二、自省验证

```graphql
# 验证 Schema 完整性
{
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      name
      kind
      fields {
        name
        type {
          name
          kind
          ofType { name }
        }
      }
    }
  }
}
```

## 三、代码验证

```java
// graphql-java Schema 验证
public class SchemaValidator {

    public List<SchemaValidationError> validate(String sdl) {
        TypeDefinitionRegistry registry = new SchemaParser()
            .parse(sdl);

        SchemaValidator schemaValidator = new SchemaValidator();
        return schemaValidator.validateSchema(registry);
    }

    // 常见验证问题
    // - 引用未定义的类型
    // - 循环类型定义（无出口）
    // - 字段参数类型不合法
    // - 接口未被实现
}
```

## 四、CI 集成

```yaml
# Schema 验证检查
schema_checks:
  breaking_changes:
    - 删除字段
    - 重命名字段
    - 新增非空字段
    - 删除枚举值
    - 缩小类型范围

  safe_changes:
    - 新增可选字段
    - 新增枚举值
    - 放宽类型范围
    - 新增可选参数
```

## 五、注意事项

1. **Schema 合并后要验证完整性**
2. **CI 中集成 Schema 变更检测**
3. **使用 graphql-inspector 检测破坏性变更**
4. **接口的所有字段必须被实现者包含**
5. **联合类型的所有成员必须是已定义的对象类型**
