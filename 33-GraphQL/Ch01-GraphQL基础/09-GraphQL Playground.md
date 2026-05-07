# GraphQL Playground

## 一、开发工具

```yaml
开发工具:
  GraphQL Playground:
    - 功能: 查询编写、文档浏览
    - 地址: http://localhost:4000/graphql
    - 特点: 内置自省查询

  Apollo Studio:
    - 功能: Schema 管理、性能分析
    - 特点: 云端协作

  GraphiQL:
    - 功能: 轻量级查询工具
    - 特点: 嵌入式集成

  Insomnia / Postman:
    - 功能: API 测试工具
    - 特点: 支持多种 API 类型
```

## 二、自省查询

```graphql
# 查询所有类型
{
  __schema {
    types {
      name
      kind
      description
    }
  }
}

# 查询特定类型的字段
{
  __type(name: "User") {
    name
    fields {
      name
      type {
        name
        kind
        ofType { name }
      }
      description
    }
  }
}

# 查询所有 Query 入口
{
  __schema {
    queryType {
      fields {
        name
        description
        args {
          name
          type { name kind }
        }
      }
    }
  }
}
```

## 三、Apollo Studio 功能

```yaml
Apollo Studio 功能:
  Schema 管理:
    - Schema 注册与版本控制
    - 变更检测与影响分析
    - Schema 文档自动生成

  查询分析:
    - 查询性能追踪
    - 字段使用统计
    - 慢查询识别

  团队协作:
    - Schema 评审
    - 操作审计日志
    - API 变更通知
```

## 四、注意事项

1. **生产环境要禁用 Playground**
2. **自省查询在生产环境要关闭**
3. **Apollo Studio 可以管理 Schema 版本**
4. **使用 vscode-graphql 插件提升开发体验**
5. **查询历史有助于调试和复现问题**
