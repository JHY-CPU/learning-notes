# Schema Stitching (已废弃，推荐 Federation)

## 一、Schema 拼接方式

```typescript
// Apollo Federation 是推荐方案
// Schema Stitching 已不再推荐
// 以下为了解原理

import { stitchSchemas } from '@graphql-tools/stitch';

const stitchedSchema = stitchSchemas({
  subschemas: [
    { schema: userServiceSchema },
    { schema: orderServiceSchema },
    { schema: productServiceSchema },
  ],
  mergeTypes: true,
});
```

## 二、Federation vs Stitching

```yaml
对比:
  Federation:
    - Apollo 官方推荐
    - 子图各自声明扩展
    - Gateway 统一编排
    - 实体自动解析
    - 生态活跃

  Stitching (旧):
    - 手动合并 Schema
    - 需要定义所有类型映射
    - 配置复杂
    - 已不推荐使用
```

## 三、迁移建议

```yaml
从 Stitching 迁移到 Federation:
  Step 1: 梳理现有类型
    - 识别核心实体
    - 确定子图边界

  Step 2: 拆分子图
    - 每个服务一个子图
    - 定义 @key 指令

  Step 3: 配置 Gateway
    - 注册子图
    - 验证编译

  Step 4: 灰度切换
    - 新旧并行
    - 验证功能
    - 全量切换
```

## 四、注意事项

1. **Federation 是微服务 GraphQL 的标准方案**
2. **Stitching 已不推荐新项目使用**
3. **迁移要渐进式**
4. **实体关系要仔细设计**
5. **Gateway 要做高可用**
