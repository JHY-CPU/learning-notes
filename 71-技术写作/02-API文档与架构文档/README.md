# 02-API文档与架构文档

## 1. API文档规范

### 1.1 OpenAPI/Swagger规范

OpenAPI Specification（OAS）是REST API描述的事实标准，当前版本3.1.0。

**核心结构**：

```yaml
openapi: 3.1.0
info:
  title: 用户管理API
  version: 1.0.0
  description: 用户注册、认证和管理接口
servers:
  - url: https://api.example.com/v1
paths:
  /users:
    get:
      summary: 获取用户列表
      operationId: listUsers
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserList'
components:
  schemas:
    User:
      type: object
      required: [id, name, email]
      properties:
        id:
          type: integer
        name:
          type: string
        email:
          type: string
          format: email
```

**最佳实践**：
- 每个端点提供清晰的summary和description
- 使用`$ref`复用schema定义
- 为每个状态码提供response示例
- 使用`enum`限制参数值
- 定义完整的错误响应schema

### 1.2 REST API文档最佳实践

- **端点命名**：使用名词复数，如`/users`、`/orders`
- **HTTP方法语义**：GET（查询）、POST（创建）、PUT（全量更新）、PATCH（部分更新）、DELETE（删除）
- **版本管理**：URL路径（`/v1/users`）或Header（`Accept: application/vnd.api+json;version=1`）
- **分页**：使用`page`+`size`或`cursor`方式
- **过滤/排序**：`?status=active&sort=-created_at`
- **错误响应**：统一格式，包含code、message、details
- **认证说明**：明确说明认证方式（Bearer Token、API Key、OAuth2）
- **速率限制**：文档中说明限流策略和响应头

### 1.3 GraphQL Schema文档

```graphql
"""用户类型，表示系统中的注册用户"""
type User {
  """用户唯一标识"""
  id: ID!
  """用户显示名称"""
  name: String!
  """电子邮件地址"""
  email: String!
  """用户角色"""
  role: UserRole!
  """用户创建时间"""
  createdAt: DateTime!
}

"""用户角色枚举"""
enum UserRole {
  ADMIN
  EDITOR
  VIEWER
}

type Query {
  """根据ID获取用户"""
  user(id: ID!): User
  """获取用户列表，支持分页"""
  users(page: Int = 1, size: Int = 20): UserConnection!
}

input CreateUserInput {
  name: String!
  email: String!
  role: UserRole = VIEWER
}

type Mutation {
  """创建新用户"""
  createUser(input: CreateUserInput!): User!
}
```

**文档要点**：
- 使用三引号为每个类型和字段添加描述
- 枚举值应说明每个选项的含义
- 输入类型标注必填和默认值
- 使用自省查询自动生成文档

### 1.4 gRPC/Protobuf文档

```protobuf
// 用户服务 - 提供用户管理的gRPC接口
service UserService {
  // 获取用户信息
  // 返回指定ID的用户详情，用户不存在时返回NOT_FOUND
  rpc GetUser(GetUserRequest) returns (User) {
    option (google.api.http) = {
      get: "/v1/users/{id}"
    };
  }

  // 创建新用户
  // 使用提供的信息创建用户，email冲突时返回ALREADY_EXISTS
  rpc CreateUser(CreateUserRequest) returns (User);
}

// 用户实体
message User {
  // 用户唯一标识（系统生成）
  string id = 1;
  // 用户显示名称（1-100字符）
  string name = 2;
  // 电子邮件地址（需唯一）
  string email = 3;
  // 用户角色
  UserRole role = 4;
  // 创建时间（RFC 3339格式）
  google.protobuf.Timestamp created_at = 5;
}

// 用户角色枚举
enum UserRole {
  USER_ROLE_UNSPECIFIED = 0; // 未指定
  USER_ROLE_ADMIN = 1;       // 管理员
  USER_ROLE_EDITOR = 2;      // 编辑者
  USER_ROLE_VIEWER = 3;      // 查看者
}
```

**文档规范**：
- 每个service和rpc添加注释说明
- message字段注释说明格式、约束和业务含义
- 枚举值说明每个选项的使用场景
- 使用`google.api.http`注解定义REST映射

## 2. API文档工具

### 2.1 Swagger UI

- 从OpenAPI spec自动生成交互式文档
- 支持在线"Try it out"测试
- 可嵌入到Web应用中
- 配置选项丰富（主题、展开级别、认证集成）

### 2.2 Redoc

- 三栏布局：导航、内容、代码示例
- 响应式设计，支持移动端
- 性能优于Swagger UI（虚拟滚动）
- 支持Markdown扩展、SVG徽章
- 嵌入方式灵活（HTML标签、React组件）

### 2.3 Postman Documentation

- 从Postman Collection自动生成文档
- 支持代码示例多语言展示
- 内置环境变量和认证管理
- 支持协作和版本管理
- 可自定义域名托管

| 工具 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| Swagger UI | 交互性强、社区大 | 视觉效果一般 | 开发调试阶段 |
| Redoc | 美观、性能好 | 无交互测试 | 对外发布文档 |
| Postman | 协作方便 | 依赖平台 | 团队API协作 |

## 3. 架构文档

### 3.1 架构决策记录（ADR）

ADR用于记录项目中的重要技术决策。

**模板**：

```markdown
# ADR-XXX: 决策标题

## 状态
[提议 | 已接受 | 已弃用 | 已替代]

## 背景
描述驱动决策的技术和业务背景。

## 决策
明确陈述做出的决策。

## 备选方案
### 方案A：xxx
- 优点：
- 缺点：

### 方案B：xxx
- 优点：
- 缺点：

## 影响
- 正面影响
- 负面影响/代价
- 风险

## 后续行动
- [ ] 待办事项
```

**实践建议**：
- ADR一旦被接受就不可修改，只能被新的ADR替代
- 与代码同仓库管理
- 按时间顺序编号
- 保持简洁（1-2页）
- 定期回顾ADR的有效性

### 3.2 C4模型

Simon Brown提出的C4模型从四个抽象层级描述软件架构：

```
Level 1: System Context（系统上下文）
    ↓ 放大
Level 2: Container（容器）
    ↓ 放大
Level 3: Component（组件）
    ↓ 放大
Level 4: Code（代码）
```

**Level 1 - 系统上下文图**：展示系统与外部用户、系统的交互关系
- 关注：谁使用系统？系统与哪些外部系统交互？
- 受众：所有利益相关者

**Level 2 - 容器图**：展示系统的技术架构
- 容器：Web应用、API、数据库、消息队列、微服务等
- 关注：容器间的通信方式和技术选择
- 受众：技术团队、架构师

**Level 3 - 组件图**：展示容器内部的组件结构
- 关注：职责划分、内部接口
- 受众：开发团队

**Level 4 - 代码图**：展示组件的类/模块结构
- 通常使用UML类图表示
- 受众：具体开发人员

**工具**：Structurizr（C4专用）、PlantUML、Mermaid、draw.io

### 3.3 系统上下文图

系统上下文图回答以下问题：
- 系统的核心用户角色有哪些？
- 系统提供哪些主要功能？
- 系统依赖哪些外部服务？
- 数据流向是怎样的？

使用矩形表示系统，人形图标表示角色，箭头表示交互。

### 3.4 部署图

描述系统在生产环境中的部署拓扑：
- 服务器节点及其规格
- 网络拓扑和安全组
- 负载均衡策略
- 数据库集群配置
- CDN和缓存部署
- 容器编排（Kubernetes集群拓扑）

### 3.5 数据流图（DFD）

描述数据在系统中的流转过程：

| 层级 | 内容 |
|------|------|
| Level 0 | 系统与外部实体的数据交换（上下文图） |
| Level 1 | 主要处理过程和数据存储 |
| Level 2 | 每个处理过程的详细分解 |

DFD元素：
- **外部实体**：矩形（用户、第三方系统）
- **处理过程**：圆角矩形（数据处理逻辑）
- **数据存储**：开口矩形（数据库、文件）
- **数据流**：箭头（数据移动方向）

## 4. RFC文档

RFC（Request for Comments）用于提出和讨论技术方案。

### 4.1 RFC模板

```markdown
# RFC-XXX: 方案标题

## 摘要
一到三句话总结提案。

## 动机
为什么需要这个改变？解决什么问题？

## 详细设计
详细描述提案的技术方案。

## 教学指南
如何向新用户解释这个功能？

## 参考实现
链接到实现的PR或代码。

## 替代方案
考虑过的其他方案及被拒绝的原因。

## 未解决的问题
尚需讨论的开放问题。

## 后续工作
此提案之后的计划。
```

### 4.2 技术方案评审流程

```
草案 → 评审中 → 已批准 → 已实施 → 已完成
  ↓                                ↓
 已拒绝                          已废弃
```

1. **起草阶段**：作者提交RFC草案
2. **公开评审**：团队成员评审和讨论（通常设定评审期限）
3. **修改迭代**：根据反馈修改提案
4. **最终决策**：技术负责人或委员会做出决策
5. **实施跟踪**：批准后进入实施，关联Issue/PR

**评审要点**：
- 方案是否解决核心问题
- 是否考虑了向后兼容性
- 性能和安全影响
- 实施复杂度和时间成本
- 可测试性和可回滚性

## 5. 文档自动化

### 5.1 从代码生成文档

| 语言/工具 | 文档生成器 | 输入 | 输出格式 |
|-----------|-----------|------|----------|
| Java | Javadoc | `/** */` 注释 | HTML |
| C/C++ | Doxygen | 多种注释格式 | HTML/PDF/LaTeX |
| Python | Sphinx | Docstring | HTML/PDF/ePub |
| JS/TS | JSDoc/TypeDoc | `/** */` 注释 | HTML |
| Go | godoc | `//` 注释 | HTML |
| Rust | rustdoc | `///` 注释 | HTML |
| .NET | DocFX | XML注释 | HTML |

### 5.2 文档即代码（Docs as Code）

核心原则：
- 文档使用**纯文本格式**（Markdown、reStructuredText）
- 文档与代码**同仓库管理**
- 使用**版本控制**（Git）管理文档变更
- 文档变更经过**代码审查**（Pull Request）
- 使用**CI/CD**自动构建和部署文档
- 文档**风格检查**自动化（Vale、alex、proselint）

工具链：
```
编写(IDE/VS Code) → 格式检查(Vale) → 构建(MkDocs/Sphinx) → 部署(GitHub Pages/ReadTheDocs)
```

### 5.3 CI/CD中的文档构建

**GitHub Actions示例流程**：

```yaml
# 文档构建和部署流程
on:
  push:
    branches: [main]
    paths: ['docs/**']

jobs:
  build-and-deploy:
    steps:
      - checkout
      - 代码风格检查（Vale）
      - 构建文档（mkdocs build / sphinx-build）
      - 链接检查（htmltest / linkchecker）
      - 部署（GitHub Pages / S3 / ReadTheDocs）
```

**CI中的文档质量检查**：
- **拼写检查**：cspell、aspell
- **风格检查**：Vale（可配置Microsoft/Google风格指南）
- **链接检查**：htmltest、linkchecker、markdown-link-check
- **可读性检查**：Flesch-Kincaid分数
- **术语一致性**：自定义术语词典
- **过期内容检测**：标记需定期审查的章节

**最佳实践**：
- 文档变更触发独立的CI流水线
- PR预览：自动生成文档预览链接
- 自动部署：合并到main分支后自动发布
- 版本快照：发布时自动创建文档版本快照
- 监控文档构建状态和构建时间
