# API文档规范


## API文档规范


技术写作APIOpenAPI


好的API文档是API成功的关键，OpenAPI是事实标准。


## OpenAPI / Swagger


```
OpenAPI 3.0 规范示例：
openapi: 3.0.3
info:
  title: 用户管理API
  version: 1.0.0
  description: 用户注册、认证和管理接口

paths:
  /users:
    get:
      summary: 获取用户列表
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
                type: object
                properties:
                  users:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'

  /users/{id}:
    get:
      summary: 获取用户详情
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: 成功
        '404':
          description: 用户不存在

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        email:
          type: string
          format: email

工具链：
- Swagger UI：交互式文档界面
- Redoc：美观的文档渲染
- Swagger Codegen：自动生成客户端代码
- Prism：Mock服务器
```


## REST API文档要素


```
每个API端点应包含：
┌─────────────────────────────────────────────┐
│  1. HTTP方法和路径                            │
│     GET /api/v1/users/{id}                   │
│                                             │
│  2. 功能描述                                  │
│     根据ID获取用户详细信息                     │
│                                             │
│  3. 请求参数                                  │
│     路径参数、查询参数、请求头、请求体         │
│                                             │
│  4. 请求示例                                  │
│     curl -X GET /api/v1/users/123 \         │
│       -H "Authorization: Bearer token"      │
│                                             │
│  5. 响应格式                                  │
│     成功和失败的响应结构                       │
│                                             │
│  6. 响应示例                                  │
│     {"id": "123", "name": "张三", ...}       │
│                                             │
│  7. 错误码说明                                │
│     400: 参数错误                             │
│     401: 未认证                               │
│     404: 用户不存在                           │
│                                             │
│  8. 认证方式                                  │
│     Bearer Token / API Key / OAuth2          │
│                                             │
│  9. 速率限制                                  │
│     100次/分钟，超出返回429                   │
└─────────────────────────────────────────────┘
```


## 认证文档与最佳实践


```
认证文档应包含：
- 支持的认证方式
- 获取Token的步骤
- Token刷新机制
- 权限范围(Scopes)说明
- 示例代码

文档最佳实践：
┌─────────────────────────────────────────────┐
│  - 代码优先：API变更同步更新文档               │
│  - 版本管理：每个API版本有独立文档             │
│  - 交互式：提供"Try it"功能                   │
│  - 多语言示例：curl/Python/JS/Go             │
│  - 变更日志：记录API的Breaking Changes       │
│  - SDK文档：自动从代码生成                    │
│  - 搜索功能：帮助开发者快速找到               │
└─────────────────────────────────────────────┘

API文档平台：
- SwaggerHub：OpenAPI协作平台
- ReadMe.io：开发者门户
- Postman：API开发+文档
- Stoplight：API设计+文档
- Redocly：OpenAPI文档渲染
```


> **Note:** API文档应与代码同步维护，使用OpenAPI规范+代码生成是最可靠的方式。
>         提供交互式示例和多语言SDK能大幅提升开发者体验。


<!-- Converted from: 01_API文档规范.html -->
