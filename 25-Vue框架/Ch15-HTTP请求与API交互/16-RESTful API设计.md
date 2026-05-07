# RESTful API 设计

## 一、概念说明

REST（Representational State Transfer）是一种 API 设计风格，通过统一的 URL 和 HTTP 方法操作资源。遵循 REST 规范的 API 更加清晰、可预测、易维护。

```js
// API 接口设计示例
const api = {
  // 用户资源
  getUsers:    ()          => axios.get('/api/users'),
  getUser:     (id)        => axios.get(`/api/users/${id}`),
  createUser:  (data)      => axios.post('/api/users', data),
  updateUser:  (id, data)  => axios.put(`/api/users/${id}`, data),
  deleteUser:  (id)        => axios.delete(`/api/users/${id}`),

  // 文章资源
  getPosts:    (params)    => axios.get('/api/posts', { params }),
  getPost:     (id)        => axios.get(`/api/posts/${id}`),
  createPost:  (data)      => axios.post('/api/posts', data),
  updatePost:  (id, data)  => axios.patch(`/api/posts/${id}`, data),
  deletePost:  (id)        => axios.delete(`/api/posts/${id}`),
}
```

## 二、具体用法

### 2.1 资源命名规范

| 说明 | 错误 | 正确 |
|------|------|------|
| 使用复数名词 | `/api/user` | `/api/users` |
| 层级关系 | `/api/getUserPosts` | `/api/users/:id/posts` |
| 小写 + 连字符 | `/api/UserInfo` | `/api/user-info` |

### 2.2 HTTP 方法与操作

| 方法 | 语义 | 幂等性 | 示例 |
|------|------|--------|------|
| GET | 查询 | 是 | `GET /api/users` |
| POST | 创建 | 否 | `POST /api/users` |
| PUT | 全量更新 | 是 | `PUT /api/users/1` |
| PATCH | 部分更新 | 是 | `PATCH /api/users/1` |
| DELETE | 删除 | 是 | `DELETE /api/users/1` |

### 2.3 响应格式约定

```json
{
  "code": 0,
  "message": "success",
  "data": { "id": 1, "name": "张三" }
}
```

```json
{
  "code": 0,
  "data": {
    "list": [],
    "total": 100,
    "page": 1,
    "size": 10
  }
}
```

## 三、注意事项与常见陷阱

- URL 中不要包含动词，用 HTTP 方法表示操作
- `PUT` 是全量更新，`PATCH` 是部分更新，按需选择
- 分页参数 `page` 和 `size` 通过 query string 传递
