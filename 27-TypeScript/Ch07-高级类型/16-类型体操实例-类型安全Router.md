# 类型体操实例 - 类型安全 Router

## 一、概念说明

实现一个**类型安全**的路由系统，确保路由参数在处理函数中具有正确的类型。使用**模板字面量类型**提取路由路径中的参数名（如 `:userId`），自动推导出 `{ userId: string }` 类型。这使得路由定义和处理函数之间保持编译时的类型一致性。

## 二、具体用法

### 2.1 路由参数提取（核心类型）

```typescript
// 从路由路径提取参数类型
type ExtractParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: string } & ExtractParams<`/${Rest}`>
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

// 测试
type Params1 = ExtractParams<"/users/:userId">;
// { userId: string }

type Params2 = ExtractParams<"/users/:userId/posts/:postId">;
// { userId: string } & { postId: string }
// 即 { userId: string; postId: string }

type Params3 = ExtractParams<"/about">;
// {}（无参数）
```

### 2.2 类型安全路由类

```typescript
interface RouteHandler<P extends Record<string, string>> {
  (params: P): void | Promise<void>;
}

class TypedRouter {
  private routes = new Map<string, RouteHandler<any>>();

  // 注册路由，处理函数参数类型自动推导
  add<P extends string>(
    path: P,
    handler: RouteHandler<ExtractParams<P>>
  ): void {
    this.routes.set(path, handler as RouteHandler<any>);
  }

  // 匹配路由并调用处理函数
  handle(path: string, params: Record<string, string>): void {
    const handler = this.routes.get(path);
    if (handler) {
      handler(params);
    }
  }
}
```

### 2.3 使用示例

```typescript
const router = new TypedRouter();

// 自动推导参数类型
router.add("/users/:userId", (params) => {
  // params 类型为 { userId: string }
  console.log(`用户 ID: ${params.userId}`);
});

router.add("/users/:userId/posts/:postId", (params) => {
  // params 类型为 { userId: string; postId: string }
  console.log(`用户 ${params.userId} 的帖子 ${params.postId}`);
});

router.add("/about", (params) => {
  // params 类型为 {}（空对象）
  console.log("关于页面");
});

// 调用
router.handle("/users/123", { userId: "123" });
// 输出: 用户 ID: 123

// 以下会编译错误
// router.add("/users/:userId", (params) => {
//   params.postId; // 错误：{} 上不存在 postId
// });
```

### 2.4 增强版：支持 HTTP 方法

```typescript
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";

interface RouteDefinition<M extends HTTPMethod, P extends string> {
  method: M;
  path: P;
  handler: (params: ExtractParams<P>, body?: any) => any;
}

function defineRoute<M extends HTTPMethod, P extends string>(
  method: M,
  path: P,
  handler: (params: ExtractParams<P>, body?: any) => any
): RouteDefinition<M, P> {
  return { method, path, handler };
}

// 使用
const getUser = defineRoute("GET", "/users/:userId", (params) => {
  // params: { userId: string }
  return { id: params.userId, name: "Alice" };
});

const createPost = defineRoute("POST", "/users/:userId/posts", (params, body) => {
  // params: { userId: string }
  // body: 任意类型（可进一步约束）
  return { id: "1", ...body };
});
```

### 2.5 路由参数值的类型

```typescript
// 提取数字参数（通过运行时转换）
type ExtractNumberParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: number } & ExtractNumberParams<`/${Rest}`>
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: number }
      : {};

type NumberParams = ExtractNumberParams<"/users/:userId">;
// { userId: number }

// 联合参数类型
type RouteParamValue = string | number;
type ExtractParamsWith<T extends string, V = string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: V } & ExtractParamsWith<`/${Rest}`, V>
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: V }
      : {};
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：运行时提取路由参数
const path = "/users/:userId/posts/:postId";
const regex = /:(\w+)/g;
const params = [];
let match;
while ((match = regex.exec(path)) !== null) {
  params.push(match[1]);
}
// params = ["userId", "postId"]（运行时）

// TypeScript：编译时提取参数类型
// type Params = ExtractParams<"/users/:userId/posts/:postId">;
// { userId: string; postId: string }
// 编译时就确定处理函数接收的参数类型
```

## 三、注意事项与常见陷阱

1. **路径必须是字面量类型**：`router.add("/users/:id", ...)` 中路径不能是 `string` 变量，否则无法提取参数
2. **`infer` 递归解析**：多个参数需要递归模板匹配提取
3. **可选参数更复杂**：`/users/:id?` 需要更复杂的模板解析逻辑
4. **运行时匹配是独立逻辑**：类型安全只在编译时保证，运行时路由匹配仍需正确实现
5. **参数值始终是 string**：路由参数从 URL 解析，始终是字符串，数字参数需显式转换
6. **TS 4.1+ 需要**：模板字面量类型是 TS 4.1 特性
