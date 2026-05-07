# 类型体操实例 - 类型安全 Router

## 一、概念说明

实现类型安全的路由系统，确保路由参数在处理函数中具有正确的类型。使用模板字面量类型提取路由参数。

## 二、具体用法

### 2.1 路由参数提取

```typescript
type ExtractParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: string } & ExtractParams<Rest>
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

type Params = ExtractParams<"/users/:userId/posts/:postId">;
// { userId: string; postId: string; }
```

### 2.2 路由定义

```typescript
type Route = {
  path: string;
  handler: (params: any) => void;
};

class Router {
  private routes: Route[] = [];

  add<P extends string>(
    path: P,
    handler: (params: ExtractParams<P>) => void
  ): void {
    this.routes.push({ path, handler });
  }

  match(path: string, params: Record<string, string>): void {
    const route = this.routes.find(r => r.path === path);
    route?.handler(params);
  }
}

const router = new Router();

router.add("/users/:userId", (params) => {
  console.log(`用户 ID: ${params.userId}`);
});

router.match("/users/:userId", { userId: "123" });
// 输出: 用户 ID: 123
```

## 三、注意事项与常见陷阱

1. **模板字面量提取**：使用 `infer` 解析路由模式
2. **类型参数必须是字面量**：`router.add("/users/:id", ...)` 而非变量
3. **可选参数**：需更复杂的模板解析
4. **运行时路由匹配**：类型安全只在编译时保证
