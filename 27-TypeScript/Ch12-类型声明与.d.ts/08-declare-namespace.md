# declare namespace

## 一、概念说明

`declare namespace` 声明命名空间，用于组织相关的类型和值。在全局声明文件中，命名空间用于避免全局污染。现代 TypeScript 中推荐使用模块替代命名空间，但在某些场景（如全局类型）中仍有用处。

## 二、具体用法

### 2.1 基本命名空间

```typescript
// 全局命名空间
declare namespace MyLib {
  function init(config: Config): void;
  function getVersion(): string;

  interface Config {
    debug: boolean;
    logLevel: 'debug' | 'info' | 'warn' | 'error';
  }

  class Client {
    constructor(endpoint: string);
    request<T>(path: string): Promise<T>;
  }
}

// 使用
MyLib.init({ debug: true, logLevel: 'info' });
const client = new MyLib.Client('https://api.example.com');
```

### 2.2 嵌套命名空间

```typescript
declare namespace MyFramework {
  namespace Router {
    function addRoute(path: string, handler: () => void): void;
    function navigate(path: string): void;
  }

  namespace Utils {
    function debounce<T extends (...args: any[]) => any>(
      fn: T,
      delay: number
    ): T;
    function deepClone<T>(obj: T): T;
  }
}

// 使用
MyFramework.Router.addRoute('/home', () => {});
MyFramework.Utils.debounce(fn, 300);
```

### 2.3 合并命名空间与函数

```typescript
// 函数 + 命名空间（可调用对象）
declare function createApp(config: { root: string }): void;
declare namespace createApp {
  function version(): string;
  function plugins(): string[];
}

// 使用
createApp({ root: '#app' });
createApp.version();
```

### 2.4 命名空间与接口合并

```typescript
declare namespace Express {
  interface Request {
    user?: { id: number };
    sessionId?: string;
  }

  interface Response {
    success(data: unknown): void;
    error(message: string, code?: number): void;
  }
}
```

### 2.5 在模块中使用命名空间

```typescript
// 作为模块导出
declare module 'my-plugin' {
  export namespace Plugin {
    interface Options {
      enabled: boolean;
      config: Record<string, unknown>;
    }

    function setup(options: Options): void;
    function teardown(): void;
  }
}
```

## 三、注意事项与常见陷阱

1. **现代 TS 中优先用模块**：命名空间用于全局类型场景
2. **命名空间可以与函数/类合并**：创建可调用对象
3. **嵌套命名空间用 `.` 访问**：`A.B.C`
4. **命名空间内部的类型不需要 `export`**：在全局命名空间中自动可用
5. **`.d.ts` 中的命名空间是全局的**：除非在模块中声明
