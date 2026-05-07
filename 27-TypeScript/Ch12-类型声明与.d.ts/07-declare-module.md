# declare module

## 一、概念说明

`declare module` 用于为 JavaScript 模块提供类型声明，或扩展已有模块的类型。最常见的是为没有类型的第三方库编写类型声明，以及声明非代码资源（如 CSS、图片）的模块类型。

## 二、具体用法

### 2.1 为 JS 库声明类型

```typescript
// types/legacy-lib.d.ts
declare module 'legacy-lib' {
  export function initialize(config: { url: string }): void;
  export function request<T>(path: string): Promise<T>;

  export class Client {
    constructor(baseUrl: string);
    get<T>(path: string): Promise<T>;
    post<T>(path: string, data: unknown): Promise<T>;
  }

  export interface Config {
    url: string;
    timeout?: number;
  }
}

// 使用 — TypeScript 知道所有类型
import { Client, type Config } from 'legacy-lib';
```

### 2.2 资源文件模块声明

```typescript
// CSS Modules
declare module '*.module.css' {
  const classes: Record<string, string>;
  export default classes;
}

// 普通 CSS
declare module '*.css' {
  const content: string;
  export default content;
}

// 图片
declare module '*.png' {
  const src: string;
  export default src;
}

declare module '*.svg' {
  import type { Component } from 'vue';
  const component: Component;
  export default component;
}

// JSON
declare module '*.json' {
  const value: Record<string, unknown>;
  export default value;
}
```

### 2.3 通配符模块声明

```typescript
// 所有 .png 文件
declare module '*.png' {
  const src: string;
  export default src;
}

// 所有以 plugin- 开头的模块
declare module 'plugin-*' {
  export function setup(): void;
  export const name: string;
}
```

### 2.4 扩展已有模块

```typescript
// 扩展 express 模块
declare module 'express' {
  interface Request {
    user?: { id: number; role: string };
    startTime?: number;
  }
}

// 扩展 Vue 模块
declare module 'vue' {
  interface ComponentCustomProperties {
    $http: HttpClient;
  }
}
```

### 2.5 条件模块声明

```typescript
// 只在特定环境下有类型
declare module 'dev-tools' {
  export function debug(message: string): void;
  export function inspect(value: unknown): void;
}

// 在生产环境中，模块不存在
// 但类型声明仍然可用，只是运行时会报错
```

## 三、注意事项与常见陷阱

1. **模块名要用引号**：`declare module 'name'` 而非 `declare module name`
2. **通配符声明只支持 `*`**：`*.css`、`*.png` 等
3. **不要在声明文件中写实现**：只有类型声明
4. **多个声明文件可以声明同一模块**：会自动合并
5. **资源声明确保 import 不报错**：如 `import logo from './logo.png'`
