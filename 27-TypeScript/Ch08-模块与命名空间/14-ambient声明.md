# ambient 声明

## 一、概念说明

Ambient 声明（`declare`）用于告诉 TypeScript 存在但**未在当前代码中定义**的变量、函数、类、模块等。`declare` 不生成任何 JavaScript 代码，仅提供类型信息。常见场景：声明全局变量、CSS/图片模块、无类型 JS 库的类型、环境变量等。

## 二、具体用法

### 2.1 声明全局变量和函数

```typescript
// 全局变量
declare const API_URL: string;
declare const __VERSION__: string;
declare let DEBUG_MODE: boolean;

// 全局函数
declare function setTimeout(callback: () => void, ms: number): number;
declare function fetch(url: string, init?: RequestInit): Promise<Response>;

// 直接使用（不需要导入）
console.log(API_URL);
const timer = setTimeout(() => {}, 1000);
```

### 2.2 声明文件模块（通配符）

```typescript
// 为 .css 文件声明模块
declare module "*.css" {
  const classes: Record<string, string>;
  export default classes;
}

// 为 .svg 文件声明模块
declare module "*.svg" {
  const content: string;
  export default content;
}

// 为 .png/.jpg 等图片文件
declare module "*.png" {
  const src: string;
  export default src;
}

// 为 .json 文件
declare module "*.json" {
  const value: unknown;
  export default value;
}

// 使用
import styles from "./App.css";    // styles: Record<string, string>
import logo from "./logo.svg";     // logo: string
import config from "./config.json"; // config: unknown
```

### 2.3 声明无类型 JS 模块

```typescript
// 为没有类型的第三方 JS 库声明类型
declare module "legacy-analytics" {
  export function track(event: string, data?: Record<string, unknown>): void;
  export function identify(userId: string, traits?: Record<string, unknown>): void;
  export const version: string;
  export default function init(apiKey: string): void;
}

// 使用
import init, { track, identify } from "legacy-analytics";
init("my-api-key");
track("page_view");
```

### 2.4 声明枚举和类

```typescript
// 声明枚举
declare enum LogLevel {
  Debug,
  Info,
  Warn,
  Error,
}

// 声明类
declare class EventEmitter {
  on(event: string, listener: (...args: any[]) => void): this;
  emit(event: string, ...args: any[]): boolean;
  off(event: string, listener: (...args: any[]) => void): this;
}
```

### 2.5 声明命名空间

```typescript
// 为全局库声明命名空间
declare namespace JQuery {
  interface AjaxSettings {
    url: string;
    method?: "GET" | "POST" | "PUT" | "DELETE";
    data?: unknown;
  }

  function ajax(settings: AjaxSettings): JQuery.jqXHR;
  function get(url: string): JQuery.jqXHR;
}

declare function $(selector: string): JQuery;
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：全局变量直接使用，无类型提示
console.log(API_URL); // 如果 API_URL 未定义，运行时报 ReferenceError

// TypeScript declare：编译时告诉 TS 这些值存在
// declare const API_URL: string;
// 编译时不检查值是否存在，由开发者保证运行时可用
```

## 三、注意事项与常见陷阱

1. **`declare` 不生成代码**：只提供类型信息，不产生 `var`/`function` 等 JS 代码
2. **必须在 `.d.ts` 文件或模块中**：普通 `.ts` 文件中如果在全局作用域使用 `declare`，文件不能有 `import`/`export`
3. **全局声明影响所有文件**：注意命名冲突，建议放在专门的 `.d.ts` 文件中
4. **`declare module "*"` 通配符**：匹配特定文件扩展名，用于资源文件类型声明
5. **`declare` vs `export`**：`declare` 声明全局/ambient，`export` 声明模块成员
6. **实际值由运行时提供**：`declare` 只告诉 TS 类型，实际值需要在运行时存在（如通过 `<script>` 引入）
