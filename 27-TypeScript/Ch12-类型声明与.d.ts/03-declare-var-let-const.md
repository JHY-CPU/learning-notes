# declare var/let/const

## 一、概念说明

`declare var/let/const` 用于声明全局变量的类型。这些声明告诉 TypeScript 某个变量在运行时存在，并提供其类型信息。常见于声明全局配置、库的全局对象等。

## 二、具体用法

### 2.1 基本全局变量声明

```typescript
// global.d.ts

// const — 不可重新赋值
declare const API_BASE_URL: string;
declare const VERSION: string;
declare const IS_PRODUCTION: boolean;

// var — 可重新赋值（不推荐，用 let）
declare var __DEV__: boolean;

// let — 可重新赋值
declare let appConfig: {
  debug: boolean;
  logLevel: string;
};
```

### 2.2 全局常量场景

```typescript
// Webpack 注入的全局变量
declare const __webpack_public_path__: string;
declare const process: {
  env: {
    NODE_ENV: 'development' | 'production' | 'test';
    [key: string]: string | undefined;
  };
};

// CDN 引入的库
declare const jQuery: JQueryStatic;
declare const $: JQueryStatic;
declare const moment: typeof import('moment');
```

### 2.3 在模块中扩展全局

```typescript
// 在模块文件中扩展全局变量
export {};

declare global {
  const MY_APP_CONFIG: {
    api_url: string;
    version: string;
  };

  var __APP_STATE__: Record<string, unknown>;
}

// 现在所有文件都可以使用 MY_APP_CONFIG
console.log(MY_APP_CONFIG.api_url);
```

### 2.4 Window 对象扩展

```typescript
declare global {
  interface Window {
    gtag: (command: string, ...args: unknown[]) => void;
    dataLayer: Record<string, unknown>[];
    myApp: {
      version: string;
      init: () => void;
    };
  }
}

// 使用
window.gtag('event', 'page_view');
window.myApp.init();
```

### 2.5 只读属性

```typescript
// 使用 readonly 约束
declare const CONFIG: Readonly<{
  API_URL: string;
  TIMEOUT: number;
  MAX_RETRIES: number;
}>;

// CONFIG.API_URL = 'new' // 错误：只读
```

## 三、注意事项与常见陷阱

1. **优先使用 `const`**：全局变量通常不应被修改
2. **`declare var` 会产生全局污染**：用 `let` 或 `const` 替代
3. **全局变量必须在所有文件中唯一**：避免名称冲突
4. **`declare global` 用于模块文件**：普通 `.d.ts` 不需要
5. **全局变量的类型要精确**：避免使用 `any`
