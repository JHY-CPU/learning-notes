# 包导出类型exports

## 一、概念说明

`package.json` 的 `exports` 字段定义了包的入口点和条件导出，其中 `types` 条件用于指定 TypeScript 类型声明文件的路径。正确配置 `exports` 确保用户在不同模块系统下都能获得正确的类型。

## 二、具体用法

### 2.1 基本 exports 配置

```json
{
  "name": "my-package",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.js"
    }
  },
  "types": "./dist/index.d.ts",
  "main": "./dist/index.js",
  "module": "./dist/index.mjs"
}
```

### 2.2 多入口导出

```json
{
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.js"
    },
    "./utils": {
      "types": "./dist/utils.d.ts",
      "import": "./dist/utils.mjs",
      "require": "./dist/utils.js"
    },
    "./types": {
      "types": "./dist/types.d.ts",
      "default": "./dist/types.js"
    }
  }
}
```

### 2.3 条件导出

```json
{
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "development": "./dist/index.dev.mjs",
      "production": "./dist/index.prod.mjs",
      "default": "./dist/index.mjs"
    }
  }
}
```

### 2.4 通配符导出

```json
{
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs",
      "require": "./dist/index.js"
    },
    "./*": {
      "types": "./dist/*.d.ts",
      "import": "./dist/*.mjs",
      "require": "./dist/*.js"
    }
  }
}

// 用户使用
// import { fn } from 'my-package/utils';
// import { helper } from 'my-package/helpers';
```

### 2.5 禁用深层导入

```json
{
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.mjs"
    },
    "./internal/*": null  // 禁止导入 internal 下的内容
  }
}
```

### 2.6 完整的库 package.json

```json
{
  "name": "my-library",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js",
      "require": "./dist/index.cjs"
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsc && rollup -c"
  },
  "devDependencies": {
    "typescript": "^5.4.0"
  }
}
```

## 三、注意事项与常见陷阱

1. **`types` 条件必须放在最前面**：确保 TypeScript 优先找到类型
2. **`exports` 会覆盖 `main`/`module`**：定义 `exports` 后旧字段被忽略
3. **通配符 `*` 只匹配一层**：`./utils/*` 不匹配 `./utils/helpers/index`
4. **`null` 值禁止导入**：`"./internal": null`
5. **需要 Node.js 12.20+ 支持**：`exports` 是较新的字段
