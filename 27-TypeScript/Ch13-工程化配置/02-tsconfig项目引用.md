# tsconfig项目引用

## 一、概念说明

项目引用（Project References）允许将大型 TypeScript 项目拆分为多个子项目，每个子项目有独立的 `tsconfig.json`。这能加速构建（增量编译）和改善代码组织。

## 二、具体用法

### 2.1 基本项目引用

```json
// 根目录 tsconfig.json
{
  "files": [],
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/api" },
    { "path": "./packages/web" }
  ]
}
```

```json
// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,       // 必须开启
    "declaration": true,     // 必须生成声明
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true
  },
  "include": ["src/**/*"]
}
```

### 2.2 构建命令

```bash
# 构建所有项目（按依赖顺序）
npx tsc --build

# 构建单个项目
npx tsc --build packages/api

# 增量构建
npx tsc --build --incremental

# 清理构建
npx tsc --build --clean
```

### 2.3 项目间依赖

```json
// packages/api/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "outDir": "./dist"
  },
  "references": [
    { "path": "../shared" }  // 依赖 shared 包
  ],
  "include": ["src/**/*"]
}
```

### 2.4 Monorepo 结构

```
monorepo/
├── packages/
│   ├── shared/           # 共享类型和工具
│   │   ├── src/
│   │   ├── tsconfig.json
│   │   └── package.json
│   ├── api/              # 后端
│   │   ├── src/
│   │   ├── tsconfig.json  # references: [shared]
│   │   └── package.json
│   └── web/              # 前端
│       ├── src/
│       ├── tsconfig.json  # references: [shared]
│       └── package.json
├── tsconfig.json          # 根配置
└── package.json
```

### 2.5 增量构建优化

```json
{
  "compilerOptions": {
    "composite": true,
    "incremental": true,      // 生成 .tsbuildinfo
    "tsBuildInfoFile": "./dist/.tsbuildinfo"
  }
}
```

## 三、注意事项与常见陷阱

1. **`composite: true` 是项目引用的必要条件**
2. **被引用的项目必须生成 `.d.ts`**：`declaration: true`
3. **项目引用只影响构建顺序**：不会自动安装依赖
4. **增量构建缓存在 `.tsbuildinfo` 文件中**：应加入 `.gitignore`
5. **循环引用会导致构建失败**：确保项目间是 DAG（有向无环图）
