# TypeScript最佳实践

## 一、概念说明

Vue 3 + TypeScript 项目的最佳实践涵盖配置、编码规范、类型设计和团队协作。核心原则：开启严格模式、避免 any、善用类型推断、保持类型简洁。本节总结推荐的项目配置和编码约定。

## 二、具体用法

### 推荐的 tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,
    "verbatimModuleSyntax": true,
    "isolatedModules": true,
    "skipLibCheck": true,
    "jsx": "preserve",
    "resolveJsonModule": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src/**/*.ts", "src/**/*.d.ts", "src/**/*.vue"]
}
```

### ESLint 配合 TypeScript

```js
// .eslintrc.cjs
module.exports = {
  extends: [
    'eslint:recommended',
    '@vue/eslint-config-typescript/recommended',
    '@vue/eslint-config-prettier'
  ],
  rules: {
    // 禁止隐式 any
    '@typescript-eslint/no-explicit-any': 'warn',
    // 禁止未使用的变量
    '@typescript-eslint/no-unused-vars': ['error', {
      argsIgnorePattern: '^_'
    }],
    // 强制使用 type import
    '@typescript-eslint/consistent-type-imports': 'error'
  }
}
```

### 类型设计原则

```ts
// 原则1：优先使用 interface 定义对象
interface User {
  id: number
  name: string
  email: string
}

// 原则2：type 用于联合类型、交叉类型、工具类型
type Status = 'active' | 'inactive' | 'pending'
type UserWithStatus = User & { status: Status }

// 原则3：避免过深的嵌套类型
// 不好
type BadConfig = {
  a: { b: { c: { d: { e: string } } } }
}

// 好：扁平化
type GoodConfig = {
  level: string
}

// 原则4：使用 as const 保持字面量类型
const routes = ['/', '/about', '/contact'] as const
type Route = typeof routes[number]  // '/' | '/about' | '/contact'
```

### Composable 类型规范

```ts
// composables/useToggle.ts
// 返回值使用接口定义
interface UseToggleReturn {
  value: Ref<boolean>
  toggle: () => void
  setTrue: () => void
  setFalse: () => void
}

export function useToggle(initial = false): UseToggleReturn {
  const value = ref(initial)
  return {
    value,
    toggle: () => { value.value = !value.value },
    setTrue: () => { value.value = true },
    setFalse: () => { value.value = false }
  }
}
```

### 团队协作规范

```text
1. 严格模式必须开启（strict: true）
2. 新文件必须用 .ts 或 .vue + lang="ts"
3. Props 和 Emits 使用泛型语法定义
4. 公共类型定义在 types/ 目录
5. 组合式函数返回值必须定义接口
6. 禁止使用 any，需要时用 unknown 代替
7. 避免 as 类型断言，使用类型守卫
8. import type 导入纯类型
9. CI 中必须通过 vue-tsc 检查
10. 定期更新 TypeScript 版本
```

### 迁移路径

```text
JavaScript → TypeScript 迁移步骤：

1. tsconfig.json 中 strict: false 开始
2. 文件逐个重命名为 .ts/.vue + lang="ts"
3. 添加基本类型注释（函数参数、返回值）
4. 修复编译错误
5. 逐步开启严格选项：
   - noImplicitAny
   - strictNullChecks
   - strictFunctionTypes
6. 最终开启 strict: true
```

## 三、注意事项与常见陷阱

1. **不要过度类型化**：Vue 能自动推断大部分类型，不需要每个变量都注释
2. **`any` 是最后手段**：先尝试 `unknown`，再用类型守卫收窄
3. **保持类型文件可读**：复杂的类型工具要加注释说明用途
4. **TypeScript 不做运行时验证**：API 返回数据仍需 zod 等库做运行时校验
5. **版本锁定很重要**：Vue、TypeScript、Vite 的版本需要保持兼容
