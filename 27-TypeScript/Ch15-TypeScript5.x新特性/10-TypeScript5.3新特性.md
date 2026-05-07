# TypeScript 5.3新特性

## 一、概念说明

TypeScript 5.3 带来了 `import` 属性、`is` 关键字在参数中的收窄等改进。

## 二、具体用法

### 2.1 Import 属性

```typescript
// 使用 with 关键字指定导入类型
import config from './config.json' with { type: 'json' };

// 动态导入
const data = await import('./data.json', { with: { type: 'json' } });
```

### 2.2 参数中的 is 收窄

```typescript
// 5.3 之前：类型守卫不能在参数位置
// 5.3+：可以在参数中使用 is

function processValue(value: string | number, isString: value is string) {
  if (isString) {
    // value 被收窄为 string
    console.log(value.toUpperCase());
  } else {
    // value 被收窄为 number
    console.log(value.toFixed(2));
  }
}
```

### 2.3 switch(true) 中的类型收窄

```typescript
function check(value: string | number | boolean) {
  switch (true) {
    case typeof value === 'string':
      return value.toUpperCase(); // string
    case typeof value === 'number':
      return value.toFixed(2); // number
    case typeof value === 'boolean':
      return value ? '是' : '否'; // boolean
  }
}
```

### 2.4 Import 属性类型检查

```typescript
// 类型声明
declare module '*.json' {
  const value: unknown;
  export default value;
}

// 使用 import 属性确保类型安全
import data from './config.json' with { type: 'json' };
```

## 三、注意事项与常见陷阱

1. **`with { type: 'json' }`**：确保 JSON 导入的正确性
2. **参数中的 `is` 收窄**：TypeScript 5.3+ 特性
3. **Import 属性是 TC39 Stage 3 提案**
4. **与 `assert` 语法的区别**：`with` 替代了 `assert`
