# TypeScript 5.6新特性

## 一、概念说明

TypeScript 5.6 带来了迭代器类型改进、正则表达式类型检查等增强。

## 二、具体用法

### 2.1 迭代器类型

```typescript
// 5.6+：更好的迭代器类型推断
function* generateNumbers(): Generator<number, void, undefined> {
  yield 1;
  yield 2;
  yield 3;
}

const gen = generateNumbers();
const first = gen.next(); // IteratorResult<number, void>

// 自定义迭代器
class Range implements Iterable<number> {
  constructor(private start: number, private end: number) {}

  *[Symbol.iterator](): Iterator<number> {
    for (let i = this.start; i <= this.end; i++) {
      yield i;
    }
  }
}

const range = new Range(1, 5);
for (const num of range) {
  console.log(num); // 1, 2, 3, 4, 5
}
```

### 2.2 正则表达式类型

```typescript
// 5.6+：正则表达式更严格的类型检查
const regex = /^[\w.-]+@[\w.-]+\.\w{2,}$/;

// 类型守卫
function isEmail(value: string): value is string {
  return regex.test(value);
}
```

### 2.3 负索引

```typescript
// 5.6+：支持负索引的类型
type NegativeIndex<T extends readonly any[]> =
  `-${T['length'] extends 0 ? never : T['length']}`;

// 更好的元组负索引类型
const arr = [1, 2, 3] as const;
arr.at(-1); // 类型: 3
```

## 三、注意事项与常见陷阱

1. **迭代器类型更精确**：`Generator`、`Iterable`、`Iterator` 的泛型支持
2. **正则表达式的类型检查**：捕获组的类型
3. **负索引的类型推断**：`.at(-1)` 返回更精确的类型
4. **TypeScript 5.6 要求 Node.js 18+**
