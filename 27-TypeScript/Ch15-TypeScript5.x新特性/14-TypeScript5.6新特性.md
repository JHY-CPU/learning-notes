# TypeScript 5.6新特性

## 一、概念说明

TypeScript 5.6 带来了迭代器类型改进、正则表达式类型检查增强、负索引类型推断等特性。迭代器和生成器的类型推断更加精确，`.at(-1)` 支持更精确的元组类型返回。

## 二、具体用法

### 2.1 迭代器类型增强

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
const emailRegex = /^[\w.-]+@[\w.-]+\.\w{2,}$/;

// 类型守卫配合正则
function isEmail(value: string): value is string {
  return emailRegex.test(value);
}

function processInput(input: string) {
  if (isEmail(input)) {
    // input: string (confirmed email format)
    sendEmail(input);
  }
}

// 5.6+：捕获组的类型推断
const match = "hello-world".match(/^(\w+)-(\w+)$/);
if (match) {
  const [, word1, word2] = match;
  // word1, word2: string | undefined
}
```

### 2.3 元组负索引

```typescript
// 5.6+：更好的元组负索引类型
const arr = [1, 2, 3] as const;
// arr: readonly [1, 2, 3]

const last = arr.at(-1); // last: 3（字面量类型）
const second = arr.at(-2); // second: 2

// 配合类型编程
type Last<T extends readonly any[]> = T extends [...any[], infer L] ? L : never;
type LastElement = Last<[1, 2, 3]>; // 3
```

### 2.4 空数组和元组推断

```typescript
// 5.6+：空数组推断改进
const empty: [] = [];

// 不可变数组推断更精确
const points = [
  [0, 0],
  [1, 1],
  [2, 2],
] as const;
// points: readonly [readonly [0, 0], readonly [1, 1], readonly [2, 2]]
```

### 2.5 迭代器辅助函数

```typescript
// 5.6+：迭代器辅助方法类型更精确
function* fibonacci(): Generator<number, void, undefined> {
  let a = 0, b = 1;
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

// take 函数
function take<T>(iter: Iterable<T>, n: number): T[] {
  const result: T[] = [];
  let i = 0;
  for (const item of iter) {
    if (i >= n) break;
    result.push(item);
    i++;
  }
  return result;
}

const first10 = take(fibonacci(), 10);
// [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### 2.6 与 JavaScript 的对比

```javascript
// JavaScript：迭代器和负索引已经存在
const arr = [1, 2, 3];
arr.at(-1); // 3

// TypeScript 5.6：更精确的类型推断
// arr.at(-1) 返回 3 而非 number
// 迭代器的返回类型和 yield 类型更精确
```

## 三、注意事项与常见陷阱

1. **迭代器类型更精确**：`Generator`、`Iterable`、`Iterator` 的泛型推断改善
2. **正则表达式的类型检查**：捕获组的类型仍有限，复杂模式需要手动断言
3. **负索引的类型推断**：`.at(-1)` 对 `as const` 元组返回精确字面量类型
4. **TypeScript 5.6 要求 Node.js 18+**：旧版本 Node.js 不支持
5. **`Generator<T, TReturn, TNext>`**：三个类型参数更清晰
6. **迭代器与异步迭代器**：`AsyncIterable` 和 `AsyncIterator` 的支持也在改进
