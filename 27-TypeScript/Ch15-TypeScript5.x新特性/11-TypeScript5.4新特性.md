# TypeScript 5.4新特性

## 一、概念说明

TypeScript 5.4 引入了 `NoInfer` 工具类型和闭包中的类型收窄保留。

## 二、具体用法

### 2.1 NoInfer 工具类型

```typescript
// 阻止类型推断
function createFSM<State extends string>(
  initial: State,
  transitions: Record<State, State[]>
) {
  // ...
}

// 问题：State 被推断为 string
createFSM('idle', { idle: ['loading'], loading: ['idle'] }); // State: "idle" | "loading"

// 使用 NoInfer
function createFSMSafe<State extends string>(
  initial: NoInfer<State>,
  transitions: Record<State, State[]>
) {
  // ...
}

// State 从 transitions 推断
createFSMSafe('idle', { idle: ['loading'], loading: ['idle'] }); // State: "idle" | "loading"
```

### 2.2 闭包中的类型收窄

```typescript
function process(items: string[] | null) {
  if (items === null) return;

  // 5.4 之前：闭包中 items 可能失去收窄
  setTimeout(() => {
    console.log(items.length); // 5.4+：items 保持 string[] 类型
  }, 100);
}
```

## 三、注意事项与常见陷阱

1. **`NoInfer` 阻止特定位置的类型推断**
2. **闭包中的收窄保留减少了 `!` 非空断言的使用**
3. **TypeScript 5.4+ 支持**
