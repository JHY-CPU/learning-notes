# TypeScript 5.1新特性

## 一、概念说明

TypeScript 5.1 带来了 getter/setter 类型改进、尾部元组元素简化等特性。

## 二、具体用法

### 2.1 getter/setter 类型

```typescript
// 5.1 之前：getter 和 setter 必须有相同类型
class Box {
  private _value: string | number = '';

  get value(): string | number {
    return this._value;
  }

  set value(v: string | number) {
    this._value = v;
  }
}

// 5.1+：getter 和 setter 可以有不同类型
class SmartBox {
  private _value = '';

  get value(): string {
    return this._value;
  }

  set value(v: string | number) {
    this._value = String(v);
  }
}

const box = new SmartBox();
box.value = 42; // OK — set 接受 number
const v: string = box.value; // OK — get 返回 string
```

### 2.2 尾部元组元素

```typescript
// 5.1 之前：元组必须有固定长度
type Tuple = [string, number];

// 5.1+：可变长度元组更简洁
type FlexTuple = [string, ...number[]];

const t1: FlexTuple = ['hello'];
const t2: FlexTuple = ['hello', 1, 2, 3];
```

### 2.3 JSX 改进

```typescript
// 支持 JSX 中的解构默认值
// <Component { name = '默认名' } />
```

## 三、注意事项与常见陷阱

1. **getter/setter 类型可以不同**：但要保证语义一致
2. **尾部元组元素简化了可变元组类型**
3. **升级前检查 getter/setter 的类型兼容性**
