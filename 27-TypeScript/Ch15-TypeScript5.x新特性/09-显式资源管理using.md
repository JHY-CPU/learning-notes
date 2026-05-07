# 显式资源管理using

## 一、概念说明

`using` 和 `await using` 是 TC39 显式资源管理提案的实现，提供类似 C# `using` 语句和 Python `with` 语句的资源管理能力。确保资源在使用后被正确释放。

## 二、具体用法

### 2.1 Disposable 接口

```typescript
// 实现 Disposable 的类
class Lock implements Disposable {
  private locked = false;

  acquire(): void {
    this.locked = true;
    console.log('锁已获取');
  }

  release(): void {
    this.locked = false;
    console.log('锁已释放');
  }

  [Symbol.dispose](): void {
    this.release();
  }
}

// 使用
function withLock() {
  const lock = new Lock();
  lock.acquire();
  using _guard = { [Symbol.dispose]: () => lock.release() };
  // 作用域结束时自动释放
}
```

### 2.2 SuppressedError

```typescript
// 如果清理时发生错误，会被收集
class Resource implements Disposable {
  [Symbol.dispose](): void {
    throw new Error('清理失败');
  }
}

try {
  using resource = new Resource();
  throw new Error('业务错误');
} catch (e) {
  // e 是 SuppressedError，包含业务错误和清理错误
  if (e instanceof SuppressedError) {
    console.log('业务错误:', e.error);
    console.log('清理错误:', e.suppressed);
  }
}
```

### 2.3 与 try-finally 对比

```typescript
// 旧方式：try-finally
function oldWay() {
  const handle = openResource();
  try {
    // 使用资源
    handle.doWork();
  } finally {
    handle.close();
  }
}

// 新方式：using
function newWay() {
  using handle = openResource();
  // handle 在作用域结束时自动关闭
  handle.doWork();
}
```

### 2.4 异步资源

```typescript
class AsyncFile implements AsyncDisposable {
  async read(): Promise<string> {
    return 'file content';
  }

  async [Symbol.asyncDispose](): Promise<void> {
    console.log('异步关闭文件');
  }
}

async function readAndProcess() {
  await using file = new AsyncFile();
  const content = await file.read();
  console.log(content);
  // file 在函数结束时异步关闭
}
```

## 三、注意事项与常见陷阱

1. **`using` 必须在作用域顶部声明**：不能在 if/for 内部
2. **释放顺序是 LIFO**：后声明的先释放
3. **`SuppressedError` 包含业务错误和清理错误**
4. **异步清理用 `await using`**
5. **TypeScript 5.2+，需要支持的运行时**（Node.js 20+）
