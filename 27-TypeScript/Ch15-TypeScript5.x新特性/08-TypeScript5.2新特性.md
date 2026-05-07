# TypeScript 5.2新特性

## 一、概念说明

TypeScript 5.2 引入了 `using` 关键字（显式资源管理），这是 TC39 Stage 3 提案，允许自动释放资源。

## 二、具体用法

### 2.1 using 声明

```typescript
// 使用 using 声明的资源在作用域结束时自动释放
function processFile() {
  using file = openFile('data.txt');
  // file 在作用域结束时自动关闭
  console.log(file.read());
}

// 实现 Disposable 接口
class FileHandle implements Disposable {
  constructor(private path: string) {
    console.log(`打开文件: ${path}`);
  }

  read(): string {
    return '文件内容';
  }

  [Symbol.dispose]() {
    console.log(`关闭文件: ${this.path}`);
  }
}

function openFile(path: string): FileHandle {
  return new FileHandle(path);
}
```

### 2.2 await using

```typescript
// 异步资源释放
async function processDb() {
  await using db = await connectDatabase();
  // db 在作用域结束时自动断开连接
  await db.query('SELECT * FROM users');
}

class AsyncResource implements AsyncDisposable {
  [Symbol.asyncDispose](): Promise<void> {
    return new Promise(resolve => {
      console.log('异步清理...');
      setTimeout(resolve, 100);
    });
  }
}
```

### 2.3 资源管理器模式

```typescript
class DatabaseConnection implements Disposable {
  private connected = true;

  query(sql: string): string[] {
    if (!this.connected) throw new Error('已断开');
    return ['result1', 'result2'];
  }

  [Symbol.dispose](): void {
    this.connected = false;
    console.log('数据库连接已关闭');
  }
}

// 使用
function processQuery() {
  using conn = new DatabaseConnection();
  const results = conn.query('SELECT 1');
  // conn 在函数结束时自动断开
}
```

## 三、注意事项与常见陷阱

1. **`using` 实现 `Disposable` 接口**：需要 `[Symbol.dispose]` 方法
2. **`await using` 实现 `AsyncDisposable` 接口**：需要 `[Symbol.asyncDispose]`
3. **资源在作用域结束时释放**：即使发生异常
4. **TypeScript 5.2+ 支持**
5. **需要更新 `lib` 配置**：`"lib": ["ES2022"]` 或更高
