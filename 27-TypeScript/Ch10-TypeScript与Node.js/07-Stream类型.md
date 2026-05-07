# Stream类型

## 一、概念说明

Stream 是 Node.js 处理大量数据的核心机制。TypeScript 提供了泛型的流类型定义，包括 `Readable`、`Writable`、`Transform` 和 `Duplex`，每种都有明确的类型约束。

## 二、具体用法

### 2.1 可读流类型

```typescript
import { Readable, Transform, Writable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

// 创建自定义可读流
function createNumberStream(max: number): Readable {
  let current = 0;

  return new Readable({
    objectMode: true, // 推送对象而非 Buffer
    read() {
      if (current < max) {
        this.push(++current);
      } else {
        this.push(null); // 结束信号
      }
    },
  });
}

// 使用
const stream = createNumberStream(5);
stream.on('data', (chunk: number) => console.log(chunk)); // 1, 2, 3, 4, 5
```

### 2.2 转换流类型

```typescript
// 创建自定义转换流
function createLineParser(): Transform {
  let buffer = '';

  return new Transform({
    objectMode: true,
    transform(chunk: Buffer, encoding, callback) {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? ''; // 保留不完整的行

      for (const line of lines) {
        if (line.trim()) {
          this.push(JSON.parse(line));
        }
      }
      callback();
    },

    flush(callback) {
      if (buffer.trim()) {
        this.push(JSON.parse(buffer));
      }
      callback();
    },
  });
}
```

### 2.3 可写流类型

```typescript
// 创建数据库写入流
function createDbWriter(): Writable {
  return new Writable({
    objectMode: true,
    async write(chunk: unknown, encoding, callback) {
      try {
        await db.insert(chunk);
        callback();
      } catch (err) {
        callback(err as Error);
      }
    },

    // 批量写入
    writev(chunks, callback) {
      const items = chunks.map(c => c.chunk);
      db.insertMany(items)
        .then(() => callback())
        .catch(callback);
    },
  });
}
```

### 2.4 管道组合

```typescript
// 使用 pipeline 组合流
async function processFile(inputPath: string, outputPath: string) {
  const readStream = fs.createReadStream(inputPath);
  const writeStream = fs.createWriteStream(outputPath);

  await pipeline(
    readStream,
    createLineParser(),   // Transform
    createDbWriter(),     // Writable
  );
}

// 错误处理
try {
  await pipeline(readable, transform, writable);
} catch (err) {
  console.error('管道错误:', err);
}
```

### 2.5 AsyncIterable 流

```typescript
import { Readable } from 'node:stream';

// 将流作为异步迭代器使用
async function processStream(readable: Readable) {
  for await (const chunk of readable) {
    // chunk 类型是 Buffer
    console.log(chunk.toString());
  }
}

// 从数组创建流
const data = [{ id: 1 }, { id: 2 }, { id: 3 }];
const stream = Readable.from(data, { objectMode: true });

for await (const item of stream) {
  console.log(item.id); // 类型安全
}
```

## 三、注意事项与常见陷阱

1. **`objectMode: true`**：允许推送任意 JS 对象而非只允许 Buffer/字符串
2. **始终使用 `pipeline`**：自动处理错误和流关闭
3. **转换流的 `flush`**：在流结束时处理剩余数据
4. **不要混用 `objectMode`**：管道中所有流的 `objectMode` 应一致
5. **`readable.destroy()`**：出错时销毁流以释放资源
