# Worker线程类型

## 一、概念说明

Node.js 的 `worker_threads` 模块允许在多线程中执行 CPU 密集型任务。TypeScript 可以类型化 Worker 之间的消息传递，确保主线程和工作线程之间的通信安全。

## 二、具体用法

### 2.1 基本 Worker 类型

```typescript
// worker.ts — 工作线程
import { parentPort, workerData } from 'node:worker_threads';

// workerData 的类型
interface WorkerInput {
  data: number[];
  operation: 'sum' | 'average' | 'max';
}

const { data, operation } = workerData as WorkerInput;

let result: number;
switch (operation) {
  case 'sum':
    result = data.reduce((a, b) => a + b, 0);
    break;
  case 'average':
    result = data.reduce((a, b) => a + b, 0) / data.length;
    break;
  case 'max':
    result = Math.max(...data);
    break;
}

parentPort?.postMessage(result);
```

```typescript
// main.ts — 主线程
import { Worker } from 'node:worker_threads';

function runWorker(data: number[], operation: 'sum' | 'average' | 'max'): Promise<number> {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./worker.js', {
      workerData: { data, operation } satisfies WorkerInput,
    });

    worker.on('message', (result: number) => resolve(result));
    worker.on('error', reject);
    worker.on('exit', (code) => {
      if (code !== 0) reject(new Error(`Worker 退出码: ${code}`));
    });
  });
}

// 使用
const sum = await runWorker([1, 2, 3, 4, 5], 'sum'); // number
console.log(sum); // 15
```

### 2.2 类型安全的消息传递

```typescript
// 定义消息协议
type WorkerMessage =
  | { type: 'compute'; payload: { numbers: number[] } }
  | { type: 'terminate' };

type WorkerResponse =
  | { type: 'result'; payload: number }
  | { type: 'error'; payload: string }
  | { type: 'progress'; payload: number };

// worker.ts
parentPort?.on('message', (msg: WorkerMessage) => {
  switch (msg.type) {
    case 'compute':
      const result = msg.payload.numbers.reduce((a, b) => a + b, 0);
      parentPort?.postMessage({ type: 'result', payload: result } satisfies WorkerResponse);
      break;
    case 'terminate':
      process.exit(0);
  }
});
```

### 2.3 Worker Pool 类型

```typescript
import { Worker } from 'node:worker_threads';

class WorkerPool<TInput, TOutput> {
  private workers: Worker[] = [];
  private queue: Array<{
    input: TInput;
    resolve: (value: TOutput) => void;
    reject: (reason: unknown) => void;
  }> = [];
  private activeWorkers = 0;

  constructor(
    private workerScript: string,
    private maxWorkers: number
  ) {}

  async execute(input: TInput): Promise<TOutput> {
    return new Promise((resolve, reject) => {
      if (this.activeWorkers < this.maxWorkers) {
        this.runWorker(input, resolve, reject);
      } else {
        this.queue.push({ input, resolve, reject });
      }
    });
  }

  private runWorker(
    input: TInput,
    resolve: (value: TOutput) => void,
    reject: (reason: unknown) => void
  ) {
    this.activeWorkers++;
    const worker = new Worker(this.workerScript, {
      workerData: input,
    });

    worker.on('message', (result: TOutput) => {
      resolve(result);
      this.activeWorkers--;
      this.processQueue();
    });

    worker.on('error', (err) => {
      reject(err);
      this.activeWorkers--;
      this.processQueue();
    });
  }

  private processQueue() {
    if (this.queue.length > 0 && this.activeWorkers < this.maxWorkers) {
      const next = this.queue.shift()!;
      this.runWorker(next.input, next.resolve, next.reject);
    }
  }
}

// 使用
const pool = new WorkerPool<number[], number>('./worker.js', 4);
const results = await Promise.all([
  pool.execute([1, 2, 3]),
  pool.execute([4, 5, 6]),
]);
```

### 2.4 SharedArrayBuffer 类型

```typescript
import { Worker, isMainThread, parentPort } from 'node:worker_threads';

// 共享内存 — 需要 --harmony-sharedarraybuffer
if (isMainThread) {
  const sharedBuffer = new SharedArrayBuffer(4);
  const sharedArray = new Int32Array(sharedBuffer);

  const worker = new Worker(__filename, { workerData: { sharedBuffer } });

  Atomics.wait(sharedArray, 0, 0); // 等待 worker 写入
  console.log(sharedArray[0]);
} else {
  const { sharedBuffer } = workerData as { sharedBuffer: SharedArrayBuffer };
  const sharedArray = new Int32Array(sharedBuffer);
  Atomics.store(sharedArray, 0, 42);
  Atomics.notify(sharedArray, 0);
}
```

## 三、注意事项与常见陷阱

1. **Worker 数据需要可序列化**：不能传递函数、类实例等
2. **`__filename` 在 ESM 中不同**：使用 `import.meta.url` 替代
3. **Worker Pool 避免线程爆炸**：限制最大 worker 数量
4. **SharedArrayBuffer 需要特殊 HTTP 头**：`Cross-Origin-Opener-Policy`
5. **Worker 中的错误不会传播到主线程**：必须通过消息传递
