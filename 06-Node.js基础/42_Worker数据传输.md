# Worker数据传输


## Worker 数据传输


结构化克隆、Transferable Objects、SharedArrayBuffer、Worker 池。


## 数据传输方式


```
// ========== 结构化克隆 (默认) ==========
// 支持: Object, Array, Map, Set, Blob, File, ImageData
// 不支持: DOM 元素, 函数, Error, Symbol
// 特点: 深拷贝, 大数据量慢

// ========== Transferable Objects (零拷贝) ==========
// ArrayBuffer, MessagePort, ImageBitmap, OffscreenCanvas
// 传输后源对象被"清空" (detached)
const buffer = new ArrayBuffer(1024);
worker.postMessage(buffer, [buffer]); // 第二个参数
// buffer.byteLength === 0 (已转移)

// ========== SharedArrayBuffer (共享内存) ==========
const shared = new SharedArrayBuffer(1024);
const view = new Int32Array(shared);
// 多个 Worker 共享同一内存
// 需要 Atomics 进行同步

// ========== Atomics 操作 ==========
Atomics.add(view, 0, 1);  // 原子加
Atomics.load(view, 0);    // 原子读
Atomics.store(view, 0, 5); // 原子写
Atomics.wait(view, 0, 0);  // 等待 (Worker 内)
Atomics.notify(view, 0, 1); // 通知
```


## 演示：数据传输

点击按钮查看


## Worker 池与实战

```javascript
// ========== Worker 池 ==========
class WorkerPool {
    constructor(script, size = navigator.hardwareConcurrency || 4) {
        this.workers = [];
        this.queue = [];
        this.activeWorkers = 0;

        for (let i = 0; i < size; i++) {
            const worker = new Worker(script);
            worker.available = true;
            worker.onmessage = (e) => {
                worker.available = true;
                this.activeWorkers--;
                if (worker.resolve) worker.resolve(e.data);
                this._processQueue();
            };
            this.workers.push(worker);
        }
    }

    exec(data) {
        return new Promise((resolve) => {
            const task = { data, resolve };
            const worker = this.workers.find(w => w.available);
            if (worker) {
                this._runTask(worker, task);
            } else {
                this.queue.push(task);
            }
        });
    }

    _runTask(worker, task) {
        worker.available = false;
        worker.resolve = task.resolve;
        this.activeWorkers++;
        worker.postMessage(task.data);
    }

    _processQueue() {
        if (this.queue.length === 0) return;
        const worker = this.workers.find(w => w.available);
        if (worker) this._runTask(worker, this.queue.shift());
    }

    terminate() {
        this.workers.forEach(w => w.terminate());
    }
}

// ========== SharedArrayBuffer 原子操作 ==========
// main.js
const shared = new SharedArrayBuffer(4); // 4字节
const view = new Int32Array(shared);
view[0] = 0;

const worker1 = new Worker('counter.js');
const worker2 = new Worker('counter.js');
worker1.postMessage(shared);
worker2.postMessage(shared);

// counter.js
self.onmessage = (e) => {
    const view = new Int32Array(e.data);
    for (let i = 0; i < 100000; i++) {
        Atomics.add(view, 0, 1); // 原子加1
    }
    self.postMessage('done');
};

// 等待所有 worker 完成后
// view[0] === 200000 (保证正确性)

// ========== 流式数据传输 ==========
async function streamToWorker(file, worker) {
    const stream = file.stream();
    const reader = stream.getReader();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        // 使用 Transferable 零拷贝传输
        worker.postMessage(value.buffer, [value.buffer]);
    }
    worker.postMessage('DONE');
}
```

<!-- Converted from: 42_Worker数据传输.html -->
