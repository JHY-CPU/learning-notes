# Worker通信

## 一、概念说明

WASM 可以在 Web Worker 中运行，避免阻塞主线程。Worker 与主线程通过 `postMessage` 通信。

```javascript
// 主线程
const worker = new Worker('worker.js');
worker.postMessage({ type: 'init', wasmUrl: 'module.wasm' });
worker.onmessage = (e) => {
    console.log('结果:', e.data);
};
```

```javascript
// worker.js
let wasmInstance;
self.onmessage = async (e) => {
    if (e.data.type === 'init') {
        const resp = await fetch(e.data.wasmUrl);
        const bytes = await resp.arrayBuffer();
        const { instance } = await WebAssembly.instantiate(bytes);
        wasmInstance = instance;
    }
};
```

## 二、具体用法

### 2.1 SharedArrayBuffer 共享内存

```javascript
// 主线程创建共享内存
const sharedBuffer = new SharedArrayBuffer(1024);
const worker = new Worker('worker.js');
worker.postMessage({ buffer: sharedBuffer });

// 写入数据
const view = new Uint32Array(sharedBuffer);
Atomics.store(view, 0, 42);
```

```javascript
// worker.js 接收共享内存
self.onmessage = (e) => {
    const view = new Uint32Array(e.data.buffer);
    const value = Atomics.load(view, 0);
    // 使用 WASM 处理共享数据
    wasmInstance.exports.process_shared(0);
};
```

### 2.2 Transferable 对象

```javascript
// 传输大数组（零拷贝）
const buffer = new ArrayBuffer(1024 * 1024);
worker.postMessage({ buffer: buffer }, [buffer]);
// buffer 在主线程中已不可用

// WASM 内存传输
const wasmMemory = new WebAssembly.Memory({ initial: 256 });
worker.postMessage({ memory: wasmMemory }, [wasmMemory.buffer]);
```

### 2.3 Rust 端 Worker 封装

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Worker, DedicatedWorkerGlobalScope};

#[wasm_bindgen]
pub fn worker_entry_point() {
    let global = js_sys::global();
    let scope: DedicatedWorkerGlobalScope = global.dyn_into().unwrap();

    scope.set_onmessage(Some(&Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        // 处理消息，调用 WASM 函数
        let result = process_data(&data);
        scope.post_message(&JsValue::from_str(&result)).unwrap();
    }) as Box<dyn FnMut(_)>).into()));
}
```

### 2.4 线程池模式

```javascript
class WasmThreadPool {
    constructor(size, wasmUrl) {
        this.workers = [];
        this.taskQueue = [];
        for (let i = 0; i < size; i++) {
            const worker = new Worker('worker.js');
            worker.onmessage = (e) => this.onResult(e.data);
            worker.postMessage({ type: 'init', wasmUrl });
            this.workers.push({ worker, busy: false });
        }
    }

    execute(task) {
        const free = this.workers.find(w => !w.busy);
        if (free) {
            free.busy = true;
            free.worker.postMessage(task);
        } else {
            this.taskQueue.push(task);
        }
    }

    onResult(result) {
        // 释放 Worker 并处理队列
    }
}
```

## 三、注意事项与常见陷阱

1. **跨域限制**：SharedArrayBuffer 需要 `Cross-Origin-Opener-Policy` 和 `Cross-Origin-Embedder-Policy` 头
2. **序列化开销**：postMessage 会序列化数据，大对象考虑使用 Transferable
3. **内存隔离**：Worker 内存独立，不能直接共享 WASM 实例
4. **Atomics 同步**：共享内存需要使用 Atomics API 进行同步
5. **生命周期管理**：Worker 不会自动终止，需要手动管理
