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


<!-- Converted from: 42_Worker数据传输.html -->
