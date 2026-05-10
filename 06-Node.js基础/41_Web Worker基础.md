# Web Worker基础


## Web Worker 基础


Worker 创建、postMessage 通信、onmessage、terminate、限制。


## Web Worker API


```
// ========== 创建 Worker ==========
const worker = new Worker('worker.js');

// ========== 发送消息 ==========
worker.postMessage({ type: 'compute', data: 42 });

// ========== 接收消息 ==========
worker.onmessage = (e) => {
    console.log('收到:', e.data);
};

// ========== 错误处理 ==========
worker.onerror = (e) => {
    console.error('Worker 错误:', e.message);
};

// ========== 终止 Worker ==========
worker.terminate();

// ========== Worker 内部 (self) ==========
// self.onmessage / self.postMessage
// importScripts('lib.js')
// 注意: 不能访问 DOM / window / document
```


## 演示：Web Worker

点击按钮查看


<!-- Converted from: 41_Web Worker基础.html -->
