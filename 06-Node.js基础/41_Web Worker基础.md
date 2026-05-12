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


## 什么是 Web Worker

Web Worker 允许在后台线程运行JS，不阻塞主线程UI渲染。解决JavaScript单线程执行耗时任务导致页面卡顿的问题。

## 工作原理

主线程创建Worker对象，通过 `postMessage` 发送数据（结构化克隆），Worker在独立线程处理后通过 `onmessage` 返回结果。Worker无法访问DOM、window、document。

## 使用限制

- 不能操作DOM和BOM（无window/document对象）
- 不能访问主线程的变量（数据通过消息传递）
- 同源策略限制（Worker脚本必须同源）
- 文件协议 `file://` 下部分浏览器不支持

## 常见场景

1. **大数据计算**：图像处理、加密解密、数学运算
2. **数据解析**：大JSON解析、CSV处理
3. **轮询/定时任务**：WebSocket心跳、后台数据同步
4. **Service Worker**：离线缓存、推送通知（特殊的Worker）

## 性能考量

- 创建Worker有开销，不宜频繁创建销毁，建议复用
- 消息传递有结构化克隆开销，大数据用Transferable对象（`postMessage(data, [buffer])`）零拷贝传递
- 共享Worker（SharedWorker）可被多个页面共享，但兼容性差
- Worker数量受CPU核心数限制

<!-- Converted from: 41_Web Worker基础.html -->
