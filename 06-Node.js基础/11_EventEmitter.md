# EventEmitter


## EventEmitter


on/emit/once/off/listeners、事件参数、错误事件、继承 EventEmitter。


## EventEmitter API


```
// ========== EventEmitter 核心 ==========
const EventEmitter = require('events');
const emitter = new EventEmitter();

// 注册事件
emitter.on('event', handler);        // 监听
emitter.once('event', handler);      // 只触发一次
emitter.addListener('event', handler); // 同 on

// 触发事件
emitter.emit('event', arg1, arg2);

// 移除事件
emitter.off('event', handler);       // 移除
emitter.removeListener('event', handler); // 同 off
emitter.removeAllListeners('event'); // 移除所有

// 其他方法
emitter.listeners('event');   // 获取所有监听器
emitter.listenerCount('event'); // 监听器数量
emitter.eventNames();         // 所有事件名
emitter.rawListeners('event'); // 原始监听器

// ========== 错误事件 ==========
// 特殊事件: 'error'
// 如果 emit('error') 没有监听器，会抛出异常
emitter.on('error', (err) => {
    console.error('错误:', err.message);
});

// ========== 继承 EventEmitter ==========
class MyClass extends EventEmitter {
    constructor() { super(); }
    doSomething() {
        this.emit('data', 'result');
    }
}

// ========== 事件参数传递 ==========
emitter.emit('user:login', { id: 1, name: 'Alice' });
emitter.on('user:login', (user) => {
    console.log(`用户 ${user.name} 登录`);
});
```


## 演示：EventEmitter

点击按钮查看


## 实战：简易事件总线

```javascript
// ========== 事件总线 (发布/订阅模式) ==========
const EventEmitter = require('events');

class EventBus extends EventEmitter {
    constructor() {
        super();
        this.setMaxListeners(20); // 提高监听器上限
    }

    // 带命名空间的事件
    emitNS(namespace, event, ...args) {
        this.emit(`${namespace}:${event}`, ...args);
    }

    onNS(namespace, event, handler) {
        this.on(`${namespace}:${event}`, handler);
        return () => this.off(`${namespace}:${event}`, handler); // 返回取消函数
    }

    // 等待事件触发 (返回 Promise)
    waitFor(event, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.off(event, handler);
                reject(new Error(`等待 ${event} 超时`));
            }, timeout);

            const handler = (...args) => {
                clearTimeout(timer);
                resolve(args);
            };
            this.once(event, handler);
        });
    }

    // 异步串行触发所有监听器
    async emitSerial(event, ...args) {
        const listeners = this.listeners(event);
        for (const listener of listeners) {
            await listener(...args);
        }
    }
}

// 使用
const bus = new EventBus();
const unsubscribe = bus.onNS('user', 'login', (user) => {
    console.log(`${user.name} 登录了`);
});
bus.emitNS('user', 'login', { name: 'Alice' });
unsubscribe(); // 取消订阅
```

## EventEmitter 与 Node.js 内部

```javascript
// ========== Node.js 内部大量使用 EventEmitter ==========
// http.Server → 继承 EventEmitter
const server = http.createServer();
server.on('request', (req, res) => {});
server.on('error', (err) => {});
server.on('close', () => {});

// stream.Readable → 继承 EventEmitter
const readable = fs.createReadStream('./file.txt');
readable.on('data', (chunk) => {});
readable.on('end', () => {});
readable.on('error', (err) => {});

// child_process → 继承 EventEmitter
const child = fork('./worker.js');
child.on('message', (msg) => {});
child.on('exit', (code) => {});
```

## 内存泄漏防范

```javascript
// ========== 监听器泄漏检测 ==========
const emitter = new EventEmitter();
emitter.setMaxListeners(10); // 默认限制 10 个

// 检测潜在泄漏
process.on('warning', (warning) => {
    if (warning.name === 'MaxListenersExceededWarning') {
        console.error('监听器数量超过上限，可能存在泄漏!');
        console.error(warning);
    }
});

// ========== 正确的清理模式 ==========
class Database extends EventEmitter {
    constructor() {
        super();
        this._onTimeout = this._onTimeout.bind(this);
    }

    connect() {
        this._timer = setInterval(this._onTimeout, 1000);
    }

    disconnect() {
        clearInterval(this._timer);
        this.removeAllListeners(); // 清理所有监听器
    }

    _onTimeout() {
        this.emit('tick');
    }
}

// 使用时注意清理
const db = new Database();
db.on('tick', handler);
// 不再使用时: db.disconnect();
```

## 异步事件处理

```javascript
// ========== 串行处理事件 ==========
class AsyncEventEmitter extends EventEmitter {
    async emitAsync(event, ...args) {
        const listeners = this.listeners(event);
        const results = [];
        for (const listener of listeners) {
            results.push(await listener(...args));
        }
        return results;
    }
}

// ========== 并行处理事件 ==========
async function emitParallel(emitter, event, ...args) {
    const listeners = emitter.listeners(event);
    return Promise.all(listeners.map(fn => fn(...args)));
}

// ========== 事件过滤 ==========
function onFiltered(emitter, event, predicate, handler) {
    emitter.on(event, (...args) => {
        if (predicate(...args)) handler(...args);
    });
}
// 只处理特定用户事件
onFiltered(emitter, 'user:update', (user) => user.role === 'admin', (user) => {
    console.log('管理员更新:', user);
});
```

<!-- Converted from: 11_EventEmitter.html -->
