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


<!-- Converted from: 11_EventEmitter.html -->
