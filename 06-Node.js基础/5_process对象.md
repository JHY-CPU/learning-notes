# process对象


## process 对象


process.argv 命令行参数、env 环境变量、cwd/chdir、exit、nextTick。


## process 核心属性和方法


```
// ========== 命令行参数 ==========
// process.argv — 返回数组
// $ node script.js --name Alice --verbose
// argv[0] = node 路径
// argv[1] = 脚本路径
// argv[2+] = 参数

// ========== 环境变量 ==========
process.env.NODE_ENV        // 'development' / 'production'
process.env.PATH            // 系统 PATH
// .env 文件通常通过 dotenv 加载
// require('dotenv').config()
// process.env.DB_HOST

// ========== 工作目录 ==========
process.cwd()    // 当前工作目录
process.chdir('/path')  // 切换工作目录

// ========== 进程控制 ==========
process.exit(0)      // 正常退出
process.exit(1)      // 异常退出
process.exitCode = 1 // 设置退出码但不退出
process.kill(pid)    // 发送信号

// ========== 事件 ==========
process.on('exit', (code) => {
    // 进程退出时
});
process.on('uncaughtException', (err) => {
    // 未捕获的异常
});
process.on('unhandledRejection', (reason) => {
    // 未处理的 Promise 拒绝
});

// ========== process.nextTick ==========
// 在当前操作完成后、下一事件循环前执行
process.nextTick(() => {
    console.log('下一 tick');
});
// nextTick 优先级高于 Promise.then
```


## 演示：process 功能

点击按钮查看


## process.argv 高级解析

```javascript
// ========== 完整参数解析示例 ==========
// $ node app.js --name Alice --port 3000 -v file1.txt file2.txt

function parseArgs(argv) {
    const args = argv.slice(2);
    const options = {};
    const positional = [];

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg.startsWith('--')) {
            const key = arg.slice(2);
            const next = args[i + 1];
            if (next && !next.startsWith('-')) {
                options[key] = next;
                i++;
            } else {
                options[key] = true;
            }
        } else if (arg.startsWith('-') && arg.length === 2) {
            options[arg.slice(1)] = true;
        } else {
            positional.push(arg);
        }
    }
    return { options, positional };
}

const { options, positional } = parseArgs(process.argv);
console.log(options);    // { name: 'Alice', port: '3000', v: true }
console.log(positional); // ['file1.txt', 'file2.txt']
```

## process 内存与性能

```javascript
// ========== 内存使用 ==========
const mem = process.memoryUsage();
console.log({
    rss: `${(mem.rss / 1024 / 1024).toFixed(2)} MB`,        // 常驻内存
    heapTotal: `${(mem.heapTotal / 1024 / 1024).toFixed(2)} MB`, // 堆总大小
    heapUsed: `${(mem.heapUsed / 1024 / 1024).toFixed(2)} MB`,   // 已用堆内存
    external: `${(mem.external / 1024 / 1024).toFixed(2)} MB`,   // C++ 对象内存
    arrayBuffers: `${(mem.arrayBuffers / 1024 / 1024).toFixed(2)} MB`
});

// ========== CPU 使用 ==========
const startUsage = process.cpuUsage();
// ... 执行一些操作 ...
const endUsage = process.cpuUsage(startUsage);
console.log(`用户CPU: ${endUsage.user / 1000}ms`);
console.log(`系统CPU: ${endUsage.system / 1000}ms`);

// ========== 运行时间 ==========
const startTime = process.hrtime.bigint();
// ... 执行操作 ...
const endTime = process.hrtime.bigint();
console.log(`耗时: ${Number(endTime - startTime) / 1e6}ms`);
```

## 进程信号处理

```javascript
// ========== 优雅关闭 ==========
const server = require('http').createServer();

function gracefulShutdown(signal) {
    console.log(`收到 ${signal}，正在优雅关闭...`);
    server.close(() => {
        console.log('HTTP 服务器已关闭');
        // 关闭数据库连接等资源
        process.exit(0);
    });

    // 超时强制退出
    setTimeout(() => {
        console.error('强制退出');
        process.exit(1);
    }, 10000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ========== 常见信号 ==========
// SIGINT  — Ctrl+C 中断
// SIGTERM — kill 命令默认信号 (请求终止)
// SIGHUP  — 终端关闭 (常用于重新加载配置)
// SIGKILL — 强制终止 (无法捕获)
```

## nextTick 与微任务

```javascript
// ========== 事件循环中的优先级 ==========
// nextTick 回调 > Promise 回调 > setTimeout 回调

console.log('1: 同步开始');

Promise.resolve().then(() => console.log('3: Promise.then'));

process.nextTick(() => console.log('2: nextTick'));

setTimeout(() => console.log('4: setTimeout'), 0);

console.log('1.5: 同步结束');

// 输出顺序:
// 1: 同步开始
// 1.5: 同步结束
// 2: nextTick
// 3: Promise.then
// 4: setTimeout

// ⚠️ 滥用 nextTick 可能导致 I/O 饥饿 (事件循环无法进入 poll 阶段)
```

<!-- Converted from: 5_process对象.html -->
