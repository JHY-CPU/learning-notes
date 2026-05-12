# 子进程child_process


## 子进程 child_process


spawn/exec/execFile/fork、进程通信、stdin/stdout 管道。


## child_process 模块


```
// ========== spawn ==========
// 启动子进程执行命令 (流式输出)
const { spawn } = require('child_process');
const ls = spawn('ls', ['-lh', '/usr']);
ls.stdout.on('data', (data) => console.log(data.toString()));
ls.stderr.on('data', (data) => console.error(data.toString()));
ls.on('close', (code) => console.log(`exit: ${code}`));

// ========== exec ==========
// 执行命令 (缓冲输出)
const { exec } = require('child_process');
exec('ls -lh /usr', (err, stdout, stderr) => {
    if (err) return console.error(err);
    console.log(stdout);
});

// ========== execFile ==========
// 直接执行文件 (不经过shell)
const { execFile } = require('child_process');
execFile('node', ['--version'], (err, stdout) => {
    console.log(stdout);
});

// ========== fork ==========
// 创建 Node 子进程 (有IPC通信)
const { fork } = require('child_process');
const child = fork('./worker.js');
child.send({ msg: 'hello' });
child.on('message', (data) => console.log(data));

// ========== 方法对比 ==========
// spawn  — 流式, 大输出, 不创建shell
// exec   — 缓冲, 小输出, 创建shell
// execFile — 缓冲, 不创建shell
// fork   — Node进程, IPC通信
```


## 演示：子进程

点击按钮查看


## 实战：进程池

```javascript
// ========== 简易进程池 ==========
const { fork } = require('child_process');
const os = require('os');

class WorkerPool {
    constructor(workerScript, size = os.cpus().length) {
        this.workers = [];
        this.taskQueue = [];
        this.workerScript = workerScript;

        for (let i = 0; i < size; i++) {
            this._createWorker();
        }
    }

    _createWorker() {
        const worker = fork(this.workerScript);
        worker.busy = false;

        worker.on('message', (result) => {
            worker.busy = false;
            if (worker.currentResolve) {
                worker.currentResolve(result);
                worker.currentResolve = null;
            }
            this._processQueue();
        });

        worker.on('exit', () => {
            // 工作进程意外退出时重启
            const idx = this.workers.indexOf(worker);
            if (idx !== -1) this.workers.splice(idx, 1);
            this._createWorker();
        });

        this.workers.push(worker);
    }

    _processQueue() {
        if (this.taskQueue.length === 0) return;
        const available = this.workers.find(w => !w.busy);
        if (available) {
            const task = this.taskQueue.shift();
            this._runTask(available, task);
        }
    }

    _runTask(worker, task) {
        worker.busy = true;
        worker.currentResolve = task.resolve;
        worker.send(task.data);
    }

    exec(data) {
        return new Promise((resolve) => {
            const task = { data, resolve };
            const available = this.workers.find(w => !w.busy);
            if (available) {
                this._runTask(available, task);
            } else {
                this.taskQueue.push(task);
            }
        });
    }

    destroy() {
        this.workers.forEach(w => w.kill());
    }
}

// worker.js
// process.on('message', (data) => {
//     const result = heavyComputation(data);
//     process.send(result);
// });

// main.js
// const pool = new WorkerPool('./worker.js', 4);
// const result = await pool.exec({ n: 1000000 });
```

## spawn 高级用法

```javascript
// ========== spawn 进阶 ==========
const { spawn } = require('child_process');

// 执行带管道的命令: cat file.txt | grep "error" | wc -l
const cat = spawn('cat', ['file.txt']);
const grep = spawn('grep', ['error']);
const wc = spawn('wc', ['-l']);

cat.stdout.pipe(grep.stdin);
grep.stdout.pipe(wc.stdin);

wc.stdout.on('data', (data) => {
    console.log(`错误行数: ${data.toString().trim()}`);
});

// ========== 环境变量与工作目录 ==========
const child = spawn('node', ['script.js'], {
    env: { ...process.env, NODE_ENV: 'production', DEBUG: 'true' },
    cwd: '/path/to/project',
    stdio: ['pipe', 'pipe', 'pipe'], // stdin, stdout, stderr
    timeout: 30000, // 30秒超时
});

// ========== 超时控制 ==========
function spawnWithTimeout(command, args, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args);
        let output = '';
        let killed = false;

        const timer = setTimeout(() => {
            child.kill('SIGTERM');
            killed = true;
        }, timeout);

        child.stdout.on('data', (data) => { output += data; });
        child.on('close', (code) => {
            clearTimeout(timer);
            if (killed) reject(new Error(`命令超时 (${timeout}ms)`));
            else resolve({ code, output });
        });
    });
}
```

## Worker Threads vs child_process

```javascript
// ========== 对比 ==========
// child_process:
//   - 独立的进程，有自己的内存空间
//   - 通过 IPC 通信 (序列化/反序列化)
//   - 适合：运行外部命令、CPU 密集但需独立环境
//
// worker_threads:
//   - 同一进程内的线程
//   - 可以共享内存 (SharedArrayBuffer)
//   - 适合：CPU 密集计算、需要共享数据

// ========== Worker Threads 示例 ==========
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

if (isMainThread) {
    // 主线程
    const worker = new Worker(__filename, { workerData: { n: 1000000 } });
    worker.on('message', (result) => console.log('结果:', result));
    worker.on('error', (err) => console.error(err));
} else {
    // 工作线程
    let sum = 0;
    for (let i = 0; i < workerData.n; i++) sum += i;
    parentPort.postMessage(sum);
}
```

<!-- Converted from: 18_子进程child_process.html -->
