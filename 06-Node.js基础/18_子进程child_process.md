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


<!-- Converted from: 18_子进程child_process.html -->
