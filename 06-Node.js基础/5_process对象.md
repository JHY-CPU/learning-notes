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


<!-- Converted from: 5_process对象.html -->
