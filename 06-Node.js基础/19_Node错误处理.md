# Node错误处理


## Node 错误处理


异步错误、uncaughtException/unhandledRejection、错误码、日志。


## Node 错误处理机制


```
// ========== 同步错误 ==========
try {
    JSON.parse('invalid');
} catch (err) {
    console.error('解析错误:', err.message);
}

// ========== 异步错误 (回调) ==========
fs.readFile('/notfound', (err, data) => {
    if (err) return console.error('读取失败:', err);
});

// ========== Promise 错误 ==========
Promise.reject(new Error('失败'))
    .catch(err => console.error(err));

// ========== async/await 错误 ==========
async function read() {
    try {
        await fs.promises.readFile('/notfound');
    } catch (err) {
        console.error(err.code); // 'ENOENT'
    }
}

// ========== 全局未捕获 ==========
process.on('uncaughtException', (err) => {
    console.error('未捕获异常:', err);
    process.exit(1);
});

process.on('unhandledRejection', (reason) => {
    console.error('未处理拒绝:', reason);
});

// ========== 错误码 ==========
// ENOENT — 文件不存在
// EACCES — 权限不足
// EEXIST — 文件已存在
// ENOTDIR — 不是目录
// ECONNREFUSED — 连接被拒
// ETIMEDOUT — 连接超时
```


## 演示：错误处理

点击按钮查看


<!-- Converted from: 19_Node错误处理.html -->
