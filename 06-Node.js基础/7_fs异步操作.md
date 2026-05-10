# fs异步操作


## fs 异步操作


fs.readFile 回调、fs.promises API、watch 文件变化、错误处理。


## fs 异步 API


```
// ========== 回调版 (fs.readFile) ==========
fs.readFile('/path/file.txt', 'utf-8', (err, data) => {
    if (err) {
        console.error('读取失败:', err);
        return;
    }
    console.log(data);
});

// ========== Promise 版 (fs.promises) ==========
// Node 10+ 推荐
const fs = require('fs/promises');
// 或: const { readFile } = require('fs').promises;

async function readConfig() {
    try {
        const data = await fs.readFile('/path/config.json', 'utf-8');
        return JSON.parse(data);
    } catch (err) {
        console.error('读取失败:', err);
    }
}

// ========== 常用异步操作 ==========
await fs.readFile(path, options);
await fs.writeFile(path, data, options);
await fs.appendFile(path, data);
await fs.mkdir(path, { recursive: true });
await fs.readdir(path);
await fs.unlink(path);
await fs.rm(path, { recursive: true });
await fs.stat(path);
await fs.access(path); // 检查权限

// ========== watch 文件变化 ==========
const watcher = fs.watch('/path/to/dir', (event, filename) => {
    console.log(`${filename}: ${event}`);
});
watcher.close(); // 停止监听
```


## 演示：异步文件操作

点击按钮查看


<!-- Converted from: 7_fs异步操作.html -->
