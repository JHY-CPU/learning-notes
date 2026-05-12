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


## fs.promises 实战：文件工具库

```javascript
// ========== 实用文件工具 ==========
const fs = require('fs/promises');
const path = require('path');

// 确保目录存在
async function ensureDir(dirPath) {
    try {
        await fs.access(dirPath);
    } catch {
        await fs.mkdir(dirPath, { recursive: true });
    }
}

// 读取 JSON 文件
async function readJSON(filePath) {
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
}

// 写入 JSON 文件
async function writeJSON(filePath, data) {
    await ensureDir(path.dirname(filePath));
    await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

// 安全读取 (文件不存在返回默认值)
async function safeRead(filePath, defaultValue = null) {
    try {
        return await fs.readFile(filePath, 'utf-8');
    } catch (err) {
        if (err.code === 'ENOENT') return defaultValue;
        throw err;
    }
}

// 追加写入
async function appendLine(filePath, line) {
    await fs.appendFile(filePath, line + '\n', 'utf-8');
}

// 临时文件创建
async function createTempFile(prefix = 'tmp') {
    const tmpDir = path.join(require('os').tmpdir(), prefix + Date.now());
    await fs.mkdir(tmpDir, { recursive: true });
    return tmpDir;
}
```

## 并发控制：限制同时操作数量

```javascript
// ========== 限制并发数的文件处理 ==========
async function processFilesWithLimit(files, handler, concurrency = 5) {
    const results = [];
    const executing = new Set();

    for (const file of files) {
        const promise = handler(file).then(result => {
            executing.delete(promise);
            return result;
        });
        executing.add(promise);
        results.push(promise);

        if (executing.size >= concurrency) {
            await Promise.race(executing);
        }
    }
    return Promise.all(results);
}

// 使用示例：并发读取多个文件（最多5个同时）
const files = ['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt', 'f.txt'];
const contents = await processFilesWithLimit(
    files,
    async (file) => {
        const data = await fs.readFile(file, 'utf-8');
        return { file, length: data.length };
    },
    3 // 最多3个并发
);
console.log(contents);
```

## watch 高级用法

```javascript
// ========== 递归监听目录变化 ==========
const fs = require('fs');

// Node.js 19+ 支持 recursive 选项
const watcher = fs.watch('./src', { recursive: true }, (event, filename) => {
    console.log(`[${event}] ${filename}`);
    // event: 'rename' (创建/删除/重命名) 或 'change' (修改)
});

// 使用 fs.watchFile (轮询方式，跨平台兼容)
fs.watchFile('./config.json', { interval: 1000 }, (curr, prev) => {
    if (curr.mtime !== prev.mtime) {
        console.log('config.json 已修改');
        // 重新加载配置
    }
});

// 停止监听
// watcher.close();
// fs.unwatchFile('./config.json');

// ========== 防抖处理 ==========
let debounceTimer;
const debouncedWatcher = fs.watch('./src', { recursive: true }, (event, filename) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        console.log(`最终变化: [${event}] ${filename}`);
    }, 300);
});
```

## 错误处理最佳实践

```javascript
// ========== 区分不同错误类型 ==========
async function safeOperation(filePath) {
    try {
        const stat = await fs.stat(filePath);
        return stat;
    } catch (err) {
        switch (err.code) {
            case 'ENOENT':
                console.error('文件不存在:', filePath);
                break;
            case 'EACCES':
                console.error('权限不足:', filePath);
                break;
            case 'EISDIR':
                console.error('路径是目录而非文件:', filePath);
                break;
            default:
                console.error('未知错误:', err.message);
        }
        throw err;
    }
}
```

<!-- Converted from: 7_fs异步操作.html -->
