# fs同步操作


## fs 同步操作


readFileSync / writeFileSync / existsSync / mkdirSync / unlinkSync / rmdirSync。


## fs 同步 API


```
// ========== fs 同步方法 (Sync) ==========
// ⚠️ 同步方法会阻塞主线程
// 适合: 启动时初始化、脚本、CLI 工具
// 不适合: 高并发服务器

const fs = require('fs');

// ========== 文件读取 ==========
// 读文件 (返回 Buffer 或字符串)
const data = fs.readFileSync('/path/file.txt', 'utf-8');
const buffer = fs.readFileSync('/path/file.txt'); // Buffer

// ========== 文件写入 ==========
// 写文件 (覆盖)
fs.writeFileSync('/path/file.txt', 'content', 'utf-8');

// 追加
fs.appendFileSync('/path/file.txt', '追加内容');

// ========== 目录操作 ==========
const exists = fs.existsSync('/path');        // 是否存在
fs.mkdirSync('/path/dir', { recursive: true }); // 创建目录
fs.rmdirSync('/path/dir');                    // 删除空目录
fs.rmSync('/path/dir', { recursive: true });  // 递归删除 (Node 14+)

// ========== 文件信息 ==========
const stats = fs.statSync('/path/file.txt');
stats.isFile()           // 是否文件
stats.isDirectory()      // 是否目录
stats.size              // 文件大小
stats.mtime             // 修改时间

// ========== 删除/重命名 ==========
fs.unlinkSync('/path/file.txt');  // 删除文件
fs.renameSync('/old', '/new');    // 重命名/移动

// ========== 读取目录 ==========
const files = fs.readdirSync('/path'); // 文件列表
```


> **Note:** ⚠️ 同步操作会阻塞事件循环，不适合生产服务器。
>
>
> 💡 使用 fs.promises 异步版本获得更好性能。
>
>
> 💡 对于 CLI 工具和初始化脚本，同步方法更方便。


## 演示：模拟文件系统

点击按钮查看


## 实战：配置文件加载器

```javascript
// ========== 同步加载配置 (适合启动阶段) ==========
const fs = require('fs');
const path = require('path');

function loadConfig(configDir) {
    const config = {};

    // 确保目录存在
    if (!fs.existsSync(configDir)) {
        fs.mkdirSync(configDir, { recursive: true });
    }

    // 读取所有 .json 配置文件
    const files = fs.readdirSync(configDir).filter(f => f.endsWith('.json'));

    for (const file of files) {
        const name = path.basename(file, '.json');
        const filePath = path.join(configDir, file);
        try {
            const content = fs.readFileSync(filePath, 'utf-8');
            config[name] = JSON.parse(content);
        } catch (err) {
            console.error(`配置文件 ${file} 读取失败:`, err.message);
        }
    }
    return config;
}

// 使用
const config = loadConfig('./config');
console.log(config);
// { database: { host: 'localhost', port: 5432 }, server: { port: 3000 } }
```

## 文件复制与目录遍历

```javascript
// ========== 同步复制文件 ==========
function copyFileSync(src, dest) {
    // 确保目标目录存在
    const destDir = path.dirname(dest);
    if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
    }
    fs.copyFileSync(src, dest);
}

// ========== 递归复制目录 ==========
function copyDirSync(src, dest) {
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }

    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);

        if (entry.isDirectory()) {
            copyDirSync(srcPath, destPath);
        } else {
            fs.copyFileSync(srcPath, destPath);
        }
    }
}

// ========== 递归遍历目录 ==========
function walkDir(dir, results = []) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            walkDir(fullPath, results);
        } else {
            results.push(fullPath);
        }
    }
    return results;
}

const allFiles = walkDir('./src');
console.log(allFiles);
```

## 性能与使用场景

| 场景 | 是否适合同步 | 说明 |
|------|------------|------|
| CLI 工具启动 | ✅ 适合 | 执行完就退出，阻塞无影响 |
| 读取配置文件 | ✅ 适合 | 启动时加载一次 |
| 单元测试 | ✅ 适合 | 简化测试代码 |
| HTTP 服务器请求处理 | ❌ 不适合 | 会阻塞所有请求 |
| 大文件读写 | ❌ 不适合 | 应使用 Stream |
| 高并发场景 | ❌ 不适合 | 应使用异步 API |

<!-- Converted from: 6_fs同步操作.html -->
