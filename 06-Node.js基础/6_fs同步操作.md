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


<!-- Converted from: 6_fs同步操作.html -->
