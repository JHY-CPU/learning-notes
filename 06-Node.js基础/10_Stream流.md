# Stream流


## Stream 流


Readable / Writable、pipe 链式、大文件处理、背压 backpressure。


## Stream 类型


```
// ========== 四种流 ==========
// Readable  — 可读流 (读取数据)
// Writable  — 可写流 (写入数据)
// Duplex    — 双向流 (既是可读也是可写)
// Transform — 转换流 (读写过程中可修改)

// ========== 可读流 Readable ==========
const readable = fs.createReadStream('file.txt', {
    highWaterMark: 64 * 1024, // 每次读取 64KB
    encoding: 'utf-8'
});

readable.on('data', (chunk) => {
    console.log('收到数据:', chunk.length);
});
readable.on('end', () => console.log('读取完成'));
readable.on('error', (err) => console.error(err));

// ========== 可写流 Writable ==========
const writable = fs.createWriteStream('output.txt');
writable.write('hello');
writable.write('world');
writable.end();
writable.on('finish', () => console.log('写入完成'));

// ========== pipe 管道 ==========
// 自动处理背压 (backpressure)
readable.pipe(writable);

// 链式 pipe
readable
    .pipe(transformStream)
    .pipe(writable);

// ========== 背压 ==========
// 当写入速度 > 读取速度时触发
// writable.write() 返回 false 表示需要暂停
// 'drain' 事件表示可以继续写入
```


## 实战：大文件复制

```javascript
// ========== 大文件复制 (自动处理背压) ==========
const fs = require('fs');

function copyFile(src, dest) {
    return new Promise((resolve, reject) => {
        const readable = fs.createReadStream(src);
        const writable = fs.createWriteStream(dest);

        readable.pipe(writable);

        writable.on('finish', resolve);
        writable.on('error', (err) => {
            fs.unlink(dest, () => {}); // 清理不完整文件
            reject(err);
        });
        readable.on('error', (err) => {
            fs.unlink(dest, () => {});
            reject(err);
        });
    });
}

// 使用
await copyFile('./large-video.mp4', './backup/video.mp4');
```

## Transform 流实战

```javascript
// ========== 自定义 Transform 流 ==========
const { Transform } = require('stream');

// CSV → JSON 转换流
class CsvToJsonTransform extends Transform {
    constructor(options) {
        super({ ...options, objectMode: true });
        this.headers = null;
        this.buffer = '';
    }

    _transform(chunk, encoding, callback) {
        this.buffer += chunk.toString();
        const lines = this.buffer.split('\n');
        this.buffer = lines.pop(); // 保留不完整的最后一行

        for (const line of lines) {
            if (!line.trim()) continue;
            const values = line.split(',').map(v => v.trim());
            if (!this.headers) {
                this.headers = values;
            } else {
                const obj = {};
                this.headers.forEach((h, i) => { obj[h] = values[i]; });
                this.push(JSON.stringify(obj) + '\n');
            }
        }
        callback();
    }

    _flush(callback) {
        if (this.buffer.trim() && this.headers) {
            const values = this.buffer.split(',').map(v => v.trim());
            const obj = {};
            this.headers.forEach((h, i) => { obj[h] = values[i]; });
            this.push(JSON.stringify(obj) + '\n');
        }
        callback();
    }
}

// 使用
fs.createReadStream('./data.csv')
    .pipe(new CsvToJsonTransform())
    .pipe(fs.createWriteStream('./data.json'));
```

## 流的高级模式

```javascript
// ========== pipeline (Node 10+，推荐) ==========
const { pipeline } = require('stream');
const { createGzip, createGunzip } = require('zlib');

// 压缩文件
pipeline(
    fs.createReadStream('./input.txt'),
    createGzip(),
    fs.createWriteStream('./input.txt.gz'),
    (err) => {
        if (err) console.error('压缩失败:', err);
        else console.log('压缩完成');
    }
);

// ========== async iterator 遍历流 (Node 10+) ==========
async function readLines(filePath) {
    const readable = fs.createReadStream(filePath, { encoding: 'utf-8' });
    let buffer = '';

    for await (const chunk of readable) {
        buffer += chunk;
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
            console.log(line);
        }
    }
    if (buffer) console.log(buffer);
}

// ========== 封装可读流 ==========
const { Readable } = require('stream');

function createNumberStream(max) {
    let current = 0;
    return new Readable({
        read() {
            if (current < max) {
                this.push(String(current++));
            } else {
                this.push(null); // 结束
            }
        }
    });
}

createNumberStream(5).pipe(process.stdout);
// 输出: 01234
```

## 性能建议

- **大文件务必使用 Stream**：一次性读入 Buffer 会消耗大量内存
- **合理设置 highWaterMark**：默认 64KB，网络传输可调小，磁盘读写可调大
- **使用 pipeline 替代手动 pipe**：自动处理错误传播和流关闭
- **对象模式**：处理非二进制数据时使用 `objectMode: true`
- **避免背压忽略**：不要在 `write()` 返回 false 时继续写入

<!-- Converted from: 10_Stream流.html -->
