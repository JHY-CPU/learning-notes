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


## 演示：Stream 流

点击按钮查看


<!-- Converted from: 10_Stream流.html -->
